"""Embedding transform implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from autoencoders import load_model
from autoencoders.data.base import TensorSpec, create_dataloaders, split_dataset
from autoencoders.data.embeddings import EmbeddingMatrix, EmbeddingTensorDataset
from autoencoders.training.display import style
from autoencoders.training import (
    AETrainer,
    AdversarialAutoencoderTrainer,
    AdversarialAutoencoderTrainingConfig,
    FactorVAETrainer,
    FactorVariationalAutoencoderTrainingConfig,
    TrainingConfig,
    VAETrainer,
    VQTrainer,
    resolve_device,
    set_seed,
)

from .config import TransformConfig

_VAE_MODEL_NAMES = {
    "vae",
    "betavae",
    "dvae",
    "hvae",
    "vamppriorvae",
    "infovae",
    "mmdvae",
    "dipvae",
    "betatcvae",
}

_VQ_MODEL_NAMES = {
    "vqvae",
    "gumbelvq",
    "fsq",
    "rfsq",
    "pqvae",
    "rqvae",
    "vqvae2",
    "opqvae",
}


class EmbeddingTransform(ABC):
    """Base interface for any embedding-space transform."""

    kind: str
    output_representation: str

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the transform on a matrix of embeddings."""

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform a matrix of embeddings."""


@dataclass
class IdentityTransform(EmbeddingTransform):
    """No-op transform used as the benchmark baseline."""

    kind: str = "identity"
    output_representation: str = "input"

    def fit(self, embeddings: np.ndarray) -> None:
        del embeddings

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings, dtype=np.float32)


class QuantizedSequenceProjector(nn.Module):
    """Fuse per-codebook vectors into one dense vector."""

    def __init__(
        self,
        *,
        source_kind: str,
        num_slots: int,
        output_dim: int,
        seed: int,
        source_slot_dim: int | None = None,
        codebook_size: int | None = None,
    ) -> None:
        super().__init__()
        self.source_kind = source_kind
        self.num_slots = num_slots
        self.output_dim = output_dim
        self.seed = seed
        self.source_slot_dim = source_slot_dim
        self.codebook_size = codebook_size

        generator = torch.Generator(device="cpu").manual_seed(seed)
        if source_kind == "quantized_projected":
            if source_slot_dim is None:
                raise ValueError("source_slot_dim is required for quantized_projected.")
            if source_slot_dim != output_dim:
                slot_weight = _xavier_uniform(
                    (num_slots, source_slot_dim, output_dim),
                    fan_in=source_slot_dim,
                    fan_out=output_dim,
                    generator=generator,
                )
                slot_bias = torch.zeros(num_slots, output_dim)
            else:
                slot_weight = torch.empty(0)
                slot_bias = torch.empty(0)
            self.register_buffer("slot_projection_weight", slot_weight)
            self.register_buffer("slot_projection_bias", slot_bias)
            self.register_buffer("index_embeddings", torch.empty(0))
        elif source_kind == "code_indices_projected":
            if codebook_size is None:
                raise ValueError("codebook_size is required for code_indices_projected.")
            self.register_buffer(
                "index_embeddings",
                _xavier_uniform(
                    (num_slots, codebook_size, output_dim),
                    fan_in=output_dim,
                    fan_out=output_dim,
                    generator=generator,
                ),
            )
            self.register_buffer("slot_projection_weight", torch.empty(0))
            self.register_buffer("slot_projection_bias", torch.empty(0))
        else:
            raise ValueError(f"Unsupported source_kind: {source_kind!r}")

        self.register_buffer(
            "concat_projection_weight",
            _xavier_uniform(
                (num_slots * output_dim, output_dim),
                fan_in=num_slots * output_dim,
                fan_out=output_dim,
                generator=generator,
            ),
        )
        self.register_buffer("concat_projection_bias", torch.zeros(output_dim))

    def project_from_vectors(self, sequence_vectors: torch.Tensor) -> torch.Tensor:
        """Project a [batch, slots, dim] tensor to [batch, output_dim]."""

        projected = sequence_vectors
        if self.slot_projection_weight.numel() > 0:
            projected = torch.einsum("bsi,sio->bso", sequence_vectors, self.slot_projection_weight)
            projected = projected + self.slot_projection_bias.unsqueeze(0)
        flat = projected.reshape(projected.shape[0], self.num_slots * self.output_dim)
        return flat @ self.concat_projection_weight + self.concat_projection_bias

    def project_from_indices(self, codebook_indices: torch.Tensor) -> torch.Tensor:
        """Lookup random slot-specific embeddings and fuse them."""

        slot_vectors = []
        for slot_index in range(self.num_slots):
            slot_vectors.append(self.index_embeddings[slot_index][codebook_indices[:, slot_index]])
        sequence_vectors = torch.stack(slot_vectors, dim=1)
        return self.project_from_vectors(sequence_vectors)


class AutoencoderEmbeddingTransform(EmbeddingTransform):
    """Embedding transform backed by an autoencoders model."""

    def __init__(self, config: TransformConfig) -> None:
        if not config.model_name:
            raise ValueError("transform.model_name is required when transform.kind='autoencoder'.")
        if not config.encoder_name:
            raise ValueError("transform.encoder_name is required when transform.kind='autoencoder'.")

        self.kind = "autoencoder"
        self.output_representation = config.output_representation
        self.config = config
        self.device = None
        self.model = None
        self.sample_spec = None
        self.checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
        self.projector = None
        self.projector_state_path = (
            self.checkpoint_dir / "benchmark_projection.pt"
            if self.checkpoint_dir is not None
            else None
        )
        self.trace_printed = False

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the autoencoder on a matrix of train embeddings."""

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, received shape {matrix.shape}.")
        if matrix.shape[0] < 2:
            raise ValueError("Need at least two samples to fit an autoencoder transform.")

        train_config = dict(self.config.training_config)
        seed = int(train_config.get("seed", 42))
        set_seed(seed)

        tensor = torch.from_numpy(matrix)
        self.sample_spec = TensorSpec(shape=(int(matrix.shape[1]),))
        self.model = self._build_model()
        self._print_pipeline_trace()

        dataset = EmbeddingTensorDataset(
            EmbeddingMatrix(
                tokens=[f"fit-{index}" for index in range(tensor.shape[0])],
                matrix=tensor,
                token_to_index={f"fit-{index}": index for index in range(tensor.shape[0])},
                name="fit-embeddings",
            )
        )
        splits = split_dataset(
            dataset,
            validation_ratio=float(train_config.pop("validation_ratio", 0.1)),
            test_ratio=float(train_config.pop("test_ratio", 0.1)),
            seed=seed,
        )
        dataloaders = create_dataloaders(
            splits,
            batch_size=int(train_config.get("batch_size", 256)),
            num_workers=int(train_config.pop("num_workers", 0)),
        )

        trainer_class, trainer_config_class = _select_trainer_components(self.config.model_name)
        trainer_args = trainer_config_class(**train_config)
        trainer = trainer_class(model=self.model, args=trainer_args)
        trainer.fit(dataloaders, metadata={"dataset": "benchmark-fit", "model": self.config.model_name})

        self.device = resolve_device(trainer_args.device)
        self.model.to(self.device)
        self.model.eval()

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(self.checkpoint_dir)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Map base embeddings into the configured output space."""

        if self.model is None:
            if self.checkpoint_dir is None:
                raise RuntimeError("Autoencoder transform is not fitted and no checkpoint_dir was provided.")
            if self.sample_spec is None:
                dim = int(np.asarray(embeddings).shape[-1])
                self.sample_spec = TensorSpec(shape=(dim,))
            model = self._build_model()
            self.model = type(model).from_pretrained(
                self.checkpoint_dir,
                sample_spec=self.sample_spec,
            )
            self.device = resolve_device(self.config.training_config.get("device", "auto"))
            self.model.to(self.device)
            self.model.eval()
            self._print_pipeline_trace(source="checkpoint")

        device = self.device or resolve_device(self.config.training_config.get("device", "auto"))
        self.device = device
        matrix = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), dtype=torch.float32, device=device)
        outputs: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, matrix.shape[0], self.config.batch_size):
                batch = matrix[start : start + self.config.batch_size]
                export = self.model.export(batch)
                representation = self._select_representation(export)
                outputs.append(representation.detach().to(device="cpu", dtype=torch.float32).numpy())

        return np.concatenate(outputs, axis=0)

    def _build_model(self) -> nn.Module:
        if self.sample_spec is None:
            raise RuntimeError("sample_spec must be set before building the autoencoder model.")
        # Forward model/backbone kwargs unchanged so benchmark configs can
        # expose the full autoencoders surface, including model-family specific
        # options such as residual quantizer counts or sinkhorn settings.
        return load_model(
            self.config.model_name,
            sample_spec=self.sample_spec,
            encoder=self.config.encoder_name,
            encoder_config=self.config.encoder_config,
            decoder=self.config.decoder_name,
            decoder_config=self.config.decoder_config,
            **self.config.model_config,
        )

    def _select_representation(self, export: Any) -> torch.Tensor:
        simple_mapping = {
            "latents": export.latents,
            "reconstruction": export.reconstruction,
            "encoded": export.encoded,
            "posterior_mean": export.posterior_mean,
            "quantized_latents": export.quantized_latents,
        }
        value = simple_mapping.get(self.output_representation)
        if value is not None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected representation {self.output_representation!r} to be a torch.Tensor.")
            return value

        if self.output_representation == "quantized_projected":
            return self._project_quantized_export(export, source_kind="quantized_projected")
        if self.output_representation == "code_indices_projected":
            return self._project_quantized_export(export, source_kind="code_indices_projected")

        available = ", ".join(sorted(key for key, item in simple_mapping.items() if item is not None))
        raise ValueError(
            f"Representation {self.output_representation!r} is unavailable for this model export. "
            f"Available simple outputs: {available or 'none'}."
        )

    def _project_quantized_export(self, export: Any, *, source_kind: str) -> torch.Tensor:
        codebook_indices = _normalize_codebook_indices(export.codebook_indices)
        if source_kind == "quantized_projected":
            sequence_vectors = self._extract_quantized_sequence_vectors(export, codebook_indices)
            projector = self._get_or_build_projector(
                source_kind=source_kind,
                num_slots=sequence_vectors.shape[1],
                source_slot_dim=sequence_vectors.shape[2],
                codebook_size=None,
            )
            return projector.project_from_vectors(sequence_vectors.to(device=self.device))

        codebook_size = _infer_codebook_size(export, codebook_indices)
        projector = self._get_or_build_projector(
            source_kind=source_kind,
            num_slots=codebook_indices.shape[1],
            source_slot_dim=None,
            codebook_size=codebook_size,
        )
        return projector.project_from_indices(codebook_indices.to(device=self.device))

    def _extract_quantized_sequence_vectors(
        self,
        export: Any,
        codebook_indices: torch.Tensor,
    ) -> torch.Tensor:
        codebooks = None
        if isinstance(getattr(export, "extras", None), dict):
            codebooks = export.extras.get("codebooks")
        if isinstance(codebooks, torch.Tensor):
            if codebooks.ndim != 3:
                raise ValueError(f"Expected quantized export codebooks to have rank 3, received {tuple(codebooks.shape)}.")
            slot_vectors = []
            for slot_index in range(codebook_indices.shape[1]):
                slot_vectors.append(codebooks[slot_index][codebook_indices[:, slot_index]])
            return torch.stack(slot_vectors, dim=1)

        quantized_latents = getattr(export, "quantized_latents", None)
        if isinstance(quantized_latents, torch.Tensor) and quantized_latents.ndim == 2:
            return quantized_latents.unsqueeze(1)
        raise ValueError(
            "Projected quantized representations require codebook-wise vectors. "
            "Expected export.extras['codebooks'] together with codebook_indices."
        )

    def _get_or_build_projector(
        self,
        *,
        source_kind: str,
        num_slots: int,
        source_slot_dim: int | None,
        codebook_size: int | None,
    ) -> QuantizedSequenceProjector:
        if self.projector is not None:
            return self.projector

        metadata = {
            "source_kind": source_kind,
            "num_slots": num_slots,
            "source_slot_dim": source_slot_dim,
            "codebook_size": codebook_size,
            "output_dim": self._target_projection_dim(),
            "seed": int(self.config.projection_seed),
        }

        if self.projector_state_path is not None and self.projector_state_path.exists():
            payload = torch.load(self.projector_state_path, map_location="cpu")
            saved_meta = payload["metadata"]
            for key, expected_value in metadata.items():
                saved_value = saved_meta.get(key)
                if saved_value != expected_value:
                    raise ValueError(
                        "Saved benchmark projector metadata does not match the current config: "
                        f"{key}={saved_value!r} (saved) vs {expected_value!r} (current)."
                    )
            projector = QuantizedSequenceProjector(
                source_kind=saved_meta["source_kind"],
                num_slots=int(saved_meta["num_slots"]),
                output_dim=int(saved_meta["output_dim"]),
                seed=int(saved_meta["seed"]),
                source_slot_dim=saved_meta["source_slot_dim"],
                codebook_size=saved_meta["codebook_size"],
            )
            projector.load_state_dict(payload["state_dict"])
            self.projector = projector.to(self.device)
            return self.projector

        projector = QuantizedSequenceProjector(
            source_kind=source_kind,
            num_slots=num_slots,
            output_dim=self._target_projection_dim(),
            seed=int(self.config.projection_seed),
            source_slot_dim=source_slot_dim,
            codebook_size=codebook_size,
        )
        self.projector = projector.to(self.device)
        if self.projector_state_path is not None:
            self.projector_state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": metadata,
                    "state_dict": projector.state_dict(),
                },
                self.projector_state_path,
            )
        return self.projector

    def _target_projection_dim(self) -> int:
        if self.config.projection_dim is not None:
            return int(self.config.projection_dim)
        latent_dim = self.config.model_config.get("latent_dim")
        if latent_dim is not None:
            return int(latent_dim)
        if self.sample_spec is not None and isinstance(self.sample_spec, TensorSpec):
            sample_dim = self.sample_spec.shape[-1]
            if sample_dim is not None:
                return int(sample_dim)
        raise ValueError("Unable to infer projection_dim. Please set transform.projection_dim explicitly.")

    def _print_pipeline_trace(self, *, source: str = "config") -> None:
        if self.trace_printed:
            return
        if self.model is None or not hasattr(self.model, "get_pipeline_trace"):
            return

        pipeline = self.model.get_pipeline_trace()
        self.trace_printed = True

        print()
        print(
            style(
                f" AE Shape Trace ({self.config.model_name}, {source}) ",
                fg="white",
                bg="magenta",
                bold=True,
            )
        )
        if not pipeline:
            print(style("  <empty>", fg="yellow", dim=True))
            print(style(" End Trace ", fg="black", bg="yellow", bold=True))
            print()
            return

        first_step = pipeline[0]
        print(
            f"{style(first_step.name, fg='cyan', bold=True)} "
            f"{style(':', fg='magenta', dim=True)} "
            f"{_format_spec(first_step.output_spec)}"
        )

        for step in pipeline[1:]:
            header = (
                f"{style(step.name, fg='cyan', bold=True)} "
                f"{style('->', fg='magenta', dim=True)} "
                f"{_format_spec(step.output_spec)}"
            )
            print(header)
            for child in step.children or []:
                child_line = (
                    f"  {style('↳', fg='yellow', bold=True)} "
                    f"{style(child.name, fg='blue')} "
                    f"{style('->', fg='magenta', dim=True)} "
                    f"{_format_spec(child.output_spec)}"
                )
                print(child_line)
        print(style(" End Trace ", fg="black", bg="yellow", bold=True))
        print()


def build_transform(config: TransformConfig) -> EmbeddingTransform:
    """Construct a transform from config."""

    if config.kind == "identity":
        return IdentityTransform()
    if config.kind == "autoencoder":
        return AutoencoderEmbeddingTransform(config)
    raise ValueError(f"Unsupported transform kind: {config.kind!r}")


def _select_trainer_components(model_name: str) -> tuple[type[Any], type[Any]]:
    if model_name == "factorvae":
        return FactorVAETrainer, FactorVariationalAutoencoderTrainingConfig
    if model_name == "aae":
        return AdversarialAutoencoderTrainer, AdversarialAutoencoderTrainingConfig
    if model_name in _VQ_MODEL_NAMES:
        return VQTrainer, TrainingConfig
    if model_name in _VAE_MODEL_NAMES:
        return VAETrainer, TrainingConfig
    return AETrainer, TrainingConfig


def _normalize_codebook_indices(codebook_indices: Any) -> torch.Tensor:
    if not isinstance(codebook_indices, torch.Tensor):
        raise ValueError("This output_representation requires export.codebook_indices.")
    if codebook_indices.ndim == 1:
        return codebook_indices.unsqueeze(1).to(dtype=torch.long)
    if codebook_indices.ndim == 2:
        return codebook_indices.to(dtype=torch.long)
    raise ValueError(
        f"Expected codebook_indices to have rank 1 or 2 for sequence projection, received {tuple(codebook_indices.shape)}."
    )


def _infer_codebook_size(export: Any, codebook_indices: torch.Tensor) -> int:
    if isinstance(getattr(export, "extras", None), dict):
        codebooks = export.extras.get("codebooks")
        if isinstance(codebooks, torch.Tensor):
            return int(codebooks.shape[1])
        codebook_size = export.extras.get("codebook_size")
        if codebook_size is not None:
            return int(codebook_size)
    return int(codebook_indices.max().item()) + 1


def _xavier_uniform(
    shape: tuple[int, ...],
    *,
    fan_in: int,
    fan_out: int,
    generator: torch.Generator,
) -> torch.Tensor:
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    return torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)


def _format_spec(spec: Any) -> str:
    return style(str(spec), fg="green")
