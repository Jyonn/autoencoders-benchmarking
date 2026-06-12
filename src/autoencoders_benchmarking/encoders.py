"""Encoder wrappers for benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from .config import EncoderConfig

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from .transforms import EmbeddingTransform


@dataclass
class SentenceTransformerBackend:
    """Shared upstream encoder used for both fitting and evaluation."""

    config: EncoderConfig

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode raw texts into dense numpy embeddings."""

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_mteb_encoder(self) -> SentenceTransformerEncoderWrapper:
        """Create the MTEB-compatible encoder wrapper."""

        return SentenceTransformerEncoderWrapper(model=self.model)


class TransformedMTEBEncoder(AbsEncoder):
    """MTEB encoder that applies a learned transform after base embeddings."""

    def __init__(self, base_encoder: EncoderProtocol, transform: EmbeddingTransform) -> None:
        self.base_encoder = base_encoder
        self.transform = transform
        self.model = getattr(base_encoder, "model", None)

        base_meta = base_encoder.mteb_model_meta
        experiment_kwargs = dict(base_meta.experiment_kwargs or {})
        experiment_kwargs.update(
            {
                "transform_kind": transform.kind,
                "transform_output": transform.output_representation,
            }
        )
        self.mteb_model_meta = base_meta.model_copy(
            update={"experiment_kwargs": experiment_kwargs},
            deep=True,
        )

    def encode(
        self,
        inputs: "DataLoader[BatchedInput]",
        *,
        task_metadata: "TaskMetadata",
        hf_split: str,
        hf_subset: str,
        prompt_type: "PromptType | None" = None,
        **kwargs: "Unpack[EncodeKwargs]",
    ) -> "Array":
        embeddings = self.base_encoder.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )
        return self.transform.transform(np.asarray(embeddings, dtype=np.float32))
