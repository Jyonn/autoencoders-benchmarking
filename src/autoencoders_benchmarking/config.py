"""Configuration objects for benchmark experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EncoderConfig:
    """Configuration for the upstream embedding encoder."""

    model_name: str
    device: str | None = None
    batch_size: int = 128
    normalize_embeddings: bool = False


@dataclass
class TaskConfig:
    """Configuration for the downstream evaluation task."""

    name: str
    output_dir: str
    eval_splits: list[str] = field(default_factory=lambda: ["test"])
    eval_subsets: list[str] | None = None
    max_fit_samples: int | None = None


@dataclass
class TransformConfig:
    """Configuration for the embedding transform.

    For ``kind="autoencoder"``, the four config mappings below are forwarded
    directly into the installed ``autoencoders`` library:

    - ``model_config`` -> model family config such as ``latent_dim``,
      ``num_quantizers``, ``codebook_size``, ``assignment_strategy``,
      ``sinkhorn_epsilon``
    - ``encoder_config`` -> encoder backbone config such as ``hidden_dims``
    - ``decoder_config`` -> decoder backbone config
    - ``training_config`` -> trainer config such as ``epochs``, ``patience``,
      ``optimizer_name``
    """

    kind: str = "identity"
    output_representation: str = "latents"
    batch_size: int = 512
    checkpoint_dir: str | None = None
    fit: bool = False
    model_name: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    encoder_name: str | None = None
    encoder_config: dict[str, Any] = field(default_factory=dict)
    decoder_name: str | None = None
    decoder_config: dict[str, Any] | None = None
    training_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str
    encoder: EncoderConfig
    task: TaskConfig
    transform: TransformConfig


def _require_mapping(value: Any, *, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{section}' to be a mapping, received {type(value).__name__}.")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected root config mapping in {config_path}.")

    encoder_payload = _require_mapping(payload.get("encoder"), section="encoder")
    task_payload = _require_mapping(payload.get("task"), section="task")
    transform_payload = _require_mapping(payload.get("transform"), section="transform")

    return ExperimentConfig(
        experiment_name=str(payload["experiment_name"]),
        encoder=EncoderConfig(**encoder_payload),
        task=TaskConfig(**task_payload),
        transform=TransformConfig(**transform_payload),
    )


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert an experiment config dataclass tree into a plain mapping."""

    return asdict(config)
