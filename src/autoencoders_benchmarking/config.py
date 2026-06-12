"""Configuration objects and loaders for benchmark experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import refconfig
from refconfig import RefConfig


@dataclass
class EncoderConfig:
    """Configuration for the upstream sentence embedding encoder."""

    model_name: str
    device: str | None = None
    batch_size: int = 128
    normalize_embeddings: bool = False


@dataclass
class TaskConfig:
    """Configuration for the downstream MTEB task."""

    name: str
    output_dir: str = "results"
    eval_splits: list[str] = field(default_factory=lambda: ["test"])
    eval_subsets: list[str] | None = None
    max_fit_samples: int | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentConfig:
    """Named config section following the autoencoders examples layout."""

    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Benchmark dataset/task bundle."""

    name: str
    embedding_dim: int | None
    encoder: EncoderConfig
    task: TaskConfig


@dataclass
class TransformConfig:
    """Configuration for the embedding transform."""

    name: str
    kind: str = "identity"
    output_representation: str = "latents"
    batch_size: int = 512
    projection_dim: int | None = None
    projection_seed: int = 42
    checkpoint_dir: str | None = None
    fit: bool = False
    model: ComponentConfig | None = None
    encoder: ComponentConfig | None = None
    decoder: ComponentConfig | None = None
    trainer: dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str | None:
        return self.model.name if self.model is not None else None

    @property
    def model_config(self) -> dict[str, Any]:
        return dict(self.model.config) if self.model is not None else {}

    @property
    def encoder_name(self) -> str | None:
        return self.encoder.name if self.encoder is not None else None

    @property
    def encoder_config(self) -> dict[str, Any]:
        return dict(self.encoder.config) if self.encoder is not None else {}

    @property
    def decoder_name(self) -> str | None:
        return self.decoder.name if self.decoder is not None else None

    @property
    def decoder_config(self) -> dict[str, Any] | None:
        return dict(self.decoder.config) if self.decoder is not None else None

    @property
    def training_config(self) -> dict[str, Any]:
        return dict(self.trainer)


@dataclass
class ExperimentConfig:
    """Top-level resolved experiment configuration."""

    experiment_name: str
    data: DataConfig
    transform: TransformConfig
    sources: dict[str, str] = field(default_factory=dict)
    placeholders: dict[str, Any] = field(default_factory=dict)

    @property
    def encoder(self) -> EncoderConfig:
        return self.data.encoder

    @property
    def task(self) -> TaskConfig:
        return self.data.task


def parse_cli_placeholders(tokens: Sequence[str]) -> dict[str, Any]:
    """Parse `--key value` or `--key=value` CLI placeholder overrides."""

    overrides: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument {token!r}.")
        if "=" in token:
            raw_key, raw_value = token[2:].split("=", 1)
            consumed = 1
        else:
            if index + 1 >= len(tokens):
                raise ValueError(f"Expected a value after {token!r}.")
            raw_key = token[2:]
            raw_value = tokens[index + 1]
            consumed = 2
        overrides[raw_key.replace("-", "_")] = _parse_cli_scalar(raw_value)
        index += consumed
    return overrides


def load_experiment_config(
    data_path: str | Path,
    transform_path: str | Path,
    *,
    placeholders: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load a benchmark experiment from split data/transform YAML files."""

    data_config_path = Path(data_path)
    transform_config_path = Path(transform_path)
    resolved = RefConfig().add(
        refconfig.CType.SMART,
        data=str(data_config_path),
        transform=str(transform_config_path),
        **(placeholders or {}),
    ).parse()
    if not isinstance(resolved, dict):
        raise TypeError("Expected RefConfig to resolve to a mapping.")

    data_payload = _normalize_sequences(_require_mapping(resolved.get("data"), section="data"))
    transform_payload = _normalize_sequences(
        _require_mapping(resolved.get("transform"), section="transform")
    )

    experiment_name = _resolve_experiment_name(resolved, data_payload, transform_payload)
    _populate_default_artifact_dirs(
        experiment_name=experiment_name,
        data_payload=data_payload,
        transform_payload=transform_payload,
    )

    return ExperimentConfig(
        experiment_name=experiment_name,
        data=_build_data_config(data_payload),
        transform=_build_transform_config(transform_payload),
        sources={
            "data": str(data_config_path),
            "transform": str(transform_config_path),
        },
        placeholders=dict(placeholders or {}),
    )


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert an experiment config dataclass tree into a plain mapping."""

    return asdict(config)


def _build_data_config(payload: dict[str, Any]) -> DataConfig:
    return DataConfig(
        name=str(payload["name"]),
        embedding_dim=_optional_int(payload.get("embedding_dim")),
        encoder=EncoderConfig(**_require_mapping(payload.get("encoder"), section="data.encoder")),
        task=TaskConfig(**_require_mapping(payload.get("task"), section="data.task")),
    )


def _build_transform_config(payload: dict[str, Any]) -> TransformConfig:
    return TransformConfig(
        name=str(payload["name"]),
        kind=str(payload.get("kind", "identity")),
        output_representation=str(payload.get("output_representation", "latents")),
        batch_size=int(payload.get("batch_size", 512)),
        projection_dim=_optional_int(payload.get("projection_dim")),
        projection_seed=int(payload.get("projection_seed", 42)),
        checkpoint_dir=_optional_str(payload.get("checkpoint_dir")),
        fit=bool(payload.get("fit", False)),
        model=_build_component(payload.get("model"), section="transform.model"),
        encoder=_build_component(payload.get("encoder"), section="transform.encoder"),
        decoder=_build_component(payload.get("decoder"), section="transform.decoder"),
        trainer=_require_mapping(payload.get("trainer") or {}, section="transform.trainer"),
    )


def _build_component(value: Any, *, section: str) -> ComponentConfig | None:
    if value is None:
        return None
    if isinstance(value, str):
        return ComponentConfig(name=value)
    mapping = _require_mapping(value, section=section)
    return ComponentConfig(
        name=_optional_str(mapping.get("name")),
        config=_require_mapping(mapping.get("config") or {}, section=f"{section}.config"),
    )


def _require_mapping(value: Any, *, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{section}' to be a mapping, received {type(value).__name__}.")
    return dict(value)


def _resolve_experiment_name(
    resolved: dict[str, Any],
    data_payload: dict[str, Any],
    transform_payload: dict[str, Any],
) -> str:
    explicit = resolved.get("experiment_name")
    if explicit is not None:
        return str(explicit)
    explicit = transform_payload.get("experiment_name")
    if explicit is not None:
        return str(explicit)
    explicit = data_payload.get("experiment_name")
    if explicit is not None:
        return str(explicit)
    return f"{data_payload['name']}-{transform_payload['name']}"


def _populate_default_artifact_dirs(
    *,
    experiment_name: str,
    data_payload: dict[str, Any],
    transform_payload: dict[str, Any],
) -> None:
    task_payload = _require_mapping(data_payload.get("task"), section="data.task")
    output_root = str(task_payload.get("output_dir", "results"))
    experiment_root = Path(output_root) / experiment_name

    transform_payload.setdefault("checkpoint_dir", str(experiment_root / "checkpoint"))
    trainer_payload = _require_mapping(transform_payload.get("trainer") or {}, section="transform.trainer")
    trainer_payload.setdefault("output_dir", str(experiment_root / "train"))
    transform_payload["trainer"] = trainer_payload


def _normalize_sequences(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _normalize_sequences(child_value, key=child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_normalize_sequences(item) for item in value]
    if _is_sequence_key(key):
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")] if "," in value else [value.strip()]
            return [_parse_cli_scalar(part) for part in parts if part]
        if value is None:
            return None
        return [value]
    return value


def _is_sequence_key(key: str | None) -> bool:
    if key is None:
        return False
    if key.endswith("_dims"):
        return True
    return key in {"eval_splits", "eval_subsets", "sinkhorn_epsilon", "levels"}


def _parse_cli_scalar(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if raw_value.isdigit() or (raw_value.startswith("-") and raw_value[1:].isdigit()):
        return int(raw_value)
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
