"""Utilities for flattening benchmark outputs into table-friendly rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Any


def summarize_results(
    results_root: str | Path = "results",
    *,
    csv_path: str | Path | None = None,
    jsonl_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Summarize all experiment directories under a results root."""

    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        rows.extend(summarize_experiment_dir(experiment_dir, results_root=root))

    if csv_path is not None:
        write_rows_csv(rows, csv_path)
    if jsonl_path is not None:
        write_rows_jsonl(rows, jsonl_path)
    return rows


def summarize_experiment_dir(
    experiment_dir: str | Path,
    *,
    results_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Summarize one experiment output directory into flattened rows."""

    experiment_path = Path(experiment_dir)
    summary_path = experiment_path / "summary.json"
    if not summary_path.exists():
        return []

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_payload, list):
        raise TypeError(f"Expected summary list in {summary_path}.")

    config_payload = _read_optional_json(experiment_path / "experiment_config.json")
    train_metrics = _read_optional_json(experiment_path / "train" / "metrics.json")
    model_meta = _find_model_meta(experiment_path / "mteb")

    rows: list[dict[str, Any]] = []
    for task_result in summary_payload:
        rows.extend(
            _flatten_task_result(
                task_result,
                experiment_path=experiment_path,
                results_root=Path(results_root) if results_root is not None else None,
                config_payload=config_payload,
                train_metrics=train_metrics,
                model_meta=model_meta,
            )
        )
    return rows


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write flat rows to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_scalar(row.get(key)) for key in fieldnames})


def write_rows_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write flat rows to JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _flatten_task_result(
    task_result: dict[str, Any],
    *,
    experiment_path: Path,
    results_root: Path | None,
    config_payload: dict[str, Any] | None,
    train_metrics: dict[str, Any] | None,
    model_meta: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    base_row = _build_base_row(
        task_result,
        experiment_path=experiment_path,
        results_root=results_root,
        config_payload=config_payload,
        train_metrics=train_metrics,
        model_meta=model_meta,
    )

    rows: list[dict[str, Any]] = []
    for split_name, split_scores in task_result.get("scores", {}).items():
        if not isinstance(split_scores, list):
            continue
        for item in split_scores:
            row = dict(base_row)
            row["split"] = split_name
            row["hf_subset"] = item.get("hf_subset")
            row["languages"] = ",".join(item.get("languages", [])) if isinstance(item.get("languages"), list) else None

            experiment_scores = item.get("scores_per_experiment", [])
            for metric_name, metric_value in item.items():
                if metric_name in {"scores_per_experiment", "hf_subset", "languages", "mteb_version"}:
                    continue
                row[f"metric_{metric_name}"] = metric_value

            if isinstance(experiment_scores, list) and experiment_scores:
                row["num_experiments"] = len(experiment_scores)
                for metric_name in _numeric_metric_names(experiment_scores):
                    values = [score.get(metric_name) for score in experiment_scores]
                    numeric_values = [float(value) for value in values if isinstance(value, (int, float)) and not _is_nan(value)]
                    if not numeric_values:
                        continue
                    row[f"metric_{metric_name}_std"] = 0.0 if len(numeric_values) == 1 else pstdev(numeric_values)

            rows.append(row)
    return rows


def _build_base_row(
    task_result: dict[str, Any],
    *,
    experiment_path: Path,
    results_root: Path | None,
    config_payload: dict[str, Any] | None,
    train_metrics: dict[str, Any] | None,
    model_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    experiment_relpath = (
        str(experiment_path.relative_to(results_root))
        if results_root is not None and experiment_path.is_relative_to(results_root)
        else str(experiment_path)
    )
    row: dict[str, Any] = {
        "experiment_name": experiment_path.name,
        "experiment_dir": experiment_relpath,
        "task_name": task_result.get("task_name"),
        "dataset_revision": task_result.get("dataset_revision"),
        "mteb_version": task_result.get("mteb_version"),
        "evaluation_time_sec": task_result.get("evaluation_time"),
    }

    if model_meta is not None:
        experiment_kwargs = model_meta.get("experiment_kwargs") or {}
        row.update(
            {
                "base_model_name": model_meta.get("name"),
                "base_model_revision": model_meta.get("revision"),
                "base_embed_dim": model_meta.get("embed_dim"),
                "base_similarity": model_meta.get("similarity_fn_name"),
                "transform_kind": experiment_kwargs.get("transform_kind"),
                "transform_output": experiment_kwargs.get("transform_output"),
            }
        )

    if config_payload is not None:
        encoder_cfg = config_payload.get("encoder", {})
        task_cfg = config_payload.get("task", {})
        transform_cfg = config_payload.get("transform", {})
        row.update(
            {
                "config_encoder_model_name": encoder_cfg.get("model_name"),
                "config_task_name": task_cfg.get("name"),
                "config_transform_kind": transform_cfg.get("kind"),
                "config_transform_model_name": transform_cfg.get("model_name"),
                "config_transform_output": transform_cfg.get("output_representation"),
                "config_checkpoint_dir": transform_cfg.get("checkpoint_dir"),
            }
        )

    if train_metrics is not None:
        training_args = train_metrics.get("training_args", {})
        row.update(
            {
                "fit_model_name": train_metrics.get("model"),
                "fit_device": train_metrics.get("device"),
                "fit_epochs_completed": train_metrics.get("epochs_completed"),
                "fit_best_epoch": train_metrics.get("best_epoch"),
                "fit_best_validation_loss": train_metrics.get("best_validation_loss"),
                "fit_final_test_loss": train_metrics.get("final_test_loss"),
                "fit_stopped_early": train_metrics.get("stopped_early"),
                "fit_patience": training_args.get("patience"),
                "fit_configured_epochs": training_args.get("epochs"),
                "fit_learning_rate": training_args.get("learning_rate"),
                "fit_optimizer_name": training_args.get("optimizer_name"),
            }
        )

    return row


def _find_model_meta(mteb_dir: Path) -> dict[str, Any] | None:
    model_meta_paths = sorted(mteb_dir.glob("*/*/model_meta.json"))
    if not model_meta_paths:
        return None
    return _read_optional_json(model_meta_paths[0])


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _numeric_metric_names(experiment_scores: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for score in experiment_scores:
        for key, value in score.items():
            if isinstance(value, (int, float)) and not _is_nan(value):
                names.add(key)
    return sorted(names)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _format_scalar(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value
