"""Classification benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mteb import MTEB, get_tasks

from .config import ExperimentConfig, experiment_config_to_dict
from .encoders import SentenceTransformerBackend, TransformedMTEBEncoder
from .transforms import build_transform


class ClassificationBenchmarkRunner:
    """Run MTEB classification experiments with embedding transforms."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.task.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[dict[str, Any]]:
        """Fit the transform if needed, evaluate, and persist a summary."""

        (self.output_dir / "experiment_config.json").write_text(
            json.dumps(experiment_config_to_dict(self.config), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        task = get_tasks(tasks=[self.config.task.name])[0]
        self._configure_task(task)
        backend = SentenceTransformerBackend(self.config.encoder)
        transform = build_transform(self.config.transform)

        if self.config.transform.fit:
            task.load_data()
            dataset = task.dataset
            if dataset is None:
                raise RuntimeError(f"Task {task.metadata.name} did not load a dataset.")
            train_split_name = str(getattr(task, "train_split", "train"))
            input_column_name = getattr(task, "input_column_name", "text")
            if not isinstance(input_column_name, str):
                raise ValueError(
                    f"Current classification runner expects a single text input column for fit time, "
                    f"received {input_column_name!r}."
                )
            train_split = dataset[train_split_name]
            if input_column_name not in train_split.column_names:
                raise ValueError(
                    f"Current classification runner expects a text column in task {task.metadata.name}. "
                    f"Available columns: {train_split.column_names}"
                )
            texts = list(train_split[input_column_name])
            if self.config.task.max_fit_samples is not None:
                texts = texts[: self.config.task.max_fit_samples]
            train_embeddings = backend.encode_texts(texts)
            transform.fit(train_embeddings)

        model = TransformedMTEBEncoder(
            base_encoder=backend.build_mteb_encoder(),
            transform=transform,
        )
        evaluation = MTEB(tasks=[task]).run(
            model,
            output_folder=str(self.output_dir / "mteb"),
            eval_splits=self.config.task.eval_splits,
            eval_subsets=self.config.task.eval_subsets,
            overwrite_results=True,
        )
        summary = [result.model_dump(mode="json") for result in evaluation]
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary

    def _configure_task(self, task: Any) -> None:
        for key, value in self.config.task.config.items():
            if not hasattr(task, key):
                raise ValueError(
                    f"Task {task.metadata.name} has no configurable attribute {key!r}. "
                    "Please check the MTEB task implementation for supported options."
                )
            setattr(task, key, value)
