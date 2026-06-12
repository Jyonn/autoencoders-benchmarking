"""Run a text-classification benchmark experiment."""

from __future__ import annotations

import argparse
import json

from autoencoders_benchmarking import (
    ClassificationBenchmarkRunner,
    load_experiment_config,
    parse_cli_placeholders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the dataset/task YAML file.")
    parser.add_argument("--transform", required=True, help="Path to the transform YAML file.")
    args, unknown = parser.parse_known_args()

    config = load_experiment_config(
        args.data,
        args.transform,
        placeholders=parse_cli_placeholders(unknown),
    )
    runner = ClassificationBenchmarkRunner(config)
    summary = runner.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
