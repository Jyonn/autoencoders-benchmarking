"""Run a text-classification benchmark experiment."""

from __future__ import annotations

import argparse
import json

from autoencoders_benchmarking import ClassificationBenchmarkRunner, load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    runner = ClassificationBenchmarkRunner(config)
    summary = runner.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
