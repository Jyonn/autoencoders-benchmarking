"""Flatten benchmark results into CSV and JSONL tables."""

from __future__ import annotations

import argparse
import json

from autoencoders_benchmarking.summarizer import summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Root directory containing experiment result folders.")
    parser.add_argument("--csv", default="results/_tables/results_summary.csv", help="Output CSV path.")
    parser.add_argument("--jsonl", default="results/_tables/results_summary.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    rows = summarize_results(
        args.results_root,
        csv_path=args.csv,
        jsonl_path=args.jsonl,
    )
    print(json.dumps({"rows": len(rows), "csv": args.csv, "jsonl": args.jsonl}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
