#!/usr/bin/env python3
"""Recompute dynamic multi-positive metrics from saved retrieval rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_visual_retrieval import multi_positive_metrics


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def combined_metrics(result_dir: Path) -> dict[str, float | int]:
    text_rows = read_jsonl(result_dir / "rankings_clean.jsonl")
    image_rows = read_jsonl(result_dir / "rankings_image_to_text.jsonl")
    text = multi_positive_metrics(
        [row["ranked_image_ids"] for row in text_rows],
        [set(row["gold_image_ids"]) for row in text_rows],
    )
    image = multi_positive_metrics(
        [row["ranked_pair_ids"] for row in image_rows],
        [set(row["gold_pair_ids"]) for row in image_rows],
    )
    if text["queries"] != image["queries"]:
        raise ValueError(f"Bidirectional query counts differ in {result_dir}")
    return {
        "map@R": (float(text["map@R"]) + float(image["map@R"])) / 2.0,
        "R-Precision": (
            float(text["R-Precision"]) + float(image["R-Precision"])
        ) / 2.0,
        "R@1": (float(text["R@1"]) + float(image["R@1"])) / 2.0,
        "queries_per_direction": int(text["queries"]),
    }


def metric_metadata() -> dict[str, str]:
    return {
        "reported_metrics": (
            "bidirectional mean mAP@R, R-Precision, and R@1 on "
            "multi-positive queries only"
        ),
        "R_definition": "number of relevant items for each query",
        "map_at_R_denominator": "R",
    }


def update_result(result_dir: Path) -> dict[str, float | int]:
    metrics = combined_metrics(result_dir)

    summary_path = result_dir / "recall_at_k.json"
    summary = read_json(summary_path)
    summary.pop("map@10", None)
    summary["multi_positive"] = metrics
    write_json(summary_path, summary)

    config_path = result_dir / "config.json"
    config = read_json(config_path)
    config.setdefault("multi_positive_evaluation", {}).update(metric_metadata())
    config["multi_positive_evaluation"].pop("reported_metric", None)
    config["multi_positive_evaluation"].pop("map_at_k_denominator", None)
    write_json(config_path, config)

    clean_path = result_dir / "metrics_clean.json"
    clean = read_json(clean_path)
    clean.setdefault("multi_positive_evaluation", {}).update(metric_metadata())
    clean["multi_positive_evaluation"].pop("reported_metric", None)
    clean["multi_positive_evaluation"].pop("map_at_k_denominator", None)
    clean["metrics"].pop("map@10", None)
    clean["metrics"]["multi_positive"] = metrics
    write_json(clean_path, clean)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for result_dir in args.result_dirs:
        metrics = update_result(result_dir.resolve())
        print(json.dumps({"result_dir": str(result_dir), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
