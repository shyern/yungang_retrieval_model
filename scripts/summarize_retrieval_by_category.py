#!/usr/bin/env python3
"""Summarize exact and same-query multi-positive retrieval by category."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(ranks: list[int]) -> dict:
    return {
        "num_queries": len(ranks),
        "recall@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall@5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": statistics.median(ranks),
    }


def grouped_metrics(rows: list[dict], rank_field: str) -> dict[str, dict]:
    ranks: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        ranks[row["category"]].append(row[rank_field])
    return {category: metrics(values) for category, values in sorted(ranks.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-source", default="visual_text")
    args = parser.parse_args()

    pairs_path = Path(args.pairs).resolve()
    rankings_path = Path(args.rankings).resolve()
    output_dir = Path(args.output_dir).resolve()
    pairs = read_jsonl(pairs_path)
    rankings = read_jsonl(rankings_path)
    pair_by_id = {item["pair_id"]: item for item in pairs}
    positives_by_query: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        if pair["query_source"] == args.query_source:
            positives_by_query[pair["query"]].append(pair["image_id"])

    enriched = []
    for ranking in rankings:
        pair = pair_by_id[ranking["pair_id"]]
        positive_ids = positives_by_query[ranking["query"]]
        best_rank = min(ranking["ranked_image_ids"].index(image_id) + 1 for image_id in positive_ids)
        enriched.append(
            {
                "pair_id": ranking["pair_id"],
                "category": pair["category"],
                "query": ranking["query"],
                "gold_image_id": ranking["gold_image_id"],
                "gold_rank": ranking["gold_rank"],
                "positive_image_ids": positive_ids,
                "best_positive_rank": best_rank,
                "num_positive_images": len(positive_ids),
            }
        )

    exact_ranks = [row["gold_rank"] for row in enriched]
    multi_ranks = [row["best_positive_rank"] for row in enriched]
    summary = {
        "pairs": str(pairs_path),
        "pairs_sha256": sha256(pairs_path),
        "rankings": str(rankings_path),
        "rankings_sha256": sha256(rankings_path),
        "query_source": args.query_source,
        "evaluation": {
            "exact_image": {
                "overall": metrics(exact_ranks),
                "by_category": grouped_metrics(enriched, "gold_rank"),
            },
            "same_query_multi_positive": {
                "overall": metrics(multi_ranks),
                "by_category": grouped_metrics(enriched, "best_positive_rank"),
                "multi_image_query_groups": sum(len(ids) > 1 for ids in positives_by_query.values()),
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "rankings_multi_positive.jsonl", enriched)
    (output_dir / "metrics_by_category.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["evaluation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
