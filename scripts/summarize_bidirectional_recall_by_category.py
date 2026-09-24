#!/usr/bin/env python3
"""Summarize bidirectional Recall@K by category from retrieval rankings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DIRECTIONS = ("text_to_image", "image_to_text")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recall_at_k(ranks: list[int]) -> dict[str, float | int]:
    return {
        "num_queries": len(ranks),
        **{
            f"recall@{k}": sum(rank <= k for rank in ranks) / len(ranks)
            for k in (1, 5, 10)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--text-to-image", required=True, type=Path)
    parser.add_argument("--image-to-text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pairs_path = args.pairs.resolve()
    text_to_image_path = args.text_to_image.resolve()
    image_to_text_path = args.image_to_text.resolve()
    pairs = read_jsonl(pairs_path)
    text_to_image = read_jsonl(text_to_image_path)
    image_to_text = read_jsonl(image_to_text_path)
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise RuntimeError("Duplicate pair_id in pairs file")
    if len(text_to_image) != len(pairs) or len(image_to_text) != len(pairs):
        raise RuntimeError("Rankings and pairs must contain the same number of rows")

    ranks: dict[str, defaultdict[str, list[int]]] = {
        direction: defaultdict(list) for direction in DIRECTIONS
    }
    seen_text_pairs = set()
    for row in text_to_image:
        pair_id = row["pair_id"]
        pair = pair_by_id[pair_id]
        ranks["text_to_image"][pair["category"]].append(int(row["gold_rank"]))
        seen_text_pairs.add(pair_id)

    seen_image_pairs = set()
    for row in image_to_text:
        pair_id = row["gold_pair_id"]
        pair = pair_by_id[pair_id]
        ranks["image_to_text"][pair["category"]].append(int(row["gold_rank"]))
        seen_image_pairs.add(pair_id)

    expected_ids = set(pair_by_id)
    if seen_text_pairs != expected_ids or seen_image_pairs != expected_ids:
        raise RuntimeError("Rankings do not cover every pair exactly once")

    categories = sorted({pair["category"] for pair in pairs})
    result = {
        "evaluation": "strict one-to-one bidirectional retrieval",
        "pairs": str(pairs_path),
        "gallery_size": len(pairs),
        "overall": {
            direction: recall_at_k(
                [rank for category in categories for rank in ranks[direction][category]]
            )
            for direction in DIRECTIONS
        },
        "by_category": {
            category: {
                direction: recall_at_k(ranks[direction][category])
                for direction in DIRECTIONS
            }
            for category in categories
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
