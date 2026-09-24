#!/usr/bin/env python3
"""Create one leakage-safe train/validation/test split for all experiments.

The split is made over relation components rather than individual rows:
multi-positive records stay intact, hard-negative pairs stay intact, and any
component connected through a shared image or description stays intact too.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")


def as_paths(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)} if value is not None else set()


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def allocate(units: list[set[str]], ratios: tuple[float, float, float], rng: random.Random) -> dict[str, str]:
    """Allocate relation components while approximately matching row ratios."""
    if not units:
        return {}
    shuffled = list(units)
    rng.shuffle(shuffled)
    # Place larger components first so multi-image groups cannot skew the split.
    shuffled.sort(key=lambda unit: len(unit), reverse=True)
    total = sum(len(unit) for unit in shuffled)
    targets = {name: total * ratio for name, ratio in zip(SPLITS, ratios)}
    current = {name: 0 for name in SPLITS}
    assignments: dict[str, str] = {}
    for unit in shuffled:
        # Choose the split with the largest normalized deficit.
        split = max(
            SPLITS,
            key=lambda name: (targets[name] - current[name]) / max(targets[name], 1.0),
        )
        for entry_id in unit:
            assignments[entry_id] = split
        current[split] += len(unit)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset/dataset.json")
    parser.add_argument("--multi", default="data/dataset/muti_positives.json")
    parser.add_argument("--hard", default="data/dataset/hard_negatives.json")
    parser.add_argument("--output-dir", default="data/dataset/splits")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must be positive and sum to 1")

    dataset = load(Path(args.dataset))
    multi_rows = load(Path(args.multi))
    hard_rows = load(Path(args.hard))
    by_id = {str(row["entry_id"]): row for row in dataset}
    if len(by_id) != len(dataset):
        raise ValueError("dataset.json contains duplicate entry_id values")
    all_ids = list(by_id)

    by_image: dict[str, set[str]] = defaultdict(set)
    for row in dataset:
        for image in as_paths(row.get("image_path")):
            by_image[image].add(str(row["entry_id"]))

    # Resolve each multi-positive row by its complete image set and description.
    multi_ids: set[str] = set()
    multi_row_to_id: dict[int, str] = {}
    for index, row in enumerate(multi_rows):
        images = as_paths(row.get("image_path"))
        candidates = {
            entry_id
            for image in images
            for entry_id in by_image.get(image, set())
            if set(as_paths(by_id[entry_id].get("image_path"))) == images
            and by_id[entry_id].get("description") == row.get("description")
        }
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot uniquely resolve multi-positive row {index}: {row.get('entry_id')}; "
                f"candidates={sorted(candidates)}"
            )
        entry_id = next(iter(candidates))
        multi_ids.add(entry_id)
        multi_row_to_id[index] = entry_id

    # Resolve each hard-negative group through both positive and negative image paths.
    hard_groups: dict[str, set[str]] = defaultdict(set)
    for row in hard_rows:
        group_id = str(row["group_id"])
        for field in ("positive_image_path", "negative_image_path"):
            for image in as_paths(row.get(field)):
                owners = by_image.get(image, set())
                if not owners:
                    raise ValueError(f"Hard-negative image is absent from dataset.json: {image}")
                hard_groups[group_id].update(owners)
    hard_ids = set().union(*hard_groups.values()) if hard_groups else set()

    # Union all rows connected by a relation, including the observed overlap
    # between hn_005 and two multi-positive records.
    uf = UnionFind(all_ids)
    for group_ids in hard_groups.values():
        group_ids = sorted(group_ids)
        for entry_id in group_ids[1:]:
            uf.union(group_ids[0], entry_id)
    # Keep any duplicate-description or shared-image records together as well.
    for owners in list(by_image.values()):
        owners = sorted(owners)
        for entry_id in owners[1:]:
            uf.union(owners[0], entry_id)

    components: dict[str, set[str]] = defaultdict(set)
    for entry_id in all_ids:
        components[uf.find(entry_id)].add(entry_id)

    # Assign component strata. A component touched by hard labels is treated as
    # hard-negative first; this preserves the hard pair even when it overlaps a
    # multi-positive record.
    strata: dict[str, list[set[str]]] = defaultdict(list)
    for component in components.values():
        if component & hard_ids:
            stratum = "hard_negative"
        elif component & multi_ids:
            stratum = "multi_positive"
        else:
            stratum = "single"
        strata[stratum].append(component)

    rng = random.Random(args.seed)
    assignments: dict[str, str] = {}
    for units in strata.values():
        assignments.update(allocate(units, ratios, rng))
    if set(assignments) != set(all_ids):
        raise AssertionError("not every dataset record received a split")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {
        split: [row for row in dataset if assignments[str(row["entry_id"])] == split]
        for split in SPLITS
    }
    for split, rows in split_rows.items():
        (output_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # The relation files inherit the master split; they are not split independently.
    multi_split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for index, row in enumerate(multi_rows):
        split = assignments[multi_row_to_id[index]]
        tagged = dict(row)
        tagged["split"] = split
        tagged["resolved_entry_id"] = multi_row_to_id[index]
        multi_split_rows[split].append(tagged)
    for split, rows in multi_split_rows.items():
        (output_dir / f"multi_positives_{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    hard_split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    hard_group_splits: dict[str, str] = {}
    for group_id, group_ids in hard_groups.items():
        group_splits = {assignments[entry_id] for entry_id in group_ids}
        if len(group_splits) != 1:
            raise AssertionError(f"hard group split across sets: {group_id} -> {group_splits}")
        hard_group_splits[group_id] = next(iter(group_splits))
    for row in hard_rows:
        tagged = dict(row)
        tagged["split"] = hard_group_splits[str(row["group_id"])]
        hard_split_rows[tagged["split"]].append(tagged)
    for split, rows in hard_split_rows.items():
        (output_dir / f"hard_negatives_{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "seed": args.seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "record_split": assignments,
        "component_id": {entry_id: uf.find(entry_id) for entry_id in all_ids},
        "multi_positive_entry_ids": sorted(multi_ids),
        "hard_negative_group_ids": sorted(hard_groups),
        "hard_negative_group_split": hard_group_splits,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report: dict[str, Any] = {
        "dataset_records": len(dataset),
        "multi_positive_rows": len(multi_rows),
        "multi_positive_records": len(multi_ids),
        "hard_negative_rows": len(hard_rows),
        "hard_negative_groups": len(hard_groups),
        "relation_components": len(components),
        "overlap_components": sum(
            bool(component & multi_ids) and bool(component & hard_ids)
            for component in components.values()
        ),
        "splits": {
            split: {
                "records": len(split_rows[split]),
                "multi_positive_rows": len(multi_split_rows[split]),
                "hard_negative_rows": len(hard_split_rows[split]),
                "hard_negative_groups": sum(
                    value == split for value in hard_group_splits.values()
                ),
            }
            for split in SPLITS
        },
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
