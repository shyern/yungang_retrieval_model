#!/usr/bin/env python3
"""Create relation-safe train/validation/test splits for the final dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")


def paths(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)} if value is not None else set()


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = self.parent[item]
        if root != item:
            self.parent[item] = self.find(root)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def allocate(
    components: list[set[str]],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, str]:
    """Greedily allocate whole components while approximating record ratios."""
    shuffled = list(components)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    total = sum(map(len, shuffled))
    target = {split: total * ratio for split, ratio in zip(SPLITS, ratios)}
    current = {split: 0 for split in SPLITS}
    assignment: dict[str, str] = {}
    for component in shuffled:
        split = max(
            SPLITS,
            key=lambda name: (target[name] - current[name]) / max(target[name], 1.0),
        )
        for entry_id in component:
            assignment[entry_id] = split
        current[split] += len(component)
    return assignment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/final_dataset/dataset_review.json"))
    parser.add_argument("--multi", type=Path, default=Path("data/final_dataset/muti_positives.json"))
    parser.add_argument("--hard", type=Path, default=Path("data/final_dataset/hard_negatives.reviewed.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/final_dataset/split"))
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must be positive and sum to 1")

    dataset = load_json(args.dataset)
    multi = load_json(args.multi)
    hard = load_json(args.hard)
    by_id = {str(row["entry_id"]): row for row in dataset}
    if len(by_id) != len(dataset):
        raise ValueError("dataset_review.json contains duplicate entry_id values")
    all_ids = list(by_id)

    image_owners: dict[str, set[str]] = defaultdict(set)
    description_owners: dict[str, set[str]] = defaultdict(set)
    for row in dataset:
        entry_id = str(row["entry_id"])
        for image in paths(row.get("image_path")):
            image_owners[image].add(entry_id)
        description_owners[str(row.get("description", ""))].add(entry_id)

    uf = UnionFind(all_ids)
    multi_ids: set[str] = set()
    multi_group_ids: dict[str, set[str]] = defaultdict(set)
    for row in multi:
        entry_id = str(row["entry_id"])
        if entry_id not in by_id:
            raise ValueError(f"Multi-positive entry_id not in dataset: {entry_id}")
        multi_ids.add(entry_id)
        # The strict subset has one record per image; the description defines
        # the positive group, so all same-description records are connected.
        group_key = str(row.get("description", ""))
        multi_group_ids[group_key].add(entry_id)
    for group in multi_group_ids.values():
        group = sorted(group)
        for entry_id in group[1:]:
            uf.union(group[0], entry_id)

    hard_group_ids: dict[str, set[str]] = defaultdict(set)
    hard_row_components: dict[int, set[str]] = {}
    for index, row in enumerate(hard):
        entry_id = str(row["entry_id"])
        if entry_id not in by_id:
            raise ValueError(f"Hard-negative entry_id not in dataset: {entry_id}")
        related = {entry_id}
        related.update(image_owners.get(str(row["positive_image_path"]), set()))
        for image in paths(row.get("negative_image_paths")):
            owners = image_owners.get(image, set())
            if not owners:
                raise ValueError(f"Negative image not in dataset_review.json: {image}")
            related.update(owners)
        # The reviewed file may omit group_id; in that case the entry_id is
        # the relation key, while shared images still merge connected rows.
        hard_group = str(row.get("group_id") or entry_id)
        hard_group_ids[hard_group].update(related)
        hard_row_components[index] = related
        related = sorted(related)
        for other in related[1:]:
            uf.union(related[0], other)

    # Keep any shared image or duplicate description together, including
    # overlaps between multi-positive and hard-negative annotations.
    for owners in list(image_owners.values()) + list(description_owners.values()):
        owners = sorted(owners)
        for other in owners[1:]:
            uf.union(owners[0], other)

    components: dict[str, set[str]] = defaultdict(set)
    for entry_id in all_ids:
        components[uf.find(entry_id)].add(entry_id)
    component_list = list(components.values())
    assignment = allocate(component_list, ratios, random.Random(args.seed))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in dataset:
        tagged = dict(row)
        tagged["split"] = assignment[str(row["entry_id"])]
        split_records[tagged["split"]].append(tagged)
    for split, rows in split_records.items():
        (out / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    multi_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in multi:
        tagged = dict(row)
        tagged["split"] = assignment[str(row["entry_id"])]
        multi_records[tagged["split"]].append(tagged)
    for split, rows in multi_records.items():
        (out / f"multi_positives_{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    hard_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    hard_group_split: dict[str, str] = {}
    for group_id, related in hard_group_ids.items():
        splits = {assignment[entry_id] for entry_id in related}
        if len(splits) != 1:
            raise AssertionError(f"Hard group crosses splits: {group_id}: {splits}")
        hard_group_split[group_id] = next(iter(splits))
    for row in hard:
        group_id = str(row.get("group_id") or row["entry_id"])
        tagged = dict(row)
        tagged["split"] = hard_group_split[group_id]
        hard_records[tagged["split"]].append(tagged)
    for split, rows in hard_records.items():
        (out / f"hard_negatives_{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "seed": args.seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "source_files": {
            "dataset": str(args.dataset),
            "dataset_sha256": sha256(args.dataset),
            "multi_positives": str(args.multi),
            "multi_positives_sha256": sha256(args.multi),
            "hard_negatives": str(args.hard),
            "hard_negatives_sha256": sha256(args.hard),
        },
        "record_split": assignment,
        "component_id": {entry_id: uf.find(entry_id) for entry_id in all_ids},
        "multi_positive_description_groups": len(multi_group_ids),
        "hard_negative_groups": len(hard_group_ids),
        "hard_negative_group_split": hard_group_split,
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "dataset_records": len(dataset),
        "multi_positive_records": len(multi),
        "multi_positive_description_groups": len(multi_group_ids),
        "hard_negative_records": len(hard),
        "hard_negative_groups": len(hard_group_ids),
        "relation_components": len(component_list),
        "components_touching_multi_and_hard": sum(
            bool(component & multi_ids)
            and any(component & related for related in hard_group_ids.values())
            for component in component_list
        ),
        "splits": {
            split: {
                "records": len(split_records[split]),
                "multi_positive_records": len(multi_records[split]),
                "hard_negative_records": len(hard_records[split]),
                "hard_negative_groups": sum(
                    value == split for value in hard_group_split.values()
                ),
            }
            for split in SPLITS
        },
    }
    (out / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
