#!/usr/bin/env python3
"""Evaluate frozen CLIP-B in both retrieval directions with multi-positives."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def retrieval_metrics(scores, gold_indices: list[set[int]]) -> tuple[dict[str, float], list[int]]:
    ranks = []
    for row, gold in zip(scores, gold_indices):
        order = row.argsort(descending=True).tolist()
        ranks.append(min(position for position, index in enumerate(order, 1) if index in gold))
    return ({
        "recall@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall@5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
    }, ranks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--text-field",
        choices=("both", "description_zh", "description_en"),
        default="both",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    rows = json.loads(args.data.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)

    image_chunks = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = []
        for row in batch:
            with Image.open(row["image_path"]) as source:
                images.append(source.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
        image_chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    image_features = torch.cat(image_chunks)

    groups: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(rows):
        groups[row["description_zh"]].add(index)
    gold = [groups[row["description_zh"]] for row in rows]

    results = {}
    rank_outputs = {}
    all_experiments = (("chinese_description", "description_zh"),
                       ("english_description", "description_en"))
    selected_experiments = (
        all_experiments
        if args.text_field == "both"
        else tuple(item for item in all_experiments if item[1] == args.text_field)
    )
    for experiment, field in selected_experiments:
        text_chunks = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            inputs = processor(text=[row[field] for row in batch], padding=True,
                               truncation=True, return_tensors="pt")
            with torch.inference_mode():
                features = model.get_text_features(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                )
            text_chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        text_features = torch.cat(text_chunks)
        similarity = text_features @ image_features.T
        t2i, t2i_ranks = retrieval_metrics(similarity, gold)
        i2t, i2t_ranks = retrieval_metrics(similarity.T, gold)
        mean_recall = sum(t2i[f"recall@{k}"] + i2t[f"recall@{k}"] for k in (1, 5, 10)) / 6
        results[experiment] = {
            "text_to_image": t2i,
            "image_to_text": i2t,
            "mean_recall": mean_recall,
        }
        rank_outputs[experiment] = [
            {"entry_id": row["entry_id"], "positive_count": len(gold[index]),
             "text_to_image_rank": t2i_ranks[index], "image_to_text_rank": i2t_ranks[index]}
            for index, row in enumerate(rows)
        ]

    report = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "device": device,
        "samples": len(rows),
        "positive_group_key": "exact original Chinese description",
        "positive_groups": len(groups),
        "multi_positive_groups": sum(len(indices) > 1 for indices in groups.values()),
        "records_in_multi_positive_groups": sum(len(indices) for indices in groups.values()
                                                 if len(indices) > 1),
        "text_field": args.text_field,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for experiment, values in rank_outputs.items():
        path = args.output.parent / f"{experiment}_bidirectional_ranks.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values),
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
