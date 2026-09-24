#!/usr/bin/env python3
"""Evaluate AltCLIP zero-shot retrieval in both directions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def score_metrics(scores, gold):
    ranks, rankings = [], []
    for row, relevant in zip(scores, gold):
        order = row.argsort(descending=True).tolist()
        ranks.append(min(position for position, index in enumerate(order, 1) if index in relevant))
        rankings.append(order)
    return {
        "recall@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall@5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
    }, ranks, rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AltCLIPModel, AltCLIPProcessor

    source = json.loads(args.data.read_text(encoding="utf-8"))
    rows, missing = [], []
    for record in source:
        image_path = args.image_root / str(record["image_path"])
        if not image_path.is_file():
            missing.append({"entry_id": record["entry_id"], "image_path": str(image_path.resolve())})
            continue
        rows.append({**record, "resolved_image_path": str(image_path.resolve())})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AltCLIPModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    processor = AltCLIPProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)

    image_chunks = []
    for start in range(0, len(rows), args.batch_size):
        images = []
        for row in rows[start : start + args.batch_size]:
            with Image.open(row["resolved_image_path"]) as source_image:
                images.append(source_image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
        image_chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        print(f"images={min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    image_features = torch.cat(image_chunks)

    text_chunks = []
    for start in range(0, len(rows), args.batch_size):
        inputs = processor(text=[row["description"] for row in rows[start : start + args.batch_size]],
                           padding=True, truncation=True, max_length=512, return_tensors="pt")
        text_inputs = {key: value.to(device) for key, value in inputs.items()
                       if key in {"input_ids", "attention_mask"}}
        with torch.inference_mode():
            features = model.get_text_features(**text_inputs)
        text_chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        print(f"queries={min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    text_features = torch.cat(text_chunks)

    groups = defaultdict(set)
    for index, row in enumerate(rows):
        groups[row["description"]].add(index)
    gold = [groups[row["description"]] for row in rows]
    similarities = text_features @ image_features.T
    t2i, t2i_ranks, t2i_order = score_metrics(similarities, gold)
    i2t, i2t_ranks, i2t_order = score_metrics(similarities.T, gold)
    ids = [row["entry_id"] for row in rows]
    mean_recall = sum(t2i[f"recall@{k}"] + i2t[f"recall@{k}"] for k in (1, 5, 10)) / 6

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model.resolve()), "model_revision": args.model_revision,
        "data": str(args.data.resolve()), "device": device, "input_records": len(source),
        "samples": len(rows), "missing_records": len(missing),
        "positive_group_key": "exact Chinese description", "positive_groups": len(groups),
        "multi_positive_groups": sum(len(value) > 1 for value in groups.values()),
        "records_in_multi_positive_groups": sum(len(value) for value in groups.values() if len(value) > 1),
        "text_to_image": t2i, "image_to_text": i2t, "mean_recall": mean_recall,
    }
    (args.output_dir / "bidirectional_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "missing_images.json").write_text(
        json.dumps(missing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, ranks, orders in (("rankings_text_to_image.jsonl", t2i_ranks, t2i_order),
                                    ("rankings_image_to_text.jsonl", i2t_ranks, i2t_order)):
        values = ({"entry_id": ids[index], "positive_count": len(gold[index]),
                   "gold_rank": ranks[index], "ranked_entry_ids": [ids[item] for item in orders[index]]}
                  for index in range(len(rows)))
        (args.output_dir / filename).write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
