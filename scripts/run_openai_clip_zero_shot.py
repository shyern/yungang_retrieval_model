#!/usr/bin/env python3
"""Run OpenAI CLIP ViT-B/32 zero-shot retrieval on a dataset split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("seed_dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field", choices=("name", "description"), default="description")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    records = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit("data must be a non-empty JSON array")
    root = args.image_root.resolve()
    items = []
    missing = []
    for record in records:
        value = record.get(args.text_field)
        image_path = Path(str(record["image_path"]))
        resolved = image_path if image_path.is_absolute() else root / image_path
        if not str(value or "").strip():
            raise SystemExit(f"empty {args.text_field}: {record.get('entry_id')}")
        if not resolved.is_file():
            if args.skip_missing:
                missing.append({"entry_id": str(record["entry_id"]), "image_path": str(resolved)})
                continue
            raise SystemExit(f"missing image: {resolved}")
        query = str(value).strip()
        items.append({
            "pair_id": str(record["entry_id"]),
            "image_id": str(record["entry_id"]),
            "query": query,
            "image_path": str(resolved),
            "positive_group_id": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
            "category": record.get("category"),
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("CUDA unavailable; pass --allow-cpu")
    started = time.time()
    model = CLIPModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)

    image_features = []
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
            inputs = processor(images=images, return_tensors="pt")
            features = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            image_features.append(torch.nn.functional.normalize(features, dim=-1).cpu())
            for image in images:
                image.close()
    image_features = torch.cat(image_features)

    text_features = []
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            inputs = processor(text=[item["query"] for item in batch], padding=True,
                               truncation=True, return_tensors="pt")
            features = model.get_text_features(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
            text_features.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    text_features = torch.cat(text_features)
    scores = text_features @ image_features.T
    image_ids = [item["image_id"] for item in items]
    rankings = []
    ranks = []
    output_rows = []
    for index, item in enumerate(items):
        order = scores[index].argsort(descending=True).tolist()
        ranked_ids = [image_ids[position] for position in order]
        gold = {item["image_id"]}
        rank = next(position for position, image_id in enumerate(ranked_ids, 1) if image_id in gold)
        rankings.append(ranked_ids)
        ranks.append(rank)
        output_rows.append({**item, "gold_rank": rank, "ranked_image_ids": ranked_ids,
                            "scores": [round(scores[index, position].item(), 8) for position in order]})

    metrics = {
        "recall@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall@5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": sorted(ranks)[len(ranks) // 2],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rankings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8"
    )
    config = {"model": str(args.model.resolve()), "data": str(args.data.resolve()),
              "image_root": str(root), "text_field": args.text_field, "device": device,
              "num_queries": len(items), "gallery_size": len(items),
              "elapsed_seconds": round(time.time() - started, 3)}
    (args.output_dir / "missing_images.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2) + "\n",
                                                          encoding="utf-8")
    config["input_records"] = len(records)
    config["missing_records"] = len(missing)
    (args.output_dir / "metrics.json").write_text(json.dumps({**config, "metrics": metrics},
                                                               ensure_ascii=False, indent=2) + "\n",
                                                              encoding="utf-8")
    print(json.dumps({**config, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
