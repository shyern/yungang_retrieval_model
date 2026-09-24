#!/usr/bin/env python3
"""Compare query and multi-crop inference strategies with frozen CLIP-B."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


PROMPTS = (
    "a photograph of {text}",
    "a detailed photograph of {text}",
    "a photograph of a Buddhist cave artwork showing {text}",
    "a close-up photograph of {text}",
)


def five_crops(image):
    width, height = image.size
    side = min(width, height)
    boxes = ((0, 0, side, side), (width - side, 0, width, side),
             (0, height - side, side, height),
             (width - side, height - side, width, height),
             ((width - side) // 2, (height - side) // 2,
              (width + side) // 2, (height + side) // 2))
    return [image.crop(box) for box in boxes]


def metrics(ranks: list[int]) -> dict[str, float]:
    return {
        "recall@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall@5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    rows = json.loads(args.data.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)

    def encode_images(multicrop: bool):
        chunks = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            views, counts = [], []
            for row in batch:
                with Image.open(row["image_path"]) as source:
                    image = source.convert("RGB")
                    selected = five_crops(image) if multicrop else [image]
                    views.extend(selected)
                    counts.append(len(selected))
            inputs = processor(images=views, return_tensors="pt")
            with torch.inference_mode():
                features = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            features = torch.nn.functional.normalize(features, dim=-1)
            if multicrop:
                features = torch.stack([part.mean(0) for part in features.split(counts)])
                features = torch.nn.functional.normalize(features, dim=-1)
            chunks.append(features.cpu())
        return torch.cat(chunks)

    def encode_text(field: str, ensemble: bool):
        all_features = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            features_per_prompt = []
            templates = PROMPTS if ensemble else ("{text}",)
            for template in templates:
                texts = [template.format(text=row[field]) for row in batch]
                inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
                with torch.inference_mode():
                    features = model.get_text_features(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                    )
                features_per_prompt.append(torch.nn.functional.normalize(features, dim=-1))
            features = torch.stack(features_per_prompt).mean(0)
            all_features.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        return torch.cat(all_features)

    started = time.time()
    image_global = encode_images(False)
    image_multicrop = encode_images(True)
    specs = (
        ("chinese_description", "description_zh", False, image_global),
        ("english_name", "name_en", False, image_global),
        ("english_description", "description_en", False, image_global),
        ("english_name_prompt_ensemble", "name_en", True, image_global),
        ("english_description_prompt_ensemble", "description_en", True, image_global),
        ("english_name_prompt_ensemble_five_crop", "name_en", True, image_multicrop),
        ("english_description_prompt_ensemble_five_crop", "description_en", True, image_multicrop),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, field, ensemble, image_features in specs:
        text_features = encode_text(field, ensemble)
        scores = text_features @ image_features.T
        ranks, ranking_rows = [], []
        ids = [row["entry_id"] for row in rows]
        for index, row_scores in enumerate(scores):
            order = row_scores.argsort(descending=True).tolist()
            rank = order.index(index) + 1
            ranks.append(rank)
            ranking_rows.append({"entry_id": ids[index], "query": rows[index][field],
                                 "gold_rank": rank,
                                 "top10": [ids[position] for position in order[:10]]})
        result = metrics(ranks)
        summary[name] = result
        (args.output_dir / f"{name}_rankings.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ranking_rows), encoding="utf-8")
        print(name, json.dumps(result), flush=True)
    report = {"model": str(args.model.resolve()), "data": str(args.data.resolve()),
              "device": device, "samples": len(rows), "prompt_templates": PROMPTS,
              "five_crop_aggregation": "mean normalized crop embeddings, then normalize",
              "elapsed_seconds": round(time.time() - started, 3), "results": summary}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
