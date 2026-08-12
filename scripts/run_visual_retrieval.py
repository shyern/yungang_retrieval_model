#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmmrag.io import read_jsonl, write_jsonl
from mmmrag.metrics import retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP text-to-image retrieval")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, CLIPModel
    except ImportError as exc:
        raise SystemExit("Install visual dependencies with: python -m pip install -e '.[vision]'") from exc

    pairs = read_jsonl(args.pairs)
    missing = [item["image_path"] for item in pairs if not Path(item["image_path"]).is_file()]
    if missing:
        raise SystemExit(f"Missing image files (first 3): {missing[:3]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    unique_images = {item["image_id"]: item["image_path"] for item in pairs}
    image_ids = list(unique_images)

    def encode_images():
        chunks = []
        for start in range(0, len(image_ids), args.batch_size):
            images = [Image.open(unique_images[item]).convert("RGB") for item in image_ids[start : start + args.batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
            chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        return torch.cat(chunks)

    image_features = encode_images()
    outputs, rankings = [], []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        inputs = processor(text=[item["query"] for item in batch], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            text_features = torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1).cpu()
        scores = text_features @ image_features.T
        for item, row in zip(batch, scores):
            order = row.argsort(descending=True).tolist()
            ranked_ids = [image_ids[index] for index in order]
            rankings.append(ranked_ids)
            outputs.append({"pair_id": item["pair_id"], "gold_image_id": item["image_id"], "ranked_image_ids": ranked_ids, "scores": [row[index].item() for index in order]})

    write_jsonl(args.output, outputs)
    print(json.dumps(retrieval_metrics(rankings, [{item["image_id"]} for item in pairs]), indent=2))


if __name__ == "__main__":
    main()

