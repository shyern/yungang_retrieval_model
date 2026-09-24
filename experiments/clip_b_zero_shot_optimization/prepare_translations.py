#!/usr/bin/env python3
"""Translate the fixed test queries to English and preserve their provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    records = json.loads(args.data.read_text(encoding="utf-8"))
    valid, missing = [], []
    for record in records:
        path = args.image_root / str(record["image_path"])
        if not path.is_file():
            missing.append({"entry_id": record["entry_id"], "image_path": str(path.resolve())})
            continue
        valid.append(record)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True).eval()

    def translate(values: list[str]) -> list[str]:
        translated = []
        for start in range(0, len(values), args.batch_size):
            batch = values[start : start + args.batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512,
                               return_tensors="pt")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=256, num_beams=4)
            translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            print(f"translated={min(start + args.batch_size, len(values))}/{len(values)}", flush=True)
        return translated

    names = translate([str(row["name"]) for row in valid])
    descriptions = translate([str(row["description"]) for row in valid])
    output_rows = []
    for record, name_en, description_en in zip(valid, names, descriptions):
        output_rows.append({
            "entry_id": record["entry_id"],
            "image_path": str((args.image_root / str(record["image_path"])).resolve()),
            "category": record["category"],
            "name_zh": record["name"],
            "description_zh": record["description"],
            "name_en": name_en,
            "description_en": description_en,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.missing_output.write_text(json.dumps(missing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"valid={len(output_rows)} missing={len(missing)} output={args.output}")


if __name__ == "__main__":
    main()
