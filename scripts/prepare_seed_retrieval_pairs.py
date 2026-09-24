#!/usr/bin/env python3
"""Convert a seed dataset JSON split into Chinese-CLIP retrieval JSONL pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-duplicate-images",
        action="store_true",
        help="Allow one image to appear in multiple independently labeled pairs",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing images instead of failing the conversion",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        help="Optional JSON file recording entries skipped due to missing images",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    project_root = Path(__file__).resolve().parents[1]
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"Expected a JSON array in {input_path}")

    pairs = []
    missing = []
    seen_images: set[str] = set()
    for record in records:
        entry_id = str(record["entry_id"])
        query = str(record["description"]).strip()
        if not query:
            raise RuntimeError(f"Empty description for {entry_id}")
        image_value = record["image_path"]
        image_paths = [image_value] if isinstance(image_value, str) else image_value
        if not isinstance(image_paths, list) or not image_paths:
            raise RuntimeError(f"Invalid image_path for {entry_id}")
        positive_group_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

        volume_match = re.match(r"v(\d{2})_", entry_id)
        cave_id = f"volume_{volume_match.group(1)}" if volume_match else "unknown"
        for image_index, image_path_value in enumerate(image_paths, start=1):
            relative_path = Path(str(image_path_value))
            resolved_path = (
                relative_path
                if relative_path.is_absolute()
                else project_root / "seed_dataset" / relative_path
            ).resolve()
            if not resolved_path.is_file():
                if args.skip_missing:
                    missing.append(
                        {
                            "entry_id": entry_id,
                            "image_path": str(image_path_value),
                            "resolved_path": str(resolved_path),
                        }
                    )
                    continue
                raise RuntimeError(f"Missing image: {resolved_path}")
            image_path = str(resolved_path)
            if image_path in seen_images and not args.allow_duplicate_images:
                raise RuntimeError(f"Duplicate image in split: {image_path}")
            seen_images.add(image_path)

            pair_id = (
                entry_id
                if len(image_paths) == 1
                else f"{entry_id}__image_{image_index:02d}"
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "query": query,
                    "image_path": image_path,
                    "image_id": pair_id,
                    "object_id": entry_id,
                    "cave_id": cave_id,
                    "category": record["category"],
                    "split": record.get("split", "unknown"),
                    "query_source": "description",
                    "positive_group_id": positive_group_id,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
    if args.missing_output:
        args.missing_output.parent.mkdir(parents=True, exist_ok=True)
        args.missing_output.write_text(
            json.dumps(missing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"records={len(records)} pairs={len(pairs)} missing={len(missing)} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
