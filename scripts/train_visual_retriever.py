#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with symmetric in-batch contrastive loss")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoProcessor, CLIPModel
    except ImportError as exc:
        raise SystemExit("Install visual dependencies with: python -m pip install -e '.[vision]'") from exc

    from mmmrag.io import read_jsonl

    records = read_jsonl(args.train)
    processor = AutoProcessor.from_pretrained(args.model)

    class PairDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            item = records[index]
            return item["query"], Image.open(item["image_path"]).convert("RGB")

    def collate(batch):
        texts, images = zip(*batch)
        return processor(text=list(texts), images=list(images), padding=True, return_tensors="pt")

    loader = DataLoader(PairDataset(), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch, return_loss=True)
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            running_loss += output.loss.item()
        print(f"epoch={epoch + 1} loss={running_loss / max(len(loader), 1):.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

