#!/usr/bin/env python3
"""Fine-tune official Chinese-CLIP checkpoints with compact LoRA adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

from mmmrag.io import read_jsonl
from mmmrag.lora import save_lora_adapter
from mmmrag.metrics import retrieval_metrics


class Tee:
    def __init__(self, *handles: Any) -> None:
        self.handles = handles

    def write(self, value: str) -> int:
        for handle in self.handles:
            handle.write(value)
        return len(value)

    def flush(self) -> None:
        for handle in self.handles:
            handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_spec(model_path: Path) -> tuple[str, str, Path]:
    name = model_path.name.lower()
    if "base" in name:
        vision, text = "ViT-B-16", "RoBERTa-wwm-ext-base-chinese"
    elif "large" in name:
        vision, text = "ViT-L-14", "RoBERTa-wwm-ext-base-chinese"
    elif "huge" in name:
        vision, text = "ViT-H-14", "RoBERTa-wwm-ext-large-chinese"
    else:
        raise SystemExit(f"Cannot infer architecture from {model_path}")
    checkpoints = sorted(model_path.glob("clip_cn_*.pt"))
    if len(checkpoints) != 1:
        raise SystemExit(f"Expected one official checkpoint in {model_path}")
    return vision, text, checkpoints[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--vision-layers", type=int, default=4)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn.functional as functional
    from cn_clip import clip as cn_clip
    from PIL import Image, __version__ as pillow_version
    from torch.utils.data import DataLoader, Dataset

    from mmmrag.lora import inject_lora

    if args.learning_rate <= 0 or args.rank < 1 or args.epochs < 1:
        raise SystemExit("learning rate, rank, and epochs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for official Chinese-CLIP LoRA training")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_path = Path(args.train).resolve()
    validation_path = Path(args.validation).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "run.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)

    train_records = read_jsonl(train_path)
    validation_records = read_jsonl(validation_path)
    if not train_records or not validation_records:
        raise SystemExit("Training and validation data must be non-empty")
    vision_name, text_name, checkpoint_path = model_spec(model_path)
    model, preprocess = cn_clip.load_from_name(
        str(checkpoint_path),
        device=device,
        vision_model_name=vision_name,
        text_model_name=text_name,
        input_resolution=224,
    )
    selected_modules = inject_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        vision_layers=args.vision_layers,
        text_layers=args.text_layers,
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    parameter_count = sum(p.numel() for p in model.parameters())

    class PairDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> tuple[Any, str, str]:
            item = self.records[index]
            with Image.open(item["image_path"]) as image:
                pixels = preprocess(image.convert("RGB"))
            return pixels, item["query"], item["positive_group_id"]

    def collate(batch: list[tuple[Any, str, str]]) -> dict[str, Any]:
        images, texts, groups = zip(*batch)
        return {
            "images": torch.stack(images),
            "texts": cn_clip.tokenize(list(texts), context_length=52),
            "groups": list(groups),
        }

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        PairDataset(validation_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda")
    adapter_config = {
        "format": "mmmrag_official_chinese_clip_lora_v1",
        "base_model_path": str(model_path),
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": sha256(checkpoint_path),
        "vision_model": vision_name,
        "text_model": text_name,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "vision_layers": args.vision_layers,
        "text_layers": args.text_layers,
        "target_policy": "last vision MLP c_fc/c_proj and last text attention query/value",
        "target_modules": selected_modules,
    }
    config = {
        "experiment": args.experiment,
        "command": " ".join(sys.argv),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": torch.cuda.get_device_name(0),
        "data": {
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
            "train_records": len(train_records),
            "validation_path": str(validation_path),
            "validation_sha256": sha256(validation_path),
            "validation_records": len(validation_records),
        },
        "model": {
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "trainable_fraction": trainable_count / parameter_count,
            "adapter": adapter_config,
        },
        "training": {
            "loss": "standard symmetric in-batch Chinese-CLIP contrastive loss",
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "cosine",
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_steps": total_steps,
            "precision": "fp16 base with fp32 LoRA parameters and gradient scaling",
            "gradient_clip": args.gradient_clip,
            "seed": args.seed,
            "early_stopping": {
                "metric": "validation_MR",
                "mode": "max",
                "patience": args.patience,
                "min_delta": args.min_delta,
            },
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pillow": pillow_version,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    @torch.inference_mode()
    def validate() -> dict[str, float]:
        model.eval()
        image_features, text_features, groups = [], [], []
        loss_sum = 0.0
        seen = 0
        for batch in validation_loader:
            images = batch["images"].to(device, non_blocking=True)
            texts = batch["texts"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                images_encoded, texts_encoded, scale = model(images, texts)
                logits = scale * images_encoded @ texts_encoded.T
                targets = torch.arange(logits.shape[0], device=device)
                loss = (
                    functional.cross_entropy(logits, targets)
                    + functional.cross_entropy(logits.T, targets)
                ) / 2.0
            count = images.shape[0]
            loss_sum += loss.item() * count
            seen += count
            image_features.append(images_encoded.float().cpu())
            text_features.append(texts_encoded.float().cpu())
            groups.extend(batch["groups"])
        images_all = torch.cat(image_features)
        texts_all = torch.cat(text_features)
        scores = texts_all @ images_all.T
        ids = [str(index) for index in range(len(groups))]
        ids_by_group: dict[str, set[str]] = {}
        for item_id, group in zip(ids, groups):
            ids_by_group.setdefault(group, set()).add(item_id)
        text_rankings = [
            [ids[index] for index in row.argsort(descending=True).tolist()]
            for row in scores
        ]
        image_rankings = [
            [ids[index] for index in row.argsort(descending=True).tolist()]
            for row in scores.T
        ]
        gold = [ids_by_group[group] for group in groups]
        text_metrics = retrieval_metrics(text_rankings, gold)
        image_metrics = retrieval_metrics(image_rankings, gold)
        mr = sum(
            text_metrics[f"recall@{k}"] + image_metrics[f"recall@{k}"]
            for k in (1, 5, 10)
        ) / 6.0
        return {
            "loss": loss_sum / seen,
            "MR": mr,
            **{f"text_to_image_{key}": value for key, value in text_metrics.items()},
            **{f"image_to_text_{key}": value for key, value in image_metrics.items()},
        }

    started = time.time()
    best_mr = -math.inf
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    history = []
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            texts = batch["texts"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                image_features, text_features, scale = model(images, texts)
                logits = scale * image_features @ text_features.T
                targets = torch.arange(logits.shape[0], device=device)
                loss = (
                    functional.cross_entropy(logits, targets)
                    + functional.cross_entropy(logits.T, targets)
                ) / 2.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = images.shape[0]
            train_loss += loss.item() * count
            seen += count
            global_step += 1

        validation = validate()
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation["MR"] > best_mr + args.min_delta:
            best_mr = validation["MR"]
            best_epoch = epoch
            stale_epochs = 0
            best_config = {**adapter_config, "selected_epoch": epoch, "validation_MR": best_mr}
            save_lora_adapter(model, output_dir / "checkpoints" / "best", best_config)
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    save_lora_adapter(
        model,
        output_dir / "checkpoints" / "last",
        {**adapter_config, "selected_epoch": len(history), "validation_MR": history[-1]["validation"]["MR"]},
    )
    summary = {
        "status": "completed",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_MR": best_mr,
        "stopped_early": len(history) < args.epochs,
        "elapsed_seconds": round(time.time() - started, 3),
        "best_adapter": str(output_dir / "checkpoints" / "best"),
        "last_adapter": str(output_dir / "checkpoints" / "last"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
