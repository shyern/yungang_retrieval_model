#!/usr/bin/env python3
"""Fine-tune Chinese-CLIP and retain reproducible best/last checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from mmmrag.chinese_clip_local import (
    build_token_weight_predictor,
    combine_global_local_scores,
    compact_text_tokens,
    late_interaction_scores,
    predict_token_weights,
    project_local_features,
    symmetric_contrastive_loss,
    valid_text_token_mask,
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_query(record: dict[str, Any]) -> tuple[str, str]:
    if str(record.get("query", "")).strip():
        return str(record["query"]).strip(), str(record.get("query_source", "query"))
    if str(record.get("visual_text", "")).strip():
        return str(record["visual_text"]).strip(), "visual_text"
    if str(record.get("raw_text", "")).strip():
        return str(record["raw_text"]).strip(), "raw_text_fallback"
    if str(record.get("description", "")).strip():
        return str(record["description"]).strip(), "description"
    if str(record.get("name", "")).strip():
        return str(record["name"]).strip(), "name_fallback"
    record_id = record.get("id") or record.get("entry_id", "<unknown>")
    raise ValueError(f"Record {record_id} has no usable query")


def record_image(record: dict[str, Any], project_root: Path) -> Path:
    value = record.get("image_path") or record.get("image")
    if not value and record.get("images"):
        images = record["images"]
        if len(images) != 1:
            record_id = record.get("id") or record.get("entry_id", "<unknown>")
            raise ValueError(
                f"Record {record_id} has {len(images)} images; expected exactly one"
            )
        value = images[0].get("path") if isinstance(images[0], dict) else images[0]
    if not value:
        record_id = record.get("id") or record.get("entry_id", "<unknown>")
        raise ValueError(f"Record {record_id} has no image path")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", default="chinese_clip_finetune")
    parser.add_argument(
        "--model",
        default="track_a_experiments/models/pretrained/OFA-Sys--chinese-clip-vit-base-patch16",
    )
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--projection-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-text-layers", type=int, default=2)
    parser.add_argument("--train-vision-layers", type=int, default=0)
    parser.add_argument(
        "--local-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the token-patch contrastive loss; 0 preserves global CLIP training",
    )
    parser.add_argument(
        "--local-score-weight",
        type=float,
        help="Validation score weight; defaults to --local-loss-weight",
    )
    parser.add_argument("--local-max-tokens", type=int, default=32)
    parser.add_argument("--local-patch-pool", type=int, default=2)
    parser.add_argument("--local-query-chunk-size", type=int, default=4)
    parser.add_argument(
        "--token-weighting",
        action="store_true",
        help="Use a trainable token importance predictor in local alignment",
    )
    parser.add_argument("--token-weight-hidden", type=int, default=192)
    parser.add_argument("--token-weight-temperature", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_for_small_data(model: Any, text_layers: int, vision_layers: int) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_groups = ["text_projection", "visual_projection", "logit_scale"]
    for parameter in model.text_projection.parameters():
        parameter.requires_grad = True
    for parameter in model.visual_projection.parameters():
        parameter.requires_grad = True
    model.logit_scale.requires_grad = True

    if text_layers:
        for layer in model.text_model.encoder.layer[-text_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        trainable_groups.append(f"text_model.encoder.layer[-{text_layers}:]")

    if vision_layers:
        for layer in model.vision_model.encoder.layers[-vision_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.vision_model.post_layernorm.parameters():
            parameter.requires_grad = True
        trainable_groups.append(f"vision_model.encoder.layers[-{vision_layers}:]")
    return trainable_groups


def retrieval_metrics(similarities: Any) -> dict[str, float]:
    import torch

    count = similarities.shape[0]
    targets = torch.arange(count)
    values: dict[str, float] = {}
    all_reciprocal_ranks = []
    all_ranks = []
    for direction, scores in (("text_to_image", similarities), ("image_to_text", similarities.T)):
        order = scores.argsort(dim=1, descending=True)
        ranks = (order == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        all_ranks.append(ranks.float())
        all_reciprocal_ranks.append(1.0 / ranks.float())
        for k in (1, 5, 10):
            values[f"{direction}_recall@{k}"] = (ranks <= k).float().mean().item()
        values[f"{direction}_mrr"] = (1.0 / ranks.float()).mean().item()
        values[f"{direction}_mean_rank"] = ranks.float().mean().item()
    values["mean_recall"] = sum(
        values[f"{direction}_recall@{k}"]
        for direction in ("text_to_image", "image_to_text")
        for k in (1, 5, 10)
    ) / 6.0
    values["mrr"] = torch.cat(all_reciprocal_ranks).mean().item()
    values["mean_rank"] = torch.cat(all_ranks).mean().item()
    return values


def save_checkpoint(
    model: Any,
    processor: Any,
    path: Path,
    token_weight_predictor: Any | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    if token_weight_predictor is not None:
        import torch

        torch.save(token_weight_predictor.state_dict(), path / "token_weight_predictor.pt")


def checkpoint_sha256(path: Path) -> str:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        weights = path / filename
        if weights.is_file():
            return sha256(weights)
    raise FileNotFoundError(f"No model weights found in {path}")


def main() -> None:
    args = parse_args()
    if args.local_loss_weight < 0:
        raise SystemExit("--local-loss-weight cannot be negative")
    if args.local_score_weight is None:
        args.local_score_weight = args.local_loss_weight
    if args.local_score_weight < 0:
        raise SystemExit("--local-score-weight cannot be negative")
    if args.token_weight_hidden < 1:
        raise SystemExit("--token-weight-hidden must be positive")
    if args.token_weight_temperature <= 0:
        raise SystemExit("--token-weight-temperature must be positive")
    try:
        import torch
        import transformers
        from PIL import Image, __version__ as pillow_version
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            ChineseCLIPModel,
            ChineseCLIPProcessor,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import the visual training stack. "
            f"Install compatible vision dependencies or fix the runtime libraries: {exc}"
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    train_path = Path(args.train).resolve()
    validation_path = Path(args.validation).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite_output:
            raise SystemExit(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "run.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is unavailable; pass --allow-cpu to explicitly train on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_records = read_jsonl(train_path)
    validation_records = read_jsonl(validation_path)
    processor = ChineseCLIPProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=False
    )
    model = ChineseCLIPModel.from_pretrained(model_path, local_files_only=True)
    text_layers = len(model.text_model.encoder.layer)
    vision_layers = len(model.vision_model.encoder.layers)
    if not 0 <= args.train_text_layers <= text_layers:
        raise SystemExit(f"--train-text-layers must be between 0 and {text_layers}")
    if not 0 <= args.train_vision_layers <= vision_layers:
        raise SystemExit(f"--train-vision-layers must be between 0 and {vision_layers}")
    if args.local_max_tokens < 1:
        raise SystemExit("--local-max-tokens must be positive")
    if args.local_patch_pool < 1:
        raise SystemExit("--local-patch-pool must be positive")
    patch_grid_size = (
        model.config.vision_config.image_size // model.config.vision_config.patch_size
    )
    if patch_grid_size % args.local_patch_pool:
        raise SystemExit(
            f"--local-patch-pool must divide the {patch_grid_size}x{patch_grid_size} patch grid"
        )
    if args.local_query_chunk_size < 1:
        raise SystemExit("--local-query-chunk-size must be positive")
    trainable_groups = freeze_for_small_data(
        model, args.train_text_layers, args.train_vision_layers
    )
    model.to(device)
    special_token_ids = tuple(processor.tokenizer.all_special_ids)
    token_weight_predictor = None
    if args.token_weighting:
        token_weight_predictor = build_token_weight_predictor(
            model.config.text_config.hidden_size, args.token_weight_hidden
        ).to(device)

    class PairDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> tuple[str, Any]:
            record = self.records[index]
            query, _ = record_query(record)
            with Image.open(record_image(record, project_root)) as image:
                return query, image.convert("RGB")

    def collate(batch: list[tuple[str, Any]]) -> dict[str, Any]:
        texts, images = zip(*batch)
        return processor(
            text=list(texts),
            images=list(images),
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        PairDataset(validation_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    projection_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("text_projection") or name.startswith("visual_projection") or name == "logit_scale")
    ]
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not (name.startswith("text_projection") or name.startswith("visual_projection") or name == "logit_scale")
    ]
    predictor_parameters = (
        list(token_weight_predictor.parameters())
        if token_weight_predictor is not None
        else []
    )
    parameter_groups = [
        {"params": projection_parameters, "lr": args.projection_learning_rate},
    ]
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": args.learning_rate})
    if predictor_parameters:
        parameter_groups.append(
            {"params": predictor_parameters, "lr": args.projection_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_parameter_count += sum(parameter.numel() for parameter in predictor_parameters)
    config = {
        "experiment": args.experiment,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "model": {
            "path": str(model_path),
            "revision": args.model_revision,
            "weights_sha256": checkpoint_sha256(model_path),
            "class": "ChineseCLIPModel",
            "processor_class": "ChineseCLIPProcessor",
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "trainable_fraction": trainable_parameter_count / parameter_count,
        },
        "data": {
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
            "train_records": len(train_records),
            "train_needs_review": sum(bool(item.get("needs_review")) for item in train_records),
            "validation_path": str(validation_path),
            "validation_sha256": sha256(validation_path),
            "validation_records": len(validation_records),
            "validation_needs_review": sum(
                bool(item.get("needs_review")) for item in validation_records
            ),
            "query_policy": (
                "query, else visual_text, else raw_text, else description, else name fallback"
            ),
        },
        "training": {
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "encoder_learning_rate": args.learning_rate,
            "projection_learning_rate": args.projection_learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "cosine",
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_steps": total_steps,
            "max_length": args.max_length,
            "gradient_clip": args.gradient_clip,
            "seed": args.seed,
            "precision": "fp32",
            "device": str(device),
            "trainable_groups": trainable_groups,
            "frozen_vision_encoder": args.train_vision_layers == 0,
            "local_alignment": {
                "enabled": args.local_loss_weight > 0,
                "loss_weight": args.local_loss_weight,
                "validation_score_weight": args.local_score_weight,
                "max_text_tokens": args.local_max_tokens,
                "patch_pool": args.local_patch_pool,
                "pooled_patch_grid": patch_grid_size // args.local_patch_pool,
                "query_chunk_size": args.local_query_chunk_size,
                "interaction": "symmetric_token_patch_maxsim",
                "projection": "shared_chinese_clip_projection_layers",
                "token_weighting": {
                    "enabled": args.token_weighting,
                    "hidden_size": args.token_weight_hidden,
                    "temperature": args.token_weight_temperature,
                    "normalization": "masked_mean_one",
                },
            },
            "early_stopping": {
                "metric": "validation_mrr",
                "mode": "max",
                "patience": args.patience,
                "min_delta": args.min_delta,
            },
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "pillow": pillow_version,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    def local_features(output: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        text_tokens, image_patches = project_local_features(
            model,
            output.text_model_output.last_hidden_state,
            output.vision_model_output.last_hidden_state,
            patch_pool=args.local_patch_pool,
        )
        token_mask = valid_text_token_mask(
            batch["input_ids"], batch["attention_mask"], special_token_ids
        )
        text_tokens, token_mask = compact_text_tokens(
            text_tokens, token_mask, args.local_max_tokens
        )
        token_weights = None
        if token_weight_predictor is not None:
            token_weights = predict_token_weights(
                token_weight_predictor,
                output.text_model_output.last_hidden_state,
                valid_text_token_mask(
                    batch["input_ids"], batch["attention_mask"], special_token_ids
                ),
                temperature=args.token_weight_temperature,
            )
            _, token_weights = compact_text_tokens(
                token_weights.unsqueeze(-1),
                valid_text_token_mask(
                    batch["input_ids"], batch["attention_mask"], special_token_ids
                ),
                args.local_max_tokens,
            )
            token_weights = token_weights.squeeze(-1)
        return text_tokens, image_patches, token_mask, token_weights

    def training_objective(batch: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        output = model(**batch, return_loss=True)
        global_loss = output.loss
        local_loss = global_loss.new_zeros(())
        if args.local_loss_weight > 0:
            text_tokens, image_patches, token_mask, token_weights = local_features(output, batch)
            local_logits = late_interaction_scores(
                text_tokens,
                image_patches,
                token_mask,
                text_token_weights=token_weights,
                query_chunk_size=args.local_query_chunk_size,
            ) * model.logit_scale.exp()
            local_loss = symmetric_contrastive_loss(local_logits)
        total_loss = global_loss + args.local_loss_weight * local_loss
        return output, total_loss, global_loss, local_loss

    @torch.inference_mode()
    def validate() -> tuple[dict[str, float], dict[str, float]]:
        model.eval()
        if token_weight_predictor is not None:
            token_weight_predictor.eval()
        image_features, text_features = [], []
        local_image_features, local_text_features, local_masks, local_weights = [], [], [], []
        losses = {"total": 0.0, "global": 0.0, "local": 0.0}
        for batch in validation_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output, total_loss, global_loss, local_loss = training_objective(batch)
            batch_count = batch["input_ids"].shape[0]
            losses["total"] += total_loss.item() * batch_count
            losses["global"] += global_loss.item() * batch_count
            losses["local"] += local_loss.item() * batch_count
            image_features.append(torch.nn.functional.normalize(output.image_embeds, dim=-1).cpu())
            text_features.append(torch.nn.functional.normalize(output.text_embeds, dim=-1).cpu())
            if args.local_score_weight > 0:
                text_tokens, image_patches, token_mask, token_weights = local_features(output, batch)
                local_text_features.append(text_tokens.cpu())
                local_image_features.append(image_patches.cpu())
                local_masks.append(token_mask.cpu())
                if token_weights is not None:
                    local_weights.append(token_weights.cpu())
        images = torch.cat(image_features)
        texts = torch.cat(text_features)
        global_scores = texts @ images.T
        scores = global_scores
        if args.local_score_weight > 0:
            local_scores = late_interaction_scores(
                torch.cat(local_text_features).to(device),
                torch.cat(local_image_features).to(device),
                torch.cat(local_masks).to(device),
                text_token_weights=(
                    torch.cat(local_weights).to(device)
                    if local_weights
                    else None
                ),
                query_chunk_size=args.local_query_chunk_size,
            ).cpu()
            scores = combine_global_local_scores(
                global_scores, local_scores, args.local_score_weight
            )
        metrics = retrieval_metrics(scores)
        global_metrics = retrieval_metrics(global_scores)
        metrics.update({f"global_{key}": value for key, value in global_metrics.items()})
        return (
            {key: value / len(validation_records) for key, value in losses.items()},
            metrics,
        )

    started = time.time()
    best_mrr = -math.inf
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if token_weight_predictor is not None:
            token_weight_predictor.train()
        train_loss = 0.0
        train_global_loss = 0.0
        train_local_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            _, total_loss, global_loss, local_loss = training_objective(batch)
            total_loss.backward()
            clip_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ] + predictor_parameters
            torch.nn.utils.clip_grad_norm_(clip_parameters, args.gradient_clip)
            optimizer.step()
            scheduler.step()
            batch_count = batch["input_ids"].shape[0]
            train_loss += total_loss.item() * batch_count
            train_global_loss += global_loss.item() * batch_count
            train_local_loss += local_loss.item() * batch_count
            seen += batch_count
            global_step += 1

        validation_losses, metrics = validate()
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss / seen,
            "train_global_loss": train_global_loss / seen,
            "train_local_loss": train_local_loss / seen,
            "validation_loss": validation_losses["total"],
            "validation_global_loss": validation_losses["global"],
            "validation_local_loss": validation_losses["local"],
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "validation": metrics,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        history.append(epoch_record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        if metrics["mrr"] > best_mrr + args.min_delta:
            best_mrr = metrics["mrr"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                model, processor, checkpoints_dir / "best", token_weight_predictor
            )
            selection_record = {
                **epoch_record,
                "checkpoint_weights_sha256": checkpoint_sha256(checkpoints_dir / "best"),
            }
            (checkpoints_dir / "best" / "selection.json").write_text(
                json.dumps(selection_record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    save_checkpoint(model, processor, checkpoints_dir / "last", token_weight_predictor)
    summary = {
        "status": "completed",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mrr": best_mrr,
        "stopped_early": len(history) < args.epochs,
        "elapsed_seconds": round(time.time() - started, 3),
        "best_checkpoint": str(checkpoints_dir / "best"),
        "best_checkpoint_weights_sha256": checkpoint_sha256(checkpoints_dir / "best"),
        "last_checkpoint": str(checkpoints_dir / "last"),
        "last_checkpoint_weights_sha256": checkpoint_sha256(checkpoints_dir / "last"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
