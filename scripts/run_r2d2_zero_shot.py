#!/usr/bin/env python3
"""Evaluate the official R2D2 ViT-L checkpoint for zero-shot retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_transformers_compatibility_aliases() -> None:
    """Expose three helpers at their Transformers 4.15 import locations."""
    import transformers.modeling_utils as modeling_utils
    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )

    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    modeling_utils.prune_linear_layer = prune_linear_layer


def load_model(repo: Path, checkpoint_path: Path, device, adapter_path: Path | None = None):
    import torch

    install_transformers_compatibility_aliases()
    sys.path.insert(0, str(repo))
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)
        from models.r2d2 import R2D2

        model = R2D2(vit_type="large", image_size=224, embed_dim=768)
    finally:
        os.chdir(previous_cwd)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    model_state = model.state_dict()
    incompatible_shapes = []
    for key in list(state_dict):
        if key in model_state and state_dict[key].shape != model_state[key].shape:
            incompatible_shapes.append(key)
            del state_dict[key]
    load_message = model.load_state_dict(state_dict, strict=False)
    details = {
        "missing_keys": list(load_message.missing_keys),
        "unexpected_keys": list(load_message.unexpected_keys),
        "incompatible_shape_keys": incompatible_shapes,
    }
    model = model.to(device)
    if adapter_path is not None:
        from scripts.train_external_clip_lora import load_external_lora
        details["adapter"] = load_external_lora(model, adapter_path, "r2d2")
    return model.eval(), details


def preprocess_image(image_path: str):
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    with Image.open(image_path) as image:
        return transform(image.convert("RGB"))


def compute_features(model, rows, device, image_batch_size, text_batch_size):
    import torch

    image_tokens, image_globals = [], []
    for start in range(0, len(rows), image_batch_size):
        images = torch.stack([
            preprocess_image(row["resolved_image_path"])
            for row in rows[start : start + image_batch_size]
        ]).to(device)
        with torch.inference_mode():
            tokens, embeddings = model.encode_image(images)
        image_tokens.append(tokens.cpu())
        image_globals.append(embeddings.cpu())
        print(f"images={min(start + image_batch_size, len(rows))}/{len(rows)}", flush=True)

    text_tokens, text_masks, text_globals = [], [], []
    for start in range(0, len(rows), text_batch_size):
        texts = [row["description"] for row in rows[start : start + text_batch_size]]
        tokenized = model.tokenize_text(texts).to(device)
        with torch.inference_mode():
            tokens, embeddings = model.encode_text(tokenized)
        text_tokens.append(tokens.cpu())
        text_masks.append(tokenized.attention_mask.cpu())
        text_globals.append(embeddings.cpu())
        print(f"queries={min(start + text_batch_size, len(rows))}/{len(rows)}", flush=True)

    return {
        "image_tokens": torch.cat(image_tokens),
        "image_globals": torch.cat(image_globals),
        "text_tokens": torch.cat(text_tokens),
        "text_masks": torch.cat(text_masks),
        "text_globals": torch.cat(text_globals),
    }


def rerank(
    model, similarities, features, device, top_k, candidate_batch_size,
    direction, progress_path,
):
    import torch

    if progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        scores = progress["scores"]
        completed = int(progress["completed"])
        print(f"resuming {direction} at query {completed + 1}", flush=True)
    else:
        scores = torch.full_like(similarities, -100.0)
        completed = 0
    for query_index in range(completed, len(similarities)):
        query_scores = similarities[query_index]
        candidate_scores, candidate_indices = query_scores.topk(k=top_k)
        for start in range(0, top_k, candidate_batch_size):
            indices = candidate_indices[start : start + candidate_batch_size]
            global_scores = candidate_scores[start : start + candidate_batch_size].to(device)
            count = len(indices)
            if direction == "image_to_text":
                image_tokens = features["image_tokens"][query_index].repeat(count, 1, 1).to(device)
                text_tokens = features["text_tokens"][indices].to(device)
                text_masks = features["text_masks"][indices].to(device)
            else:
                image_tokens = features["image_tokens"][indices].to(device)
                text_tokens = features["text_tokens"][query_index].repeat(count, 1, 1).to(device)
                text_masks = features["text_masks"][query_index].repeat(count, 1).to(device)
            image_masks = torch.ones(image_tokens.shape[:-1], dtype=torch.long, device=device)
            with torch.inference_mode():
                text_output = model.text_joint_layer(
                    encoder_embeds=text_tokens,
                    attention_mask=text_masks,
                    encoder_hidden_states=image_tokens,
                    encoder_attention_mask=image_masks,
                    return_dict=True,
                )
                image_output = model.img_joint_layer(
                    encoder_embeds=model.img_joint_proj(image_tokens),
                    attention_mask=image_masks,
                    encoder_hidden_states=text_tokens,
                    encoder_attention_mask=text_masks,
                    return_dict=True,
                )
                match_score = (
                    torch.softmax(model.itm_head(text_output.last_hidden_state[:, 0]), dim=-1)[:, 1]
                    + torch.softmax(model.itm_head_i(image_output.last_hidden_state[:, 0]), dim=-1)[:, 1]
                ) / 2
            scores[query_index, indices] = (match_score + global_scores).cpu()
        torch.save({"scores": scores, "completed": query_index + 1}, progress_path)
        print(f"{direction}={query_index + 1}/{len(similarities)}", flush=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()

    import torch

    if args.data.suffix == ".jsonl":
        source = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        source = json.loads(args.data.read_text(encoding="utf-8"))
    rows, missing = [], []
    for record in source:
        raw_path = Path(str(record["image_path"]))
        image_path = raw_path if raw_path.is_absolute() else args.image_root / raw_path
        entry_id = record.get("entry_id", record.get("pair_id"))
        if not image_path.is_file():
            missing.append({"entry_id": entry_id, "image_path": str(image_path.resolve())})
            continue
        rows.append({
            **record,
            "entry_id": entry_id,
            "description": record.get("description", record.get("query")),
            "resolved_image_path": str(image_path.resolve()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, load_details = load_model(
        args.repo.resolve(), args.checkpoint.resolve(), device,
        args.adapter.resolve() if args.adapter else None,
    )
    (args.output_dir / "checkpoint_load.json").write_text(
        json.dumps(load_details, indent=2) + "\n", encoding="utf-8"
    )

    feature_cache = args.output_dir / "feature_cache.pt"
    if feature_cache.is_file():
        features = torch.load(feature_cache, map_location="cpu", weights_only=True)
        print(f"loaded feature cache: {feature_cache}", flush=True)
    else:
        features = compute_features(
            model, rows, device, args.image_batch_size, args.text_batch_size
        )
        torch.save(features, feature_cache)

    similarities = features["image_globals"] @ features["text_globals"].T
    i2t_cache = args.output_dir / "scores_image_to_text.pt"
    if i2t_cache.is_file():
        image_to_text_scores = torch.load(i2t_cache, map_location="cpu", weights_only=True)
    else:
        image_to_text_scores = rerank(
            model, similarities, features, device, args.top_k,
            args.candidate_batch_size, "image_to_text",
            args.output_dir / "progress_image_to_text.pt",
        )
        torch.save(image_to_text_scores, i2t_cache)
    t2i_cache = args.output_dir / "scores_text_to_image.pt"
    if t2i_cache.is_file():
        text_to_image_scores = torch.load(t2i_cache, map_location="cpu", weights_only=True)
    else:
        text_to_image_scores = rerank(
            model, similarities.T, features, device, args.top_k,
            args.candidate_batch_size, "text_to_image",
            args.output_dir / "progress_text_to_image.pt",
        )
        torch.save(text_to_image_scores, t2i_cache)

    groups = defaultdict(set)
    for index, row in enumerate(rows):
        groups[row["description"]].add(index)
    gold = [groups[row["description"]] for row in rows]
    t2i, t2i_ranks, t2i_order = score_metrics(text_to_image_scores, gold)
    i2t, i2t_ranks, i2t_order = score_metrics(image_to_text_scores, gold)
    ids = [row["entry_id"] for row in rows]
    mean_recall = sum(t2i[f"recall@{k}"] + i2t[f"recall@{k}"] for k in (1, 5, 10)) / 6
    report = {
        "model": "R2D2 ViT-L 250M",
        "official_repository": "https://github.com/yuxie11/R2D2",
        "repo_revision": args.repo_revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "checkpoint_sha256": sha256(args.checkpoint),
        "scoring": f"official global pre-ranking plus bidirectional ITM reranking, top_k={args.top_k}",
        "data": str(args.data.resolve()), "device": str(device),
        "input_records": len(source), "samples": len(rows), "missing_records": len(missing),
        "positive_group_key": "exact Chinese description", "positive_groups": len(groups),
        "multi_positive_groups": sum(len(value) > 1 for value in groups.values()),
        "records_in_multi_positive_groups": sum(len(value) for value in groups.values() if len(value) > 1),
        "text_to_image": t2i, "image_to_text": i2t, "mean_recall": mean_recall,
    }
    (args.output_dir / "bidirectional_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "missing_images.json").write_text(
        json.dumps(missing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for filename, ranks, orders in (
        ("rankings_text_to_image.jsonl", t2i_ranks, t2i_order),
        ("rankings_image_to_text.jsonl", i2t_ranks, i2t_order),
    ):
        values = (
            {"entry_id": ids[index], "positive_count": len(gold[index]),
             "gold_rank": ranks[index], "ranked_entry_ids": [ids[item] for item in orders[index]]}
            for index in range(len(rows))
        )
        (args.output_dir / filename).write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
