#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from mmmrag.chinese_clip_local import (
    build_token_weight_predictor,
    combine_global_local_scores,
    compact_text_tokens,
    late_interaction_scores,
    predict_token_weights,
    project_image_patches,
    project_text_tokens,
    valid_text_token_mask,
    official_cn_clip_image_patches,
    official_cn_clip_text_tokens,
)
from mmmrag.io import read_jsonl, write_jsonl
from mmmrag.metrics import retrieval_metrics


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_weights_path(model_path: Path) -> Path:
    official_checkpoints = sorted(model_path.glob("clip_cn_*.pt"))
    if len(official_checkpoints) == 1:
        return official_checkpoints[0]
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_path / filename
        if candidate.is_file():
            return candidate
    raise SystemExit(f"No model weights found in {model_path}")


def rank_metrics(ranks: list[int]) -> dict[str, float]:
    return {
        "recall@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall@5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall@10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": statistics.median(ranks),
    }


def positive_group_id(item: dict) -> str:
    return str(item.get("positive_group_id") or item["query"])


def multi_positive_metrics(
    rankings: list[list[str]], gold: list[set[str]]
) -> dict[str, float | int]:
    """Evaluate each multi-positive query at its own number of positives, R."""
    selected = [
        (ranked_ids, relevant_ids)
        for ranked_ids, relevant_ids in zip(rankings, gold)
        if len(relevant_ids) > 1
    ]
    if not selected:
        return {"map@R": 0.0, "R-Precision": 0.0, "R@1": 0.0, "queries": 0}

    average_precisions = []
    r_precisions = []
    recall_at_one = []
    for ranked_ids, relevant_ids in selected:
        r = len(relevant_ids)
        hits = 0
        precision_sum = 0.0
        for position, item_id in enumerate(ranked_ids[:r], 1):
            if item_id in relevant_ids:
                hits += 1
                precision_sum += hits / position
        average_precisions.append(precision_sum / r)
        r_precisions.append(hits / r)
        recall_at_one.append(float(bool(ranked_ids) and ranked_ids[0] in relevant_ids))

    count = len(selected)
    return {
        "map@R": sum(average_precisions) / count,
        "R-Precision": sum(r_precisions) / count,
        "R@1": sum(recall_at_one) / count,
        "queries": count,
    }


def hard_negative_metrics(
    text_results: list[dict], image_results: list[dict], hard_rows: list[dict],
    pairs: list[dict],
) -> dict[str, float | int]:
    """Strict R@1 and pairwise accuracy on explicit hard-negative sets.

    R@1 requires the positive item to occupy rank 1 in the complete gallery
    ranking.  HN accuracy remains a pairwise diagnostic: the positive must
    score strictly above every annotated hard negative.
    """
    image_by_name = {Path(str(x["image_path"])).name: str(x["image_id"]) for x in pairs}
    pair_by_text = {str(x["query"]): str(x["pair_id"]) for x in pairs}
    pair_by_image = {str(x["image_id"]): str(x["pair_id"]) for x in pairs}
    t2i_hits = []; i2t_hits = []
    t2i_ties = []; i2t_ties = []
    t2i_acc = []; i2t_acc = []
    text_by_query = {str(x["query"]): x for x in text_results}
    image_by_id = {str(x["image_id"]): x for x in image_results}
    for row in hard_rows:
        text = str(row.get("description", ""))
        positive_name = Path(str(row["positive_image_path"])).name
        positive_image = image_by_name.get(positive_name)
        negatives = row.get("negative_image_paths", [])
        if isinstance(negatives, str): negatives = [negatives]
        negative_images = [image_by_name.get(Path(str(x)).name) for x in negatives]
        negative_images = [x for x in negative_images if x is not None]
        tr = text_by_query.get(text)
        if tr is not None and positive_image is not None and negative_images:
            score = dict(zip(tr["ranked_image_ids"], tr["scores"]))
            if positive_image in score:
                ranked = tr["ranked_image_ids"]
                other_scores = [v for k, v in score.items() if k != positive_image]
                t2i_hits.append(float(bool(ranked) and ranked[0] == positive_image
                                        and positive_image in score
                                        and (not other_scores or score[positive_image] > max(other_scores))))
                best_negative = max(score[x] for x in negative_images if x in score)
                positive_score = score[positive_image]
                t2i_ties.append(float(positive_score == best_negative))
                t2i_acc.append(float(positive_score > best_negative))
        ir = image_by_id.get(positive_image) if positive_image is not None else None
        hard_texts = [pair_by_text.get(text)]
        # Every negative image is annotated as a hard negative for this text.
        # For image-to-text, use descriptions whose positive image is one of the
        # negative images when such records exist; otherwise use this row's text.
        if ir is not None and hard_texts:
            score = dict(zip(ir["ranked_pair_ids"], ir["scores"]))
            positive_pair = pair_by_image.get(positive_image)
            candidates = [x for x in hard_texts if x is not None and x in score]
            if positive_pair in score and candidates:
                # A positive image's annotated hard-negative text is the text
                # attached to each row where this image is listed as negative.
                neg_pairs = []
                for other in hard_rows:
                    other_negs = other.get("negative_image_paths", [])
                    if isinstance(other_negs, str): other_negs = [other_negs]
                    if positive_name in {Path(str(x)).name for x in other_negs}:
                        pid = pair_by_text.get(str(other.get("description", "")))
                        if pid is not None: neg_pairs.append(pid)
                neg_pairs = [x for x in set(neg_pairs) if x in score and x != positive_pair]
                if neg_pairs:
                    ranked = ir["ranked_pair_ids"]
                    other_scores = [v for k, v in score.items() if k != positive_pair]
                    i2t_hits.append(float(bool(ranked) and ranked[0] == positive_pair
                                           and positive_pair in score
                                           and (not other_scores or score[positive_pair] > max(other_scores))))
                    best_negative = max(score[x] for x in neg_pairs)
                    positive_score = score[positive_pair]
                    i2t_ties.append(float(positive_score == best_negative))
                    i2t_acc.append(float(positive_score > best_negative))
    return {
        "T2I_R@1": sum(t2i_hits) / len(t2i_hits) if t2i_hits else 0.0,
        "I2T_R@1": sum(i2t_hits) / len(i2t_hits) if i2t_hits else 0.0,
        "R@1": (sum(t2i_hits) / len(t2i_hits) + sum(i2t_hits) / len(i2t_hits)) / 2
        if t2i_hits and i2t_hits else 0.0,
        "T2I_HN_Accuracy": sum(t2i_acc) / len(t2i_acc) if t2i_acc else 0.0,
        "I2T_HN_Accuracy": sum(i2t_acc) / len(i2t_acc) if i2t_acc else 0.0,
        "HN_Accuracy": (sum(t2i_acc) / len(t2i_acc) + sum(i2t_acc) / len(i2t_acc)) / 2
        if t2i_acc and i2t_acc else 0.0,
        "ties_per_direction": {"T2I": int(sum(t2i_ties)), "I2T": int(sum(i2t_ties))},
        "tie_rate_per_direction": {
            "T2I": sum(t2i_ties) / len(t2i_ties) if t2i_ties else 0.0,
            "I2T": sum(i2t_ties) / len(i2t_ties) if i2t_ties else 0.0,
        },
        "queries_per_direction": {"T2I": len(t2i_hits), "I2T": len(i2t_hits)},
    }


def official_model_spec(model_path: Path) -> tuple[str, str]:
    name = model_path.name.lower()
    if "base" in name:
        return "ViT-B-16", "RoBERTa-wwm-ext-base-chinese"
    if "large" in name:
        return "ViT-L-14", "RoBERTa-wwm-ext-base-chinese"
    if "huge" in name:
        return "ViT-H-14", "RoBERTa-wwm-ext-large-chinese"
    raise SystemExit(f"Cannot infer official Chinese-CLIP architecture from {model_path}")


class Tee:
    def __init__(self, log_handle):
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        sys.__stdout__.write(value)
        self.log_handle.write(value)
        return len(value)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.log_handle.flush()


def evaluate(args, pairs_path: Path, model_path: Path, output_dir: Path) -> None:
    import torch
    import transformers
    from PIL import Image, __version__ as pillow_version
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    all_pairs = read_jsonl(pairs_path)
    gallery = all_pairs[: args.max_gallery] if args.max_gallery else all_pairs
    queries = (
        all_pairs
        if args.query_source == "all"
        else [item for item in all_pairs if item.get("query_source") == args.query_source]
    )
    if args.max_queries:
        queries = queries[: args.max_queries]
    if not queries or not gallery:
        raise SystemExit("Selected query or gallery set is empty")

    missing = [item["image_path"] for item in gallery if not Path(item["image_path"]).is_file()]
    if missing:
        raise SystemExit(f"Missing image files (first 3): {missing[:3]}")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is unavailable; rerun with GPU access or explicitly pass --allow-cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    started = time.time()
    backend = args.backend
    official_checkpoints = sorted(model_path.glob("clip_cn_*.pt"))
    if backend == "auto":
        backend = "official" if len(official_checkpoints) == 1 else "huggingface"
    if backend == "official":
        if len(official_checkpoints) != 1:
            raise SystemExit(f"Expected one official .pt checkpoint in {model_path}")
        weights_path = official_checkpoints[0]
    else:
        weights_path = next(
            (
                model_path / filename
                for filename in ("model.safetensors", "pytorch_model.bin")
                if (model_path / filename).is_file()
            ),
            None,
        )
        if weights_path is None:
            raise SystemExit(f"No Hugging Face model weights found in {model_path}")
    print(f"model={model_path}")
    print(f"backend={backend} weights={weights_path}")
    print(f"device={device} gpu={gpu_name}")
    print(f"queries={len(queries)} gallery={len(gallery)}")
    if backend == "official":
        if args.token_weighting:
            raise SystemExit("Token weighting is currently unavailable for the official backend")
        from cn_clip import clip as cn_clip

        vision_model_name, text_model_name = official_model_spec(model_path)
        model, official_preprocess = cn_clip.load_from_name(
            str(weights_path),
            device=device,
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
            input_resolution=224,
        )
        adapter_config = None
        if args.adapter:
            from mmmrag.lora import load_lora_adapter

            adapter_path = Path(args.adapter).resolve()
            adapter_config = load_lora_adapter(model, adapter_path)
            print(f"adapter={adapter_path}")
        local_heads = None
        if args.local_score_weight > 0:
            if not args.local_heads:
                raise SystemExit("--local-heads is required when --local-score-weight > 0")
            heads = torch.load(Path(args.local_heads).resolve(), map_location=device, weights_only=True)
            local_heads = (
                torch.nn.Linear(1280, 512, bias=False).to(device),
                torch.nn.Linear(1024, 512, bias=False).to(device),
            )
            local_heads[0].load_state_dict(heads["image_projection"])
            local_heads[1].load_state_dict(heads["text_projection"])
            for head in local_heads: head.eval()
        model.eval()
        processor = None
    else:
        adapter_config = None
        model = ChineseCLIPModel.from_pretrained(
            model_path, local_files_only=True
        ).to(device).eval()
        if args.adapter:
            from scripts.train_external_clip_lora import load_external_lora
            adapter_path = Path(args.adapter).resolve()
            adapter_config = load_external_lora(model, adapter_path, "qa_clip")
            print(f"adapter={adapter_path}")
        processor = ChineseCLIPProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=False
        )
    token_weight_predictor = None
    if args.token_weighting:
        predictor_path = model_path / "token_weight_predictor.pt"
        if not predictor_path.is_file():
            raise SystemExit(
                f"Token weighting requested but predictor is missing: {predictor_path}"
            )
        token_weight_predictor = build_token_weight_predictor(
            model.config.text_config.hidden_size, args.token_weight_hidden
        ).to(device)
        token_weight_predictor.load_state_dict(
            torch.load(predictor_path, map_location=device, weights_only=True)
        )
        token_weight_predictor.eval()

    image_ids = [item["image_id"] for item in gallery]
    image_chunks = []
    image_patch_chunks = []
    for start in range(0, len(gallery), args.batch_size):
        batch = gallery[start : start + args.batch_size]
        images = []
        for item in batch:
            with Image.open(item["image_path"]) as image:
                images.append(image.convert("RGB"))
        with torch.inference_mode():
            if backend == "official":
                pixel_values = torch.stack(
                    [official_preprocess(image) for image in images]
                ).to(device)
                features = model.encode_image(pixel_values)
                if args.local_score_weight > 0:
                    image_patch_chunks.append(
                        torch.nn.functional.normalize(
                            local_heads[0](official_cn_clip_image_patches(model, pixel_values).float()), dim=-1
                        ).cpu()
                    )
            else:
                inputs = processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
            if backend != "official" and args.local_score_weight > 0:
                vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
                features = model.visual_projection(vision_outputs.pooler_output)
                image_patch_chunks.append(
                    project_image_patches(
                        model,
                        vision_outputs.last_hidden_state,
                        patch_pool=args.local_patch_pool,
                    ).cpu()
                )
            elif backend != "official":
                features = model.get_image_features(pixel_values=pixel_values)
        image_chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
        print(f"images={min(start + args.batch_size, len(gallery))}/{len(gallery)}", flush=True)
    image_features = torch.cat(image_chunks)
    image_patch_features = (
        torch.cat(image_patch_chunks).to(device) if image_patch_chunks else None
    )

    gallery_ids_by_group: dict[str, set[str]] = {}
    for item in gallery:
        gallery_ids_by_group.setdefault(positive_group_id(item), set()).add(
            item["image_id"]
        )
    query_ids_by_group: dict[str, set[str]] = {}
    for item in queries:
        query_ids_by_group.setdefault(positive_group_id(item), set()).add(
            item["pair_id"]
        )

    results, ranks, text_chunks = [], [], []
    text_rankings: list[list[str]] = []
    text_gold: list[set[str]] = []
    local_score_chunks = []
    special_token_ids = (
        tuple(processor.tokenizer.all_special_ids) if processor is not None else ()
    )
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start : start + args.batch_size]
        with torch.inference_mode():
            if backend == "official":
                text_inputs = cn_clip.tokenize(
                    [item["query"] for item in batch],
                    context_length=52,
                ).to(device)
                features = model.encode_text(text_inputs)
                if args.local_score_weight > 0:
                    raw_tokens, raw_mask = official_cn_clip_text_tokens(model, text_inputs)
                    raw_tokens, raw_mask = compact_text_tokens(raw_tokens, raw_mask, args.local_max_tokens)
                    local_text_tokens = local_heads[1](raw_tokens.float())
                    local_text_tokens = torch.nn.functional.normalize(local_text_tokens, dim=-1)
                    local_token_masks = raw_mask
            else:
                inputs = processor(
                    text=[item["query"] for item in batch],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                text_inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                # transformers 4.57 expects a pooler output in get_text_features,
                # while this checkpoint has no pooler. Match the model forward path.
                text_outputs = model.text_model(**text_inputs, return_dict=True)
                features = model.text_projection(text_outputs.last_hidden_state[:, 0, :])
        normalized_features = torch.nn.functional.normalize(features, dim=-1).cpu()
        text_chunks.append(normalized_features)
        global_scores = normalized_features @ image_features.T
        scores = global_scores
        if args.local_score_weight > 0:
            with torch.inference_mode():
                if backend == "official":
                    token_mask = local_token_masks
                    text_tokens = local_text_tokens
                    local_scores = late_interaction_scores(
                        text_tokens, image_patch_features, token_mask,
                        query_chunk_size=args.local_query_chunk_size,
                    ).cpu()
                else:
                    token_mask = valid_text_token_mask(
                        text_inputs["input_ids"], text_inputs["attention_mask"], special_token_ids
                    )
                    text_tokens = project_text_tokens(
                        model, text_outputs.last_hidden_state
                    )
                    text_tokens, token_mask = compact_text_tokens(
                        text_tokens, token_mask, args.local_max_tokens
                    )
                    token_weights = None
                    if token_weight_predictor is not None:
                        full_mask = valid_text_token_mask(
                            text_inputs["input_ids"], text_inputs["attention_mask"], special_token_ids
                        )
                        token_weights = predict_token_weights(
                            token_weight_predictor, text_outputs.last_hidden_state, full_mask,
                            temperature=args.token_weight_temperature,
                        )
                        _, token_weights = compact_text_tokens(
                            token_weights.unsqueeze(-1), full_mask, args.local_max_tokens
                        )
                        token_weights = token_weights.squeeze(-1)
                    local_scores = late_interaction_scores(
                        text_tokens, image_patch_features, token_mask,
                        text_token_weights=token_weights,
                        query_chunk_size=args.local_query_chunk_size,
                    )
                    local_scores = local_scores.cpu()
            local_score_chunks.append(local_scores)
            scores = combine_global_local_scores(
                global_scores, local_scores, args.local_score_weight
            )
        for item, row in zip(batch, scores):
            order = row.argsort(descending=True).tolist()
            ranked_ids = [image_ids[index] for index in order]
            gold_image_ids = gallery_ids_by_group[positive_group_id(item)]
            gold_ranks = [
                rank
                for rank, image_id in enumerate(ranked_ids, 1)
                if image_id in gold_image_ids
            ]
            gold_rank = gold_ranks[0]
            ranks.append(gold_rank)
            text_rankings.append(ranked_ids)
            text_gold.append(gold_image_ids)
            results.append(
                {
                    "pair_id": item["pair_id"],
                    "query": item["query"],
                    "positive_group_id": positive_group_id(item),
                    "gold_image_id": item["image_id"],
                    "gold_image_ids": sorted(gold_image_ids),
                    "gold_rank": gold_rank,
                    "gold_ranks": gold_ranks,
                    "ranked_image_ids": ranked_ids,
                    "scores": [round(row[index].item(), 8) for index in order],
                }
            )
        print(f"queries={min(start + args.batch_size, len(queries))}/{len(queries)}", flush=True)

    text_to_image_metrics = retrieval_metrics(text_rankings, text_gold)
    text_to_image_metrics = {
        key: value
        for key, value in text_to_image_metrics.items()
        if not key.startswith("map@")
    }
    text_to_image_metrics["median_rank"] = statistics.median(ranks)
    text_features = torch.cat(text_chunks)
    all_local_scores = torch.cat(local_score_chunks) if local_score_chunks else None
    query_index_by_image = {item["image_id"]: index for index, item in enumerate(queries)}
    image_to_text_results, image_to_text_ranks = [], []
    image_to_text_rankings: list[list[str]] = []
    image_to_text_gold: list[set[str]] = []
    if len(query_index_by_image) == len(queries) and set(image_ids) == set(query_index_by_image):
        global_image_to_text_scores = image_features @ text_features.T
        image_to_text_scores = (
            combine_global_local_scores(
                global_image_to_text_scores,
                all_local_scores.T,
                args.local_score_weight,
            )
            if all_local_scores is not None
            else global_image_to_text_scores
        )
        query_pair_ids = [item["pair_id"] for item in queries]
        for gallery_item, row in zip(gallery, image_to_text_scores):
            image_id = gallery_item["image_id"]
            order = row.argsort(descending=True).tolist()
            ranked_pair_ids = [query_pair_ids[index] for index in order]
            gold_index = query_index_by_image[image_id]
            gold_pair_ids = query_ids_by_group[positive_group_id(gallery_item)]
            gold_ranks = [
                rank
                for rank, pair_id in enumerate(ranked_pair_ids, 1)
                if pair_id in gold_pair_ids
            ]
            gold_rank = gold_ranks[0]
            image_to_text_ranks.append(gold_rank)
            image_to_text_rankings.append(ranked_pair_ids)
            image_to_text_gold.append(gold_pair_ids)
            image_to_text_results.append(
                {
                    "image_id": image_id,
                    "positive_group_id": positive_group_id(gallery_item),
                    "gold_pair_id": query_pair_ids[gold_index],
                    "gold_pair_ids": sorted(gold_pair_ids),
                    "gold_rank": gold_rank,
                    "gold_ranks": gold_ranks,
                    "ranked_pair_ids": ranked_pair_ids,
                    "scores": [round(row[index].item(), 8) for index in order],
                }
            )
    image_to_text_metrics = (
        retrieval_metrics(image_to_text_rankings, image_to_text_gold)
        if image_to_text_ranks
        else None
    )
    if image_to_text_metrics:
        image_to_text_metrics = {
            key: value
            for key, value in image_to_text_metrics.items()
            if not key.startswith("map@")
        }
        image_to_text_metrics["median_rank"] = statistics.median(image_to_text_ranks)
    metrics = {"text_to_image": text_to_image_metrics, "image_to_text": image_to_text_metrics}
    if image_to_text_metrics:
        metrics["MR"] = sum(
            text_to_image_metrics[f"recall@{k}"] + image_to_text_metrics[f"recall@{k}"]
            for k in (1, 5, 10)
        ) / 6.0
        metrics["mean_recall"] = metrics["MR"]
        metrics["mrr"] = (text_to_image_metrics["mrr"] + image_to_text_metrics["mrr"]) / 2
        text_multi = multi_positive_metrics(text_rankings, text_gold)
        image_multi = multi_positive_metrics(image_to_text_rankings, image_to_text_gold)
        if text_multi["queries"] != image_multi["queries"]:
            raise ValueError("Bidirectional multi-positive query counts do not match")
        metrics["multi_positive"] = {
            "map@R": (text_multi["map@R"] + image_multi["map@R"]) / 2.0,
            "R-Precision": (
                text_multi["R-Precision"] + image_multi["R-Precision"]
            ) / 2.0,
            "R@1": (text_multi["R@1"] + image_multi["R@1"]) / 2.0,
            "queries_per_direction": text_multi["queries"],
        }
    if args.hard_negatives:
        hard_path = Path(args.hard_negatives).resolve()
        hard_rows = json.loads(hard_path.read_text())
        metrics["hard_negative_subset"] = hard_negative_metrics(
            results, image_to_text_results, hard_rows, all_pairs
        )
    ranks_by_cave: dict[str, list[int]] = {}
    for item, rank in zip(queries, ranks):
        ranks_by_cave.setdefault(item["cave_id"], []).append(rank)
    metrics_by_cave = {
        cave_id: {
            "num_queries": len(cave_ranks),
            "recall@1": sum(rank <= 1 for rank in cave_ranks) / len(cave_ranks),
            "recall@5": sum(rank <= 5 for rank in cave_ranks) / len(cave_ranks),
            "recall@10": sum(rank <= 10 for rank in cave_ranks) / len(cave_ranks),
            "mrr": sum(1.0 / rank for rank in cave_ranks) / len(cave_ranks),
            "mean_rank": sum(cave_ranks) / len(cave_ranks),
            "median_rank": statistics.median(cave_ranks),
        }
        for cave_id, cave_ranks in sorted(ranks_by_cave.items())
    }
    config = {
        "experiment": args.experiment,
        "model": model_path.name,
        "model_path": str(model_path),
        "model_revision": args.model_revision,
        "model_weights_sha256": file_sha256(weights_path),
        "backend": backend,
        "adapter": (
            {
                "path": str(Path(args.adapter).resolve()),
                "config": adapter_config,
                "weights_sha256": file_sha256(
                    Path(args.adapter).resolve() / "adapter_model.pt"
                ),
            }
            if args.adapter
            else None
        ),
        "test_file": str(pairs_path),
        "test_sha256": file_sha256(pairs_path),
        "query_subset": {"query_source": args.query_source},
        "num_queries": len(queries),
        "gallery_size": len(gallery),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": device,
        "gpu": gpu_name,
        "precision": str(next(model.parameters()).dtype).replace("torch.", ""),
        "image_processor": (
            "official_cn_clip" if backend == "official" else "slow (checkpoint-compatible)"
        ),
        "multi_positive_evaluation": {
            "group_key": "positive_group_id (SHA-256 of exact description)",
            "positive_groups": len(gallery_ids_by_group),
            "multi_positive_groups": sum(
                len(ids) > 1 for ids in gallery_ids_by_group.values()
            ),
            "records_in_multi_positive_groups": sum(
                len(ids) for ids in gallery_ids_by_group.values() if len(ids) > 1
            ),
            "reported_metrics": (
                "bidirectional mean mAP@R, R-Precision, and R@1 on "
                "multi-positive queries only"
            ),
            "R_definition": "number of relevant items for each query",
            "map_at_R_denominator": "R",
        },
        "local_alignment": {
            "enabled": args.local_score_weight > 0,
            "score_weight": args.local_score_weight,
            "max_text_tokens": args.local_max_tokens,
            "patch_pool": args.local_patch_pool,
            "query_chunk_size": args.local_query_chunk_size,
            "interaction": "symmetric_token_patch_maxsim",
            "token_weighting": {
                "enabled": args.token_weighting,
                "hidden_size": args.token_weight_hidden,
                "temperature": args.token_weight_temperature,
                "normalization": "masked_mean_one",
            },
        },
        "seed": None,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "pillow": pillow_version,
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "command": " ".join(sys.argv),
    }
    write_jsonl(output_dir / "rankings_clean.jsonl", results)
    if image_to_text_results:
        write_jsonl(output_dir / "rankings_image_to_text.jsonl", image_to_text_results)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics_clean.json").write_text(
        json.dumps(
            {**config, "metrics": metrics, "metrics_by_cave": metrics_by_cave},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    recall_at_k = {
        "experiment": args.experiment,
        "model": model_path.name,
        "test_file": str(pairs_path),
        "num_queries": len(queries),
        "gallery_size": len(gallery),
        "device": device,
        "gpu": gpu_name,
        "text_to_image": {
            f"recall@{k}": text_to_image_metrics[f"recall@{k}"]
            for k in (1, 5, 10)
        },
        "image_to_text": {
            f"recall@{k}": image_to_text_metrics[f"recall@{k}"]
            for k in (1, 5, 10)
        } if image_to_text_metrics else None,
        "MR": metrics.get("MR"),
        "multi_positive": metrics.get("multi_positive"),
    }
    (output_dir / "recall_at_k.json").write_text(
        json.dumps(recall_at_k, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot Chinese-CLIP retrieval")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Optional compatible LoRA adapter directory")
    parser.add_argument("--local-heads", help="Path to official CN-CLIP local_heads.pt")
    parser.add_argument("--experiment", default="chinese_clip_retrieval_test")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument(
        "--backend",
        choices=("auto", "official", "huggingface"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--query-source", default="visual_text")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-gallery", type=int)
    parser.add_argument("--local-score-weight", type=float, default=0.0)
    parser.add_argument("--local-max-tokens", type=int, default=32)
    parser.add_argument("--local-patch-pool", type=int, default=2)
    parser.add_argument("--local-query-chunk-size", type=int, default=4)
    parser.add_argument("--token-weighting", action="store_true")
    parser.add_argument("--token-weight-hidden", type=int, default=192)
    parser.add_argument("--token-weight-temperature", type=float, default=1.0)
    parser.add_argument(
        "--hard-negatives",
        help="Optional reviewed hard-negative JSON for subset R@1 and HN Accuracy",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if args.local_score_weight < 0:
        raise SystemExit("--local-score-weight cannot be negative")
    if args.local_max_tokens < 1:
        raise SystemExit("--local-max-tokens must be positive")
    if args.local_patch_pool < 1:
        raise SystemExit("--local-patch-pool must be positive")
    if args.local_query_chunk_size < 1:
        raise SystemExit("--local-query-chunk-size must be positive")
    if args.token_weight_hidden < 1:
        raise SystemExit("--token-weight-hidden must be positive")
    if args.token_weight_temperature <= 0:
        raise SystemExit("--token-weight-temperature must be positive")

    pairs_path = Path(args.pairs).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("w", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(Tee(log_handle)):
            evaluate(args, pairs_path, model_path, output_dir)


if __name__ == "__main__":
    main()
