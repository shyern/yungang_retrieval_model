#!/usr/bin/env python3
"""LoRA fine-tuning for local QA-CLIP and R2D2 checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

from mmmrag.io import read_jsonl
from mmmrag.lora import LoRALinear


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("qa_clip", "r2d2"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--repo")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.)
    p.add_argument("--dropout", type=float, default=.05)
    p.add_argument("--vision-layers", type=int, default=4)
    p.add_argument("--text-layers", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_r2d2_aliases() -> None:
    import transformers.modeling_utils as modeling_utils
    from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    modeling_utils.prune_linear_layer = prune_linear_layer


def load_backend(a: argparse.Namespace, device: Any) -> tuple[Any, Any]:
    if a.backend == "qa_clip":
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        processor = ChineseCLIPProcessor.from_pretrained(a.model, local_files_only=True, use_fast=False)
        model = ChineseCLIPModel.from_pretrained(a.model, local_files_only=True).to(device)
        return model, processor
    if not a.repo:
        raise SystemExit("--repo is required for R2D2")
    import torch
    install_r2d2_aliases()
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    old = Path.cwd()
    try:
        os.chdir(repo)
        from models.r2d2 import R2D2
        model = R2D2(vit_type="large", image_size=224, embed_dim=768)
    finally:
        os.chdir(old)
    checkpoint = torch.load(a.model, map_location="cpu", weights_only=False)["model"]
    current = model.state_dict()
    checkpoint = {k: v for k, v in checkpoint.items() if k in current and v.shape == current[k].shape}
    model.load_state_dict(checkpoint, strict=False)
    return model.to(device), None


def inject_backend_lora(model: Any, a: argparse.Namespace) -> list[str]:
    import torch.nn as nn
    for parameter in model.parameters():
        parameter.requires_grad = False
    if a.backend == "qa_clip":
        v_count = len(model.vision_model.encoder.layers)
        t_count = len(model.text_model.encoder.layer)
        v_prefix = "vision_model.encoder.layers."
        t_prefix = "text_model.encoder.layer."
        def target(name: str) -> bool:
            return (
                any(name == f"{v_prefix}{i}.self_attn.{p}" for i in range(v_count-a.vision_layers, v_count) for p in ("q_proj", "v_proj"))
                or any(name == f"{t_prefix}{i}.attention.self.{p}" for i in range(t_count-a.text_layers, t_count) for p in ("query", "value"))
            )
    else:
        v_count = len(model.visual_encoder.encoder.layers)
        t_count = len(model.text_encoder.encoder.layer)
        def target(name: str) -> bool:
            return (
                any(name == f"visual_encoder.encoder.layers.{i}.self_attn.{p}" for i in range(v_count-a.vision_layers, v_count) for p in ("q_proj", "v_proj"))
                or any(name == f"text_encoder.encoder.layer.{i}.attention.self.{p}" for i in range(t_count-a.text_layers, t_count) for p in ("query", "value"))
            )
    selected = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and target(name):
            parent_name, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attribute, LoRALinear.build(module, a.rank, a.alpha, a.dropout))
            selected.append(name)
    expected = 2 * (a.vision_layers + a.text_layers)
    if len(selected) != expected:
        raise RuntimeError(f"Injected {len(selected)} LoRA modules, expected {expected}: {selected}")
    return selected


def load_external_lora(model: Any, checkpoint_dir: Path, backend: str) -> dict[str, Any]:
    """Inject and load an adapter produced by this training script."""
    import torch
    from types import SimpleNamespace
    config = json.loads((checkpoint_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config["backend"] != backend:
        raise ValueError(f"Adapter backend {config['backend']} does not match {backend}")
    inject_backend_lora(model, SimpleNamespace(**config))
    state = torch.load(checkpoint_dir / "adapter_model.pt", map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"Adapter parameters missing from model: {missing[:3]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
    return config


def preprocess_r2d2(path: str) -> Any:
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)),
    ])
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def mean_recall(image_features: Any, text_features: Any, groups: list[str]) -> float:
    import torch
    scores = text_features @ image_features.T
    mask = torch.tensor([[a == b for b in groups] for a in groups], dtype=torch.bool)
    values = []
    for matrix, positives in ((scores, mask), (scores.T, mask.T)):
        order = matrix.argsort(dim=1, descending=True)
        ranked_positive = positives.gather(1, order)
        ranks = ranked_positive.float().argmax(dim=1) + 1
        values.extend((ranks <= k).float().mean().item() for k in (1, 5, 10))
    return sum(values) / 6


def main() -> None:
    a = parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    set_seed(a.seed)
    device = torch.device("cuda")
    out = Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    train, validation = read_jsonl(Path(a.train)), read_jsonl(Path(a.validation))
    model, processor = load_backend(a, device)
    selected = inject_backend_lora(model, a)

    class DS(Dataset):
        def __init__(self, rows: list[dict[str, Any]]): self.rows = rows
        def __len__(self) -> int: return len(self.rows)
        def __getitem__(self, i: int) -> tuple[Any, str, str]:
            row = self.rows[i]
            if a.backend == "r2d2": pixels = preprocess_r2d2(row["image_path"])
            else:
                with Image.open(row["image_path"]) as image: pixels = image.convert("RGB")
            return pixels, row["query"], row["positive_group_id"]

    def collate(items: list[tuple[Any, str, str]]) -> dict[str, Any]:
        images, texts, groups = zip(*items)
        if a.backend == "qa_clip":
            batch = processor(text=list(texts), images=list(images), padding=True, truncation=True, max_length=256, return_tensors="pt")
        else:
            batch = {"pixel_values": torch.stack(images), "texts": list(texts)}
        batch["groups"] = list(groups)
        return batch

    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(DS(train), batch_size=a.batch_size, shuffle=True, generator=generator, num_workers=a.num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(DS(validation), batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True, collate_fn=collate)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=a.learning_rate, weight_decay=a.weight_decay)

    def encode(batch: dict[str, Any]) -> tuple[Any, Any]:
        if a.backend == "qa_clip":
            values = {k: v.to(device) for k, v in batch.items() if k != "groups"}
            vision = model.vision_model(pixel_values=values["pixel_values"], return_dict=True)
            image_features = model.visual_projection(vision.pooler_output)
            text_values = {k: v for k, v in values.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            text = model.text_model(**text_values, return_dict=True)
            text_features = model.text_projection(text.last_hidden_state[:, 0, :])
            return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1)
        pixels = batch["pixel_values"].to(device)
        tokens = model.tokenize_text(batch["texts"]).to(device)
        return model.encode_image(pixels)[1], model.encode_text(tokens)[1]

    @torch.no_grad()
    def validate() -> float:
        model.eval(); images=[]; texts=[]; groups=[]
        for batch in val_loader:
            im, tx = encode(batch); images.append(im.cpu()); texts.append(tx.cpu()); groups.extend(batch["groups"])
        return mean_recall(torch.cat(images), torch.cat(texts), groups)

    config = vars(a) | {"selected_modules": selected, "selection_metric": "validation_MR", "loss": "bidirectional_CLIP"}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    best = -math.inf; stale = 0; history=[]
    for epoch in range(1, a.epochs + 1):
        model.train(); total=0.; seen=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                im, tx = encode(batch); logits = tx @ im.T / .07
                target = torch.arange(len(im), device=device)
                loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
            total += loss.item() * len(im); seen += len(im)
        mr = validate(); record={"epoch":epoch,"train_loss":total/seen,"validation_MR":mr}; history.append(record); print(json.dumps(record), flush=True)
        if mr > best + 1e-4:
            best=mr; stale=0
            state={n:p.detach().cpu() for n,p in model.named_parameters() if n.endswith("lora_A") or n.endswith("lora_B")}
            checkpoint=out/"checkpoints"/"best"; checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint/"adapter_model.pt")
            (checkpoint/"adapter_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
        else: stale += 1
        if stale >= a.patience: break
    (out/"metrics.jsonl").write_text("".join(json.dumps(x)+"\n" for x in history))
    (out/"summary.json").write_text(json.dumps({"best_validation_MR":best,"best_epoch":max(history,key=lambda x:x["validation_MR"])["epoch"]},indent=2)+"\n")


if __name__ == "__main__": main()
