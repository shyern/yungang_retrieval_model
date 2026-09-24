"""Small LoRA adapter utilities for the official Chinese-CLIP implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LoRALinear:
    """Factory namespace used to avoid importing torch at module import time."""

    @staticmethod
    def build(base: Any, rank: int, alpha: float, dropout: float) -> Any:
        import math

        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        class AdapterLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                self.rank = rank
                self.alpha = alpha
                self.scaling = alpha / rank
                self.dropout = nn.Dropout(dropout)
                device = base.weight.device
                self.lora_A = nn.Parameter(
                    torch.empty(rank, base.in_features, device=device, dtype=torch.float32)
                )
                self.lora_B = nn.Parameter(
                    torch.zeros(base.out_features, rank, device=device, dtype=torch.float32)
                )
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            def forward(self, inputs: Any) -> Any:
                base_output = self.base(inputs)
                dropped = self.dropout(inputs.to(self.lora_A.dtype))
                update = functional.linear(
                    functional.linear(dropped, self.lora_A), self.lora_B
                )
                return base_output + (update * self.scaling).to(base_output.dtype)

        return AdapterLinear()


def inject_lora(
    model: Any,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    vision_layers: int = 4,
    text_layers: int = 4,
) -> list[str]:
    """Freeze a Chinese-CLIP model and inject adapters into its last layers."""
    import torch.nn as nn

    if rank < 1 or vision_layers < 1 or text_layers < 1:
        raise ValueError("rank and selected layer counts must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = False

    vision_count = len(model.visual.transformer.resblocks)
    text_count = len(model.bert.encoder.layer)
    vision_start = max(0, vision_count - vision_layers)
    text_start = max(0, text_count - text_layers)
    selected = []
    candidates = list(model.named_modules())
    for name, module in candidates:
        if not isinstance(module, nn.Linear):
            continue
        vision_target = any(
            name == f"visual.transformer.resblocks.{index}.mlp.{projection}"
            for index in range(vision_start, vision_count)
            for projection in ("c_fc", "c_proj")
        )
        text_target = any(
            name == f"bert.encoder.layer.{index}.attention.self.{projection}"
            for index in range(text_start, text_count)
            for projection in ("query", "value")
        )
        if not (vision_target or text_target):
            continue
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, LoRALinear.build(module, rank, alpha, dropout))
        selected.append(name)
    expected = vision_layers * 2 + text_layers * 2
    if len(selected) != expected:
        raise RuntimeError(f"Injected {len(selected)} adapters; expected {expected}")
    return selected


def adapter_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }


def save_lora_adapter(model: Any, path: Path, config: dict[str, Any]) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state_dict(model), path / "adapter_model.pt")
    (path / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_lora_adapter(model: Any, path: Path) -> dict[str, Any]:
    import torch

    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    inject_lora(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config.get("dropout", 0.0),
        vision_layers=config["vision_layers"],
        text_layers=config["text_layers"],
    )
    state = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"Adapter parameters missing from model: {missing[:3]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
    return config
