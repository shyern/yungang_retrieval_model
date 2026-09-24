"""Token-patch late interaction utilities for Chinese-CLIP retrieval."""

from __future__ import annotations

from typing import Any, Iterable


def official_cn_clip_local_features(model: Any, images: Any, tokens: Any) -> tuple[Any, Any, Any]:
    """Extract normalized patch and token features from an official cn_clip model."""
    import torch.nn.functional as functional
    import torch

    b = images.shape[0]
    visual = model.visual
    x = visual.conv1(images.type(model.dtype)).reshape(b, visual.conv1.out_channels, -1).permute(0, 2, 1)
    x = torch.cat([visual.class_embedding.to(x.dtype).expand(b, 1, -1), x], 1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)
    vision_states = visual.transformer(x).permute(1, 0, 2)
    text_states = model.bert(tokens, attention_mask=tokens.ne(model.tokenizer.vocab["[PAD]"]).type(model.dtype))[0]
    # The official full-method trainer projects raw 1280-D transformer patch
    # states with its learned local head; do not apply the global 1024-D proj.
    patches = vision_states[:, 1:, :]
    text = text_states
    mask = tokens.ne(model.tokenizer.vocab["[PAD]"])
    for key in ("[CLS]", "[SEP]"):
        mask &= tokens.ne(model.tokenizer.vocab[key])
    return patches, text, mask


def official_cn_clip_image_patches(model: Any, images: Any) -> Any:
    """Return normalized official cn_clip ViT patch features."""
    import torch
    import torch.nn.functional as functional
    visual = model.visual
    b = images.shape[0]
    x = visual.conv1(images.type(model.dtype)).reshape(b, visual.conv1.out_channels, -1).permute(0, 2, 1)
    x = torch.cat([visual.class_embedding.to(x.dtype).expand(b, 1, -1), x], 1)
    x = (x + visual.positional_embedding.to(x.dtype)).permute(1, 0, 2)
    states = visual.transformer(visual.ln_pre(x)).permute(1, 0, 2)
    return states[:, 1:, :]


def official_cn_clip_text_tokens(model: Any, tokens: Any) -> tuple[Any, Any]:
    """Return normalized official cn_clip token features and content mask."""
    import torch.nn.functional as functional
    pad = model.tokenizer.vocab["[PAD]"]
    states = model.bert(tokens, attention_mask=tokens.ne(pad).type(model.dtype))[0]
    features = states
    mask = tokens.ne(pad)
    for key in ("[CLS]", "[SEP]"):
        mask &= tokens.ne(model.tokenizer.vocab[key])
    return features, mask


def build_token_weight_predictor(hidden_size: int, bottleneck: int = 192) -> Any:
    """Build a small token scorer without changing the CLIP checkpoint schema."""
    import torch.nn as nn

    if hidden_size < 1 or bottleneck < 1:
        raise ValueError("hidden_size and bottleneck must be positive")
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, bottleneck),
        nn.GELU(),
        nn.Linear(bottleneck, 1),
    )


def predict_token_weights(
    predictor: Any,
    text_hidden_states: Any,
    text_token_mask: Any,
    temperature: float = 1.0,
) -> Any:
    """Predict positive, mask-aware token weights with mean weight one."""
    import torch
    import torch.nn.functional as functional

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = predictor(text_hidden_states).squeeze(-1) / temperature
    weights = functional.softplus(logits) + 1e-6
    mask = text_token_mask.to(weights.dtype)
    weights = weights * mask
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights * mask.sum(dim=1, keepdim=True).clamp_min(1.0)


def valid_text_token_mask(
    input_ids: Any,
    attention_mask: Any,
    special_token_ids: Iterable[int],
) -> Any:
    """Return tokens that carry content, excluding padding and special tokens."""
    mask = attention_mask.bool().clone()
    for token_id in special_token_ids:
        mask &= input_ids.ne(token_id)
    return mask


def project_local_features(
    model: Any,
    text_hidden_states: Any,
    vision_hidden_states: Any,
    patch_pool: int = 1,
) -> tuple[Any, Any]:
    """Project text tokens and image patches into CLIP's shared embedding space."""
    return (
        project_text_tokens(model, text_hidden_states),
        project_image_patches(model, vision_hidden_states, patch_pool),
    )


def project_text_tokens(model: Any, text_hidden_states: Any) -> Any:
    """Project and normalize every text token with CLIP's text projection."""
    import torch.nn.functional as functional

    return functional.normalize(model.text_projection(text_hidden_states), dim=-1)


def project_image_patches(
    model: Any,
    vision_hidden_states: Any,
    patch_pool: int = 1,
) -> Any:
    """Pool, project, and normalize ViT patch tokens."""
    import math

    import torch.nn.functional as functional

    # CLIP applies this layer norm to its global vision token. Reusing it for
    # patches keeps global and local features on the same normalized manifold.
    patch_hidden_states = model.vision_model.post_layernorm(
        vision_hidden_states[:, 1:, :]
    )
    if patch_pool < 1:
        raise ValueError("patch_pool must be positive")
    if patch_pool > 1:
        patch_count = patch_hidden_states.shape[1]
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count or grid_size % patch_pool:
            raise ValueError(
                f"Cannot pool {patch_count} patches by a factor of {patch_pool}"
            )
        patch_grid = patch_hidden_states.reshape(
            patch_hidden_states.shape[0], grid_size, grid_size, -1
        ).permute(0, 3, 1, 2)
        patch_grid = functional.avg_pool2d(
            patch_grid, kernel_size=patch_pool, stride=patch_pool
        )
        patch_hidden_states = patch_grid.flatten(2).transpose(1, 2)
    image_patches = model.visual_projection(patch_hidden_states)
    return functional.normalize(image_patches, dim=-1)


def compact_text_tokens(
    text_tokens: Any,
    text_token_mask: Any,
    max_tokens: int,
) -> tuple[Any, Any]:
    """Keep evenly spaced content tokens so long descriptions retain their tail."""
    import torch

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    compact = text_tokens.new_zeros(
        (text_tokens.shape[0], max_tokens, text_tokens.shape[-1])
    )
    compact_mask = torch.zeros(
        (text_tokens.shape[0], max_tokens),
        dtype=torch.bool,
        device=text_tokens.device,
    )
    for batch_index in range(text_tokens.shape[0]):
        indices = text_token_mask[batch_index].nonzero(as_tuple=False).flatten()
        if indices.numel() > max_tokens:
            positions = torch.linspace(
                0,
                indices.numel() - 1,
                steps=max_tokens,
                device=indices.device,
            ).round().long()
            indices = indices[positions]
        count = indices.numel()
        compact[batch_index, :count] = text_tokens[batch_index, indices]
        compact_mask[batch_index, :count] = True
    return compact, compact_mask


def late_interaction_scores(
    text_tokens: Any,
    image_patches: Any,
    text_token_mask: Any,
    text_token_weights: Any | None = None,
    query_chunk_size: int = 4,
) -> Any:
    """Compute symmetric MaxSim scores for every text-image combination.

    Text-to-image scores average each content token's best matching patch.
    Image-to-text scores average each patch's best matching content token.
    Chunking bounds the temporary ``query x image x token x patch`` tensor.
    """
    import torch

    if text_tokens.ndim != 3 or image_patches.ndim != 3:
        raise ValueError("Expected text tokens and image patches to be rank-3 tensors")
    if text_tokens.shape[0] != text_token_mask.shape[0]:
        raise ValueError("Text token mask batch dimension does not match text tokens")
    if text_tokens.shape[1] != text_token_mask.shape[1]:
        raise ValueError("Text token mask length does not match text tokens")
    if text_token_weights is not None and text_token_weights.shape != text_token_mask.shape:
        raise ValueError("Text token weights must have the same shape as the token mask")
    if not text_token_mask.any(dim=1).all():
        raise ValueError("Every query must contain at least one non-special text token")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")

    chunks = []
    floor = torch.finfo(text_tokens.dtype).min
    for start in range(0, text_tokens.shape[0], query_chunk_size):
        query_tokens = text_tokens[start : start + query_chunk_size]
        query_mask = text_token_mask[start : start + query_chunk_size]
        query_weights = (
            query_mask.to(query_tokens.dtype)
            if text_token_weights is None
            else text_token_weights[start : start + query_chunk_size]
        )
        similarities = torch.einsum("qld,ipd->qilp", query_tokens, image_patches)

        token_scores = similarities.amax(dim=-1)
        token_weights = query_weights[:, None, :].to(token_scores.dtype)
        text_to_image = (token_scores * token_weights).sum(dim=-1)
        text_to_image /= token_weights.sum(dim=-1).clamp_min(1.0)

        masked_similarities = similarities.masked_fill(
            ~query_mask[:, None, :, None], floor
        )
        weighted_similarities = masked_similarities * token_weights[:, :, :, None]
        image_to_text = weighted_similarities.amax(dim=-2).sum(dim=-1)
        image_to_text /= token_weights.sum(dim=-1).clamp_min(1.0)
        chunks.append((text_to_image + image_to_text) * 0.5)
    return torch.cat(chunks, dim=0)


def symmetric_contrastive_loss(logits_per_text: Any) -> Any:
    """Apply the standard bidirectional in-batch retrieval objective."""
    import torch
    import torch.nn.functional as functional

    if logits_per_text.shape[0] != logits_per_text.shape[1]:
        raise ValueError("Contrastive training requires equal text and image batch sizes")
    targets = torch.arange(logits_per_text.shape[0], device=logits_per_text.device)
    return (
        functional.cross_entropy(logits_per_text, targets)
        + functional.cross_entropy(logits_per_text.t(), targets)
    ) * 0.5


def combine_global_local_scores(global_scores: Any, local_scores: Any, weight: float) -> Any:
    """Blend equally scaled cosine scores while keeping their numeric range stable."""
    if weight < 0:
        raise ValueError("Local score weight cannot be negative")
    if weight == 0:
        return global_scores
    return (global_scores + weight * local_scores) / (1.0 + weight)
