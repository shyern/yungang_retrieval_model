"""Losses and sampling helpers for the MP + LC + HN retrieval method."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def multi_positive_contrastive_loss(logits: Any, positive_mask: Any) -> Any:
    """Bidirectional SupCon with equal total weight for each positive group."""
    import torch
    if logits.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must have the same rank-2 shape")
    if not positive_mask.any(dim=1).all() or not positive_mask.any(dim=0).all():
        raise ValueError("every query and gallery item needs at least one positive")
    def direction(values: Any, mask: Any) -> Any:
        log_prob = values - torch.logsumexp(values, dim=1, keepdim=True)
        positive_count = mask.sum(dim=1).clamp_min(1)
        per_query = -(log_prob.masked_fill(~mask, 0.0).sum(dim=1) / positive_count)
        # A group of size R appears as R query records. Weight every record by
        # 1/R so each semantic positive group contributes equal total weight.
        query_weight = positive_count.reciprocal()
        return (per_query * query_weight).sum() / query_weight.sum()
    return (direction(logits, positive_mask) + direction(logits.t(), positive_mask.t())) * .5


def local_contrastive_loss(
    local_scores: Any, positive_mask: Any, temperature: float
) -> Any:
    """Multi-positive local supervised contrastive loss, bidirectionally."""
    if temperature <= 0: raise ValueError("temperature must be positive")
    return multi_positive_contrastive_loss(local_scores / temperature, positive_mask)


def hard_negative_ranking_loss(
    ranking_scores: Any, positive_mask: Any, explicit_t2i: Any, explicit_i2t: Any, margin: float
) -> tuple[Any, dict[str, int]]:
    """Rank every positive above the explicit-first hardest negative."""
    import torch
    import torch.nn.functional as F
    b=ranking_scores.shape[0]
    t_losses=[]; i_losses=[]; explicit_t=explicit_i=0
    for i in range(b):
        t_candidates=explicit_t2i[i].nonzero().flatten()
        if t_candidates.numel(): explicit_t+=1
        else: t_candidates=(~positive_mask[i]).nonzero().flatten()
        i_candidates=explicit_i2t[i].nonzero().flatten()
        if i_candidates.numel(): explicit_i+=1
        else: i_candidates=(~positive_mask[:,i]).nonzero().flatten()
        if t_candidates.numel():
            positives=positive_mask[i].nonzero().flatten()
            negative=ranking_scores[i,t_candidates].max()
            t_losses.append(F.relu(margin-ranking_scores[i,positives]+negative).mean())
        if i_candidates.numel():
            positives=positive_mask[:,i].nonzero().flatten()
            negative=ranking_scores[i_candidates,i].max()
            i_losses.append(F.relu(margin-ranking_scores[positives,i]+negative).mean())
    if not t_losses and not i_losses:
        zero=ranking_scores.sum()*0
        return zero, {"explicit_t2i_queries": explicit_t, "explicit_i2t_queries": explicit_i}
    t_loss=torch.stack(t_losses).mean() if t_losses else ranking_scores.sum()*0
    i_loss=torch.stack(i_losses).mean() if i_losses else ranking_scores.sum()*0
    return (t_loss+i_loss)*.5, {
        "explicit_t2i_queries": explicit_t, "explicit_i2t_queries": explicit_i
    }


def relation_maps(rows: list[dict[str, Any]]) -> tuple[dict[str,set[str]],dict[str,set[str]]]:
    """Map text->negative image keys and negative image key->negative texts."""
    t2i: dict[str,set[str]]=defaultdict(set); i2t: dict[str,set[str]]=defaultdict(set)
    for row in rows:
        text=str(row["description"])
        paths=row["negative_image_paths"]
        if not isinstance(paths,list): paths=[paths]
        for path in paths:
            key=Path(path).name; t2i[text].add(key); i2t[key].add(text)
    return dict(t2i),dict(i2t)


def relation_aware_batches(
    records: list[dict[str, Any]], text_to_images: dict[str,set[str]], batch_size: int, seed: int
) -> list[list[int]]:
    """Pack positive-group peers and annotated negatives together where possible."""
    import random
    if batch_size < 2: raise ValueError("batch_size must be at least two")
    by_group: dict[str,list[int]]=defaultdict(list); by_image: dict[str,int]={}
    for i,r in enumerate(records):
        by_group[str(r["positive_group_id"])].append(i); by_image[Path(r["image_path"]).name]=i
    rng=random.Random(seed); remaining=set(range(len(records))); order=list(remaining); rng.shuffle(order); batches=[]
    for anchor in order:
        if anchor not in remaining: continue
        batch=[]
        def add(index: int) -> None:
            if index in remaining and len(batch)<batch_size:
                batch.append(index); remaining.remove(index)
        add(anchor); row=records[anchor]
        peers=by_group[str(row["positive_group_id"])][:]; rng.shuffle(peers)
        for index in peers: add(index)
        negatives=[by_image[k] for k in text_to_images.get(str(row["query"]),set()) if k in by_image]
        rng.shuffle(negatives)
        for index in negatives: add(index)
        fillers=list(remaining); rng.shuffle(fillers)
        for index in fillers: add(index)
        batches.append(batch)
    if len(batches)>1 and len(batches[-1])==1:
        batches[-2].extend(batches.pop())
    return batches
