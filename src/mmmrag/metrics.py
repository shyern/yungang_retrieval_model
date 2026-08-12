from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def retrieval_metrics(
    rankings: Iterable[list[str]], gold: Iterable[set[str]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    ranking_list = list(rankings)
    gold_list = list(gold)
    if len(ranking_list) != len(gold_list):
        raise ValueError("rankings and gold must have equal length")
    if not ranking_list:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "mean_rank": 0.0}

    ranks = []
    for ranked_ids, relevant_ids in zip(ranking_list, gold_list):
        rank = next((i for i, item in enumerate(ranked_ids, 1) if item in relevant_ids), len(ranked_ids) + 1)
        ranks.append(rank)

    result = {f"recall@{k}": sum(rank <= k for rank in ranks) / len(ranks) for k in ks}
    result["mrr"] = sum(1.0 / rank for rank in ranks) / len(ranks)
    result["mean_rank"] = sum(ranks) / len(ranks)
    return result


def evidence_metrics(predicted: list[set[str]], gold: list[set[str]]) -> dict[str, float]:
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have equal length")
    if not gold:
        return {"evidence_recall": 0.0, "evidence_precision": 0.0, "complete_set_recall": 0.0}
    recalls, precisions, complete = [], [], []
    for pred, target in zip(predicted, gold):
        recalls.append(len(pred & target) / len(target) if target else 1.0)
        precisions.append(len(pred & target) / len(pred) if pred else 0.0)
        complete.append(target <= pred)
    return {
        "evidence_recall": sum(recalls) / len(recalls),
        "evidence_precision": sum(precisions) / len(precisions),
        "complete_set_recall": sum(complete) / len(complete),
    }


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = list(_normalize(prediction))
    answer_tokens = list(_normalize(answer))
    if not pred_tokens or not answer_tokens:
        return float(pred_tokens == answer_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(answer_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, answer: str) -> float:
    return float(_normalize(prediction) == _normalize(answer))


def _normalize(text: str) -> str:
    punctuation = "，。！？；：,.!?;:、 \t\n"
    return "".join(char.lower() for char in text if char not in punctuation)
