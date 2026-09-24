#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mmmrag.io import read_jsonl
from mmmrag.metrics import evidence_metrics, exact_match, token_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions", required=True)
    args = parser.parse_args()

    questions = {item["question_id"]: item for item in read_jsonl(args.questions)}
    predictions = read_jsonl(args.predictions)
    gold = [questions[item["question_id"]] for item in predictions]
    result = evidence_metrics(
        [set(item["evidence_ids"]) for item in predictions],
        [set(item["gold_evidence_ids"]) for item in gold],
    )
    result["answer_em"] = sum(exact_match(item["prediction"], target["answer"]) for item, target in zip(predictions, gold)) / max(len(gold), 1)
    result["answer_f1"] = sum(token_f1(item["prediction"], target["answer"]) for item, target in zip(predictions, gold)) / max(len(gold), 1)
    result["sufficiency_rate"] = sum(item["sufficient"] for item in predictions) / max(len(predictions), 1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

