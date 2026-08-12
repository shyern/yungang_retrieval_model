#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mmmrag.io import read_jsonl, write_jsonl
from mmmrag.rag import EvidenceAwareRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-aware sparse RAG baseline")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=2)
    args = parser.parse_args()

    pipeline = EvidenceAwareRAG(read_jsonl(args.corpus), args.top_k, args.max_iterations)
    predictions = [pipeline.run(sample) for sample in read_jsonl(args.questions)]
    write_jsonl(args.output, predictions)
    sufficient = sum(item["sufficient"] for item in predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}; sufficient={sufficient}/{len(predictions)}")


if __name__ == "__main__":
    main()
