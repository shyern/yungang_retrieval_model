#!/usr/bin/env python3
"""Search the local/global score blend on validation, then evaluate test."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--validation-pairs", required=True)
    p.add_argument("--test-pairs")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--local-heads", required=True)
    p.add_argument("--validation-hard-negatives")
    p.add_argument("--test-hard-negatives")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--etas", type=float, nargs="+", default=[0.0, .1, .2, .3, .5, .7, 1.0])
    p.add_argument("--batch-size", type=int, default=12)
    args = p.parse_args()
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    common = ["--model", args.model, "--adapter", args.adapter, "--local-heads", args.local_heads,
              "--backend", "official", "--query-source", "description", "--batch-size", str(args.batch_size)]
    rows = []
    for eta in args.etas:
        out = root / f"validation_eta_{eta:g}"
        cmd = ["python", "scripts/run_visual_retrieval.py", "--pairs", args.validation_pairs,
               "--output-dir", str(out), "--experiment", f"validation_eta_{eta:g}",
               "--local-score-weight", str(eta), *common]
        if args.validation_hard_negatives:
            cmd += ["--hard-negatives", args.validation_hard_negatives]
        subprocess.run(cmd, check=True)
        metrics = json.loads((out / "metrics_clean.json").read_text())["metrics"]
        rows.append({"eta": eta, "MR": metrics.get("MR", 0.0),
                     "multi_positive": metrics.get("multi_positive"),
                     "hard_negative_subset": metrics.get("hard_negative_subset")})
    best = max(rows, key=lambda x: (x["MR"], -x["eta"]))
    (root / "eta_search.json").write_text(json.dumps({"candidates": rows, "best": best}, ensure_ascii=False, indent=2) + "\n")
    result = {"best_eta": best["eta"], "validation": best}
    if args.test_pairs:
        test_out = root / "test_best_eta"
        cmd = ["python", "scripts/run_visual_retrieval.py", "--pairs", args.test_pairs,
               "--output-dir", str(test_out), "--experiment", "test_best_eta",
               "--local-score-weight", str(best["eta"]), *common]
        if args.test_hard_negatives:
            cmd += ["--hard-negatives", args.test_hard_negatives]
        subprocess.run(cmd, check=True)
        result["test_dir"] = str(test_out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
