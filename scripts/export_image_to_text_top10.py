"""Export verbatim top-10 image-to-text retrieval results for four images."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "track_a_experiments/runs/zero_shot_three_models_final_split/test_pairs.jsonl"
RANKINGS = ROOT / "track_a_experiments/runs/full_method_mpaware_joint_hn_test/rankings_image_to_text.jsonl"
OUT = ROOT / "data/examples/image_to_text_top10_examples.json"

image_ids = ["v01_0167_007", "v06_0158_003", "v07_0182_035", "v01_0161_013"]

pairs = {}
with PAIRS.open() as f:
    for line in f:
        row = json.loads(line)
        pairs[row["pair_id"]] = row

rankings = {}
with RANKINGS.open() as f:
    for line in f:
        row = json.loads(line)
        rankings[row["image_id"]] = row

result = []
for image_id in image_ids:
    ranking = rankings[image_id]
    ids = ranking["ranked_pair_ids"][:10]
    result.append({
        "image_id": image_id,
        "image_path": pairs[image_id]["image_path"],
        "gold_pair_id": ranking["gold_pair_id"],
        "gold_rank": ranking["gold_rank"],
        "retrieved_descriptions": [
            {"rank": rank, "pair_id": pair_id, "description": pairs[pair_id]["query"]}
            for rank, pair_id in enumerate(ids, 1)
        ],
    })

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(OUT)
