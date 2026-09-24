"""Export four retrieval examples with verbatim project descriptions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "track_a_experiments/runs/zero_shot_three_models_final_split/test_pairs.jsonl"
RANKINGS = ROOT / "track_a_experiments/runs/full_method_mpaware_joint_hn_test/rankings_image_to_text.jsonl"
OUT = ROOT / "data/examples/four_retrieval_examples.json"

anchors = {
    "供养人图像": "v01_0167_007",
    "动物造像": "v06_0158_003",
    "伎乐天与乐器": "v07_0182_035",
    "龙纹与装饰": "v01_0161_013",
}

records = {}
with PAIRS.open() as f:
    for line in f:
        row = json.loads(line)
        records[row["pair_id"]] = row

ranked = {}
with RANKINGS.open() as f:
    for line in f:
        row = json.loads(line)
        ranked[row["image_id"]] = row

result = []
for theme, image_id in anchors.items():
    assert image_id in ranked
    ids = [image_id] + [x for x in ranked[image_id]["ranked_pair_ids"] if x != image_id][:9]
    assert all(pair_id in records for pair_id in ids)
    result.append({
        "theme": theme,
        "image_id": image_id,
        "source_ranked_pair_ids": ranked[image_id]["ranked_pair_ids"][:10],
        "items": [
            {"rank": i, "label": "correct" if i == 1 else ("hard_negative" if i <= 6 else "negative"),
             "pair_id": pair_id, "description": records[pair_id]["query"]}
            for i, pair_id in enumerate(ids, 1)
        ],
    })

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(OUT)
