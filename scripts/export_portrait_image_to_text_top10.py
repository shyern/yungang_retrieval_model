"""Export four portrait (height > width) image-to-text top-10 examples."""
import json
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[1]
pairs_path = root / "track_a_experiments/runs/zero_shot_three_models_final_split/test_pairs.jsonl"
rank_path = root / "track_a_experiments/runs/full_method_mpaware_joint_hn_test/rankings_image_to_text.jsonl"
out_path = root / "data/examples/portrait_image_to_text_top10_examples.json"
ids = ["v01_0170_010", "v02_0182_013", "v03_0160_015", "v06_0154_019"]
pairs = {x["pair_id"]: x for x in map(json.loads, pairs_path.open())}
rankings = {x["image_id"]: x for x in map(json.loads, rank_path.open())}
result = []
for image_id in ids:
    image = Image.open(pairs[image_id]["image_path"])
    width, height = image.size
    image.close()
    assert height > width
    ranking = rankings[image_id]
    result.append({
        "image_id": image_id,
        "image_path": pairs[image_id]["image_path"],
        "width": width,
        "height": height,
        "gold_pair_id": ranking["gold_pair_id"],
        "gold_rank": ranking["gold_rank"],
        "retrieved_descriptions": [
            {"rank": n, "pair_id": pair_id, "description": pairs[pair_id]["query"]}
            for n, pair_id in enumerate(ranking["ranked_pair_ids"][:10], 1)
        ],
    })
out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(out_path)
