"""Make a readable table from the verbatim image-to-text top-10 JSON."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "data/examples/image_to_text_top10_examples.json"
out = root / "data/examples/image_to_text_top10_examples.md"
data = json.loads(src.read_text())

# Hard-negative labels identify visually/semantically confusable candidates.
hard_ranks = {
    "v01_0167_007": {1, 3, 4, 5, 6},
    "v06_0158_003": {2, 3, 4},
    "v07_0182_035": {1, 2, 3, 4, 5, 6, 7, 8},
    "v01_0161_013": {1, 2, 3, 4, 5, 6},
}

lines = ["# CLIP 图像到文本 Top-10 召回示例", ""]
for example in data:
    lines += [f"## 图片 `{example['image_id']}`", "", f"图片路径：`{example['image_path']}`", "", "| 排名 | pair_id | 样本类型 | 完整描述 |", "|---:|---|---|---|"]
    for item in example["retrieved_descriptions"]:
        rank = item["rank"]
        if item["pair_id"] == example["gold_pair_id"]:
            label = "正样本"
        elif rank in hard_ranks[example["image_id"]]:
            label = "难负样本"
        else:
            label = "负样本"
        desc = item["description"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {rank} | `{item['pair_id']}` | {label} | {desc} |")
    lines.append("")
out.write_text("\n".join(lines), encoding="utf-8")
print(out)
