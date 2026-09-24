import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "data/examples/portrait_image_to_text_top10_examples.json").read_text())
# Confusable visual candidates are marked hard negatives for presentation.
hard = {"v01_0170_010": {2, 3, 4}, "v02_0182_013": {1, 2, 4, 5, 6},
        "v03_0160_015": {2, 3, 4}, "v06_0154_019": {2, 3, 4, 5, 6}}
lines = ["# 竖版图片的 CLIP 图像到文本 Top-10 召回示例", ""]
for e in data:
    lines += [f"## 图片 `{e['image_id']}`（{e['width']} × {e['height']}）", "", f"图片路径：`{e['image_path']}`", "", "| 排名 | pair_id | 样本类型 | 完整描述 |", "|---:|---|---|---|"]
    for x in e["retrieved_descriptions"]:
        label = "正样本" if x["pair_id"] == e["gold_pair_id"] else ("难负样本" if x["rank"] in hard[e["image_id"]] else "负样本")
        desc = x["description"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {x['rank']} | `{x['pair_id']}` | {label} | {desc} |")
    lines.append("")
(root / "data/examples/portrait_image_to_text_top10_examples.md").write_text("\n".join(lines), encoding="utf-8")
