# Data contracts

## Retrieval pairs

Each line in the retrieval JSONL represents one query-positive pair. Multiple
captions may point to the same image or object.

```json
{"pair_id":"p001","query":"菩萨右手持莲蕾","image_path":"images/crop_001.jpg","image_id":"img001","object_id":"obj001","cave_id":"cave_020","attributes":["菩萨","持物:莲蕾"],"split":"train"}
```

Use `object_id` as the primary leakage group. Near-duplicate views of one
physical object must not cross train/test boundaries. A stronger generalization
split holds out entire caves or iconographic categories.

Hard negatives should be recorded explicitly when available:

```json
{"pair_id":"p002","query":"菩萨右手持莲蕾","image_path":"images/crop_002.jpg","image_id":"img002","object_id":"obj002","hard_negative_for":["p001"],"attributes":["菩萨","持物:宝珠"]}
```

## Evidence corpus

Every evidence unit has a stable ID and one modality:

```json
{"evidence_id":"v001","modality":"visual","content":"第20窟主尊面部与袈裟局部","image_path":"images/v001.jpg","object_id":"obj020","relations":["r001"]}
{"evidence_id":"t001","modality":"text","content":"文献对第20窟主尊造型的描述。","source":"catalogue:42","relations":["r001"]}
{"evidence_id":"r001","modality":"relation","content":"obj020 located_in cave020; obj020 described_by t001","relations":["v001","t001"]}
```

For visual entries, `content` is searchable metadata, not a substitute for the
image. Later baselines should concatenate or rerank lexical scores with image
embedding scores.

## QA samples

```json
{"question_id":"q001","question":"...","answer":"...","required_modalities":["visual","text","relation"],"gold_evidence_ids":["v001","t001","r001"],"target_hints":{"visual":"主尊局部","text":"造型文献","relation":"洞窟与对象关系"}}
```

`gold_evidence_ids` should contain the minimal sufficient evidence set. When
several sets are valid, add `alternative_evidence_sets`, and count a prediction
correct if it covers any one complete set.

## Annotation checks

- Report expert agreement for object identity, attributes, and evidence support.
- Include unanswerable questions and visually confusable counterexamples.
- Record image and literature licenses before scaling annotation.
- Deduplicate crops using both object metadata and perceptual hashes.

