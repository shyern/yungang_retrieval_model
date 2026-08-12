# Initial experiment matrix

## Track A: fine-grained visual retrieval

| Experiment | Retriever | Training data | Purpose |
|---|---|---|---|
| A0 | CLIP zero-shot | none | Generic foundation baseline |
| A1 | CLIP fine-tuning | random negatives | Domain adaptation baseline |
| A2 | CLIP fine-tuning | in-batch + hard negatives | Test fine-grained discrimination |
| A3 | A2 + object crop/global image fusion | hard negatives | Test local evidence value |
| A4 | zero/few-shot on external heritage data | external test | Generalization |

Report Recall@1/5/10, MRR, mean rank, and results by attribute, cave, object
scale, and hard-negative difficulty. Also report object-disjoint and cave-
disjoint splits.

## Track B: Multimodal RAG

| Experiment | Visual evidence | Text evidence | Relations | Sufficiency loop |
|---|---|---|---|---|
| B0 | none | retrieved | no | no |
| B1 | retrieved | retrieved | no | no |
| B2 | retrieved | retrieved | yes | no |
| B3 | retrieved | retrieved | yes | yes |
| B4 | oracle | retrieved | yes | no |
| B5 | oracle | oracle | oracle | no |
| B6 | confusable wrong image | oracle | oracle | no |

Keep the generator fixed across B0-B6. Measure visual evidence recall, text
evidence recall, complete evidence-set recall, answer EM/F1, citation precision,
faithfulness, latency, and number of retrieval rounds.

Tune candidate depth on validation data and report both retrieval recall and
evidence precision. A bundle containing every modality is not necessarily a
minimal or semantically sufficient evidence set.

## Key analysis

Bucket questions by visual Recall@K and plot complete evidence recall and answer
accuracy for each bucket. Then compare retrievers while holding the generator
constant. Oracle and corrupted-evidence conditions establish the upper bound
and quantify how strongly visual retrieval errors propagate.

## Immediate engineering tasks

1. Replace sample metadata with 50 real image crops and inspect CLIP failures.
2. Create hard-negative groups from the same iconographic category.
3. Annotate 20 QA examples whose answers genuinely require both modalities.
4. Connect visual rankings to the RAG corpus using stable `evidence_id` values.
5. Add one generator adapter and require it to cite evidence IDs in JSON output.
