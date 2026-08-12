# Fine-Grained Multimodal Evidence Acquisition

This repository is a runnable starting point for studying the visual-evidence
bottleneck in domain-specific Multimodal RAG. It separates the work into two
linked tracks:

1. **Fine-grained visual retrieval**: retrieve the correct object or region
   from visually similar candidates.
2. **Evidence-aware RAG**: retrieve visual, textual, and relational evidence,
   detect missing evidence types, and iteratively fill the evidence bundle.

The first milestone is not a large end-to-end model. It is a controlled
experimental pipeline that can answer: *when visual Recall@K improves, how
much do joint evidence recall and final answer quality improve?*

## Repository layout

```text
configs/                 Experiment settings
data/examples/           Small executable examples of the data contracts
docs/                    Annotation and experiment guidance
scripts/                 CLI entry points
src/mmmrag/              Reusable retrieval, RAG, and metric code
tests/                    Dependency-free baseline tests
```

## Run the first baseline

The sparse RAG baseline only needs Python 3.10+:

```bash
PYTHONPATH=src python scripts/run_rag_baseline.py \
  --corpus data/examples/evidence.jsonl \
  --questions data/examples/questions.jsonl \
  --output outputs/rag_predictions.jsonl

PYTHONPATH=src python scripts/evaluate_rag.py \
  --predictions outputs/rag_predictions.jsonl \
  --questions data/examples/questions.jsonl

python -m unittest discover -s tests -v
```

This baseline is deliberately transparent: BM25-style lexical retrieval per
modality, relation expansion, evidence-sufficiency checks, and targeted query
expansion. It gives the project a testable lower bound before introducing an
LLM or VLM.

## Run visual retrieval

Install the optional vision dependencies:

```bash
python -m pip install -e '.[vision]'
```

Prepare images and update `data/examples/visual_pairs.jsonl`, then run:

```bash
PYTHONPATH=src python scripts/run_visual_retrieval.py \
  --pairs data/examples/visual_pairs.jsonl \
  --output outputs/visual_rankings.jsonl \
  --model openai/clip-vit-base-patch32
```

The script evaluates text-to-image retrieval and reports Recall@1/5/10, MRR,
and mean rank. Replace the example paths with real object crops before use.

For a stronger domain baseline, fine-tune CLIP with in-batch contrastive loss:

```bash
PYTHONPATH=src python scripts/train_visual_retriever.py \
  --train data/processed/retrieval/train.jsonl \
  --validation data/processed/retrieval/validation.jsonl \
  --output-dir outputs/clip-domain
```

## Recommended implementation order

1. Annotate 300-500 pilot object/region pairs, including same-category hard
   negatives, and split by physical object or cave rather than random image.
2. Establish CLIP zero-shot and domain-fine-tuned retrieval results.
3. Build 100-200 pilot QA samples with explicit required visual, text, and
   relation evidence IDs.
4. Compare oracle evidence, retrieved evidence, corrupted visual evidence, and
   text-only evidence under the same generator.
5. Add a learned sufficiency classifier only after the rule-based baseline has
   exposed common missing-evidence patterns.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the concrete experiment
matrix and [docs/DATA.md](docs/DATA.md) for the annotation contract.

