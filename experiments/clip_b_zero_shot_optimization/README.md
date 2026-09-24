# CLIP-B zero-shot optimization

This directory isolates preprocessing and inference experiments that do not
update the OpenAI CLIP ViT-B/32 weights. Source dataset files and the original
baseline outputs are not modified.

The comparison uses the same valid subset of
`data/final_dataset/split/test.json`; records whose images are unavailable are
listed in `data/missing_images.json`.

Experiments:

1. Chinese `description` baseline.
2. Machine-translated English `name`.
3. Machine-translated English `description`.
4. English prompt ensemble.
5. Prompt ensemble with deterministic five-crop image aggregation.

## Result

All rows use the same 573 available test records and the same frozen
`openai/clip-vit-base-patch32` checkpoint.

| Input/inference strategy | R@1 | R@5 | R@10 | MRR | Mean rank |
|---|---:|---:|---:|---:|---:|
| Chinese description (baseline) | 0.17% | 1.05% | 1.75% | 0.0127 | 280.97 |
| English name | 0.70% | 3.84% | 4.54% | 0.0271 | 245.76 |
| English description | **1.75%** | 4.89% | **9.08%** | **0.0470** | **167.79** |
| English name + prompt ensemble | 0.70% | 2.79% | 4.89% | 0.0258 | 248.39 |
| English description + prompt ensemble | 1.57% | **5.41%** | 8.20% | 0.0452 | 177.13 |
| English name + prompt ensemble + five crop | 0.52% | 2.09% | 4.71% | 0.0239 | 244.31 |
| English description + prompt ensemble + five crop | 1.57% | **5.41%** | 8.73% | 0.0467 | 171.90 |

Machine translation provides the only large and consistent gain. Prompt
ensembling trades a small R@5 increase for lower R@1/R@10, while five-crop
averaging does not beat the single-image English-description result. The
recommended frozen-model configuration is therefore the plain translated
English description with the standard full-image preprocessing.

Translation quality is imperfect for specialist Buddhist terminology, so this
result is a reproducible automatic preprocessing baseline rather than an upper
bound from expert English captions.

## Bidirectional multi-positive result

The bidirectional evaluation treats every image sharing the exact original
Chinese description as relevant. The 573 records contain 484 positive groups;
68 are multi-positive groups covering 157 records.

| Query preprocessing | Direction | R@1 | R@5 | R@10 | MRR | Mean rank |
|---|---|---:|---:|---:|---:|---:|
| Chinese description | Text-to-Image | 0.17% | 1.22% | 2.09% | 0.0153 | 257.40 |
| Chinese description | Image-to-Text | 0.35% | 1.40% | 1.92% | 0.0151 | 282.71 |
| English description | Text-to-Image | **2.09%** | **5.41%** | **10.65%** | **0.0528** | **150.90** |
| English description | Image-to-Text | **1.57%** | **5.24%** | **8.03%** | **0.0424** | **200.96** |

Mean Recall is the arithmetic mean of R@1, R@5, and R@10 in both directions:
1.19% for the Chinese baseline and 5.50% for the translated-English setup.
