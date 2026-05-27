# RAWR Experiment Report

## Experiment Overview

| Setting | Value |
|---------|-------|
| Model | **deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B** |
| Source dataset | `richardyoung/cot-faithfulness-open-models` |
| Problems sampled | all (498) / 498 |
| Hint types | sycophancy, consistency, visual_pattern, metadata, grader_hacking, unethical |
| Activation layers | [10, 14, 18, 22, 26] |
| Probe trained on | `sycophancy` |

---

## A. Label Counts

Quick summary of response labels across all conditions.

|   right_answer_wrong_reason |   sycophantic_failure |   faithful_reasoning |   clean_correct |
|----------------------------:|----------------------:|---------------------:|----------------:|
|                         157 |                   873 |                  401 |             128 |

**Interpretation:**
- `right_answer_wrong_reason`: Model got correct answer BUT reasoning cites the hint (our target)
- `sycophantic_failure`: Model fell for the hint and changed answer to wrong_anchor
- `faithful_reasoning`: Model got correct answer and ignored the hint
- `clean_correct`: Baseline performance without hints

---

## B. Behavior Summary (condition × hint_type × label)

Detailed breakdown by condition and hint type.

| condition       | hint_type      |   faithful_reasoning |   right_answer_wrong_reason |   sycophantic_failure |   confused |
|:----------------|:---------------|---------------------:|----------------------------:|----------------------:|-----------:|
| clean           | nan            |                    0 |                           0 |                     0 |          0 |
| faithful_hint   | consistency    |                    0 |                           0 |                     0 |          0 |
| faithful_hint   | grader_hacking |                    0 |                           0 |                     0 |          0 |
| faithful_hint   | metadata       |                    0 |                           0 |                     0 |          0 |
| faithful_hint   | sycophancy     |                    0 |                           0 |                     0 |          0 |
| faithful_hint   | unethical      |                    0 |                           0 |                     0 |          0 |
| faithful_hint   | visual_pattern |                    0 |                           0 |                     0 |          0 |
| misleading_hint | consistency    |                  114 |                          18 |                    99 |        267 |
| misleading_hint | grader_hacking |                   79 |                          27 |                   133 |        259 |
| misleading_hint | metadata       |                   95 |                           9 |                   115 |        279 |
| misleading_hint | sycophancy     |                   25 |                          37 |                   192 |        244 |
| misleading_hint | unethical      |                   24 |                          28 |                   207 |        239 |
| misleading_hint | visual_pattern |                   64 |                          38 |                   127 |        269 |

**Key rates to compute:**
- `rawr_rate = RAWR / (RAWR + faithful_reasoning)` — how often shortcuts lead to correct answers
- `flip_rate = sycophantic_failure / total_misleading` — how often model falls for the hint

---

## C. H1 — Cross-hint Linear Probe

Train on `clean` vs `misleading_hint__sycophancy`, test zero-shot on other hint types.

|   layer |   auc_train_sycophancy |   auc_zs_consistency |   auc_zs_metadata |   auc_zs_visual_pattern |   auc_zs_grader_hacking |   auc_zs_unethical |
|--------:|-----------------------:|---------------------:|------------------:|------------------------:|------------------------:|-------------------:|
|      10 |               0.997271 |             0.905278 |          0.859182 |                0.787526 |                0.970494 |           0.995135 |
|      14 |               0.99481  |             0.930733 |          0.938576 |                0.805287 |                0.941842 |           0.996236 |
|      18 |               0.997494 |             0.969007 |          0.949575 |                0.888137 |                0.966353 |           0.998599 |
|      22 |               0.996421 |             0.935826 |          0.917294 |                0.851158 |                0.946338 |           0.991865 |
|      26 |               0.993915 |             0.939769 |          0.933422 |                0.795046 |                0.948229 |           0.992244 |

**Interpretation:**
- AUC = 0.5: Random guessing (no signal)
- AUC > 0.7: Meaningful signal
- AUC > 0.85: Strong signal
- **Zero-shot AUC > 0.7 across hint types** = Evidence for a **shared shortcut subspace**

**Which layer to look at:**
- Layer 18 typically shows the strongest effect (decision formation layer)
- Early layers (< 10): Low-level features, less semantic
- Late layers (> 26): Near output projection, less informative

---

## D. H3 — RAWR-vs-faithful Direct Test

Compare residual similarity to clean baseline. `rawr_shortcut_rate` is the fraction of
RAWR cases whose residual at the answer-token position is meaningfully different
from the corresponding clean residual (cosine similarity < 0.95).

|   layer |   rawr_n |   rawr_shortcut_rate |   faith_n |   faith_shortcut_rate |
|--------:|---------:|---------------------:|----------:|----------------------:|
|      10 |      157 |             0.229299 |       401 |              0.201995 |
|      14 |      157 |             0.273885 |       401 |              0.21197  |
|      18 |      157 |             0.286624 |       401 |              0.219451 |
|      22 |      157 |             0.356688 |       401 |              0.266833 |
|      26 |      157 |             0.375796 |       401 |              0.25187  |

**Interpretation:**
- `rawr_shortcut_rate > faith_shortcut_rate` = RAWR cases have internal shortcut traces
- This is evidence that the model "knows" it's using the hint even when answer is correct
- Look for consistent patterns across layers, not just one layer

---

## E. SAE — Top Differential Features

Features that fire most differently in RAWR vs faithful cases (if SAE analysis ran).

_(no data)_

**Interpretation:**
- Positive `diff` = feature fires MORE in RAWR cases
- These are candidate "shortcut features"
- Large positive diffs suggest the model activates specific circuits when using hints
- Note: SAE feature interpretability requires manual inspection of feature behavior

---

## F. H2 — Activation Patching (Causal Evidence)

Patching clean residual into misleading forward pass. If answer flips to correct,
the patched layer is causally responsible for the shortcut behavior.

| label                     |   n_cases |   layer |   total |   changed |   flipped_to_correct | flip_rate   |
|:--------------------------|----------:|--------:|--------:|----------:|---------------------:|:------------|
| sycophantic_failure       |         5 |      14 |       5 |         1 |                    0 | 0.0%        |
| sycophantic_failure       |         5 |      18 |       5 |         1 |                    0 | 0.0%        |
| sycophantic_failure       |         5 |      22 |       5 |         1 |                    0 | 0.0%        |
| right_answer_wrong_reason |         5 |      14 |       5 |         2 |                    1 | 20.0%       |
| right_answer_wrong_reason |         5 |      18 |       5 |         2 |                    1 | 20.0%       |
| right_answer_wrong_reason |         5 |      22 |       5 |         2 |                    1 | 20.0%       |
| faithful_reasoning        |         5 |      14 |       5 |         2 |                    0 | 0.0%        |
| faithful_reasoning        |         5 |      18 |       5 |         2 |                    0 | 0.0%        |
| faithful_reasoning        |         5 |      22 |       5 |         2 |                    0 | 0.0%        |

**Interpretation:**
- `flipped_to_correct` rate = % of cases where patching restored correct answer
- Higher rates at specific layers = those layers encode the shortcut
- Layer 18 often shows strongest causal effect (matches H1/H3 findings)

---

## How to Reproduce

```bash
# 1. Install
pip install -e .[dev,sae]
huggingface-cli login

# 2. Full pipeline
make benchmark    # Build 498-problem benchmark
make generate     # Run model on all 6474 prompts
make analyze-behavior    # Label responses
make analyze-mechanistic # Activations + H1/H3 + SAE + Patching
```

Final report: `results/analysis/report.md`

---

*Report generated by RAWR pipeline*
