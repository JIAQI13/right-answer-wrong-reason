# AGENTS.md — Right-Answer Wrong-Reason (RAWR)

## 1. Project Overview

**Purpose**: Detect "right-answer, wrong-reason" (RAWR) failures in open-weight reasoning models by combining behavioral analysis with mechanistic interpretability (TransformerLens activation probing).

**Architecture**: A 6-stage pipeline that:
1. Builds a triplet benchmark from Young's (2026) `cot-faithfulness-open-models` dataset
2. Runs DeepSeek-R1-Distill-Llama-8B on clean / faithful-hint / misleading-hint prompt variants
3. Labels responses by answer correctness and whether reasoning cites the hint
4. Extracts residual-stream activations at the `</think>` token position
5. Runs linear probes and RAWR-vs-faithful distance tests
6. Generates a markdown report with behavioral + mechanistic evidence

**Key Insight**: The token immediately after `</think>` is where the model "has decided but hasn't spoken yet." If the residual at this position encodes shortcut evidence from misleading hints, we have internal proof of RAWR even when the final answer is correct.

**Three-condition design** (project requirement):
- `clean`: no hint (baseline)
- `faithful_hint × 6`: hint points to gold answer (control)
- `misleading_hint × 6`: hint points to wrong answer (failure mode)

**Six hint types** (from Young):
- `sycophancy`, `consistency`, `visual_pattern`, `metadata`, `grader_hacking`, `unethical`

## 2. Build & Commands

### Installation
```bash
make install          # pip install -e .[dev]
# Optional: pip install -e .[sae]  # for SAE feature analysis
huggingface-cli login  # DeepSeek weights require license agreement
```

### Pipeline (one-command via Makefile)
```bash
make benchmark        # scripts/01_build_benchmark.py → data/processed/triplet_benchmark.jsonl
make generate         # scripts/02_generate.py → results/generations/<model>/responses.jsonl
make analyze-behavior # scripts/03_label_behavior.py → results/analysis/labeled_responses.csv
make analyze-mechanistic  # scripts/04-06 → activations + probe + report
make all              # run full pipeline
make smoke            # dry run with 1.5B model (CPU-friendly)
make clean            # remove results/* and data/processed/*
```

### Dry-run / Development
```bash
# Quick smoke test (9 prompts, 1.5B model)
make smoke

# Limit generation for testing
python scripts/02_generate.py --config configs/default.yaml --limit 9 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

## 3. Code Style

- **Python version**: >= 3.10
- **Linting/formatting**: `ruff` (included in `[dev]` extras)
- **Package layout**: `src/rawr/` (src-layout via setuptools)
- **Module convention**: Each module is < 200 lines, single responsibility
- **CLI wrappers**: `scripts/0{1-6}_*.py` are thin wrappers that call `rawr.<module>.main()`
- **Configuration**: YAML-based (`configs/default.yaml`), override via `--config`

### Naming Conventions
- Conditions: `clean`, `faithful_hint`, `misleading_hint`
- Hint types: match Young's directory names exactly
- Labels: `clean_correct`, `faithful_reasoning`, `right_answer_wrong_reason`, `sycophantic_failure`, `confused`
- Activation keys: `<condition>__<hint_type>__L<layer>` (e.g., `misleading_hint__sycophancy__L18`)

## 4. Testing

### Test Framework
- `pytest` is included in `[dev]` dependencies
- No dedicated test suite yet — this is a research prototype
- **Smoke test**: `make smoke` validates the pipeline end-to-end with a small model

### Manual Validation Checklist
1. `data/processed/triplet_benchmark.jsonl` has records with `variants.clean`, `variants.faithful_hint`, `variants.misleading_hint`
2. `results/generations/<model>/responses.jsonl` has `thinking_text` and `visible_text` fields
3. `results/analysis/labeled_responses.csv` has a `label` column
4. `results/activations/<model>/` contains `.pt` files per problem

### Expected Outputs
- **behavior_summary.csv**: condition × hint_type × label counts
- **probe_results.csv**: layer × (in-domain AUC | zero-shot AUCs)
- **rawr_vs_faithful.csv**: per-layer shortcut detection rates
- **report.md**: synthesized findings in `results/analysis/`

## 5. Security

### Data Protection
- No user PII is processed — all data is from public HuggingFace datasets
- Model weights are gated by license (DeepSeek-R1, Llama-3.1) but contain no user data
- Generated responses are stored locally in `results/generations/`

### Secrets Management
- HuggingFace token: use `huggingface-cli login` (stores in `~/.cache/huggingface/`)
- Never commit `.env` files (included in `.gitignore`)
- No API keys are hardcoded in source

### Git Safety
- `.gitignore` excludes: `results/generations/*/`, `results/activations/*/`, `data/processed/*`, `.venv/`, `__pycache__/`
- Large model weights are never committed (downloaded at runtime)

## 6. Configuration

### Primary Config File: `configs/default.yaml`

```yaml
model:
  name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  dtype: bfloat16
  device: auto
  max_new_tokens: 1024
  temperature: 0.0
  thinking_open: "<think>"
  thinking_close: "</think>"

benchmark:
  young_n_problems: 50           # null = use all 498
  hint_types: [sycophancy, consistency, visual_pattern, metadata, grader_hacking, unethical]
  conditions: [clean, faithful_hint, misleading_hint]

generate:
  use_young_precomputed: false   # set to true to skip local generation

activations:
  layers: [10, 14, 18, 22, 26]   # middle-to-late layers for decision content
  hook_name: hook_resid_post
  token_position: answer_token   # right after </think>

analysis:
  probe_train_hint: sycophancy
  probe_test_hints: [consistency, metadata, visual_pattern, grader_hacking, unethical]

sae:
  enabled: false
  release: deepseek-r1-distill-llama-8b-residual
  sae_id: layer_18/width_32k/canonical
```

### Environment Variables
- `HF_TOKEN`: HuggingFace access token (alternative to `huggingface-cli login`)
- `HF_ENDPOINT`: Custom HuggingFace mirror (for network-restricted environments)
- `CUDA_VISIBLE_DEVICES`: GPU selection

### Hardware Requirements
- **GPU**: 1× A100 or RTX 4090 (24 GB) recommended for 8B model in BF16
- **CPU-only**: Use `make smoke` with `DeepSeek-R1-Distill-Qwen-1.5B`
- **No GPU at all**: Set `generate.use_young_precomputed: true` to download pre-generated responses

## Key Modules (src/rawr/)

| Module | Responsibility |
|--------|----------------|
| `prompts.py` | 6 hint templates + `HINT_KEYWORDS` for citation detection |
| `benchmark.py` | Load Young dataset → triplet jsonl (clean + 2×6 hints) |
| `generate.py` | HF inference + `<think>` parsing + answer extraction |
| `label.py` | Assign labels based on answer + hint citation |
| `activations.py` | TransformerLens hook_resid_post at `</think>` position |
| `analysis.py` | H1 cross-hint probe + H3 RAWR-vs-faithful distance test |
| `patching.py` | H2 activation patching (causal evidence) |
| `sae_features.py` | Optional SAE feature differential analysis |
| `report.py` | Auto-compose markdown report from CSVs |

## COCO.md Recipes (Quick Reference)

The repo includes tested recipes in `COCO.md` for common operations:
- Bootstrap and verify benchmark
- HuggingFace login and model download
- 5-prompt smoke test
- Switch between 1.5B and 8B models
- Add new hint types
- Inspect individual RAWR cases
- Enable SAE feature analysis

## Important Notes for Agents

1. **When iterating, prefer small files**: Each module is < 200 lines. Point to one file at a time.
2. **For multi-file changes**: Plan first ("show me the diffs") then execute.
3. **Label semantics matter**: `right_answer_wrong_reason` = misleading condition + correct answer + CoT cites hint. This is the project's target failure mode.
4. **Activation position is critical**: We probe at `</think>` + 1, not at the final answer token. This is where the decision is encoded but not yet verbalized.
5. **Three hypotheses to validate**:
   - H1: Cross-hint linear probe generalizes (shared shortcut subspace)
   - H2: Activation patching flips answers (causal evidence)
   - H3: RAWR residuals are farther from clean than faithful residuals (internal trace)
