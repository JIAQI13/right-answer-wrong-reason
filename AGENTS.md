# AGENTS.md — Right-Answer Wrong-Reason (RAWR)

## 1. Project Overview

**Purpose**: Detect "right-answer, wrong-reason" (RAWR) failures in open-weight reasoning models by combining behavioral analysis with mechanistic interpretability (TransformerLens activation probing).

**Architecture**: An 8-stage pipeline that:
1. Builds a triplet benchmark from Young's (2026) `cot-faithfulness-open-models` dataset
2. Runs DeepSeek-R1-Distill on clean / faithful-hint / misleading-hint prompt variants (batch inference)
3. Labels responses by answer correctness and whether reasoning cites the hint
4. Extracts residual-stream activations at the `</think/>` token position via TransformerLens
5. Runs H1 cross-hint linear probes and H3 RAWR-vs-faithful distance tests
6. Runs H2 selective activation patching (causal evidence)
7. Generates a markdown report with behavioral + mechanistic evidence
8. (Optional) Generates LaTeX tables/plots and compiles a PDF report via pdflatex

**Key Insight**: The token immediately after `</think/>` is where the model "has decided but hasn't spoken yet." If the residual at this position encodes shortcut evidence from misleading hints, we have internal proof of RAWR even when the final answer is correct.

**Three-condition design**:
- `clean`: no hint (baseline)
- `faithful_hint x 6`: hint points to gold answer (control)
- `misleading_hint x 6`: hint points to wrong answer (treatment)

**Six hint types** (from Young):
- `sycophancy`, `consistency`, `visual_pattern`, `metadata`, `grader_hacking`, `unethical`

**Three hypotheses**:
- H1: Cross-hint linear probe generalizes (shared shortcut subspace)
- H2: Activation patching flips answers (causal evidence)
- H3: RAWR residuals are farther from clean than faithful residuals (internal trace)

**Current model**: `DeepSeek-R1-Distill-Qwen-1.5B` (runs on RTX 3080 10GB)

## 2. Build & Commands

### Installation
```bash
make install          # pip install -e .[dev]
# Optional: pip install -e .[sae]  # for SAE feature analysis
huggingface-cli login  # DeepSeek weights require license agreement
```

### Pipeline (one-command via Makefile)
```bash
make benchmark        # Step 1: Build triplet benchmark (498 x 13 = 6474 prompts)
make generate         # Step 2: Run model inference (batch_size=32)
make analyze-behavior # Step 3: Label responses by behavior type
make analyze-mechanistic  # Steps 4-7: Activations + H1/H3 + SAE + H2 patching + report
make all              # Run full pipeline
make smoke            # Dry run with 1.5B model (9 prompts)
make clean            # Remove results/* and data/processed/*

# LaTeX report generation (Step 8)
make report-latex     # Generate LaTeX assets + compile PDF report
make latex-assets     # Generate only LaTeX tables and figures
make latex-compile    # Compile existing LaTeX sources to PDF
```

### Windows (no make)
```powershell
python scripts/01_build_benchmark.py --config configs/default.yaml
python scripts/02_generate.py --config configs/default.yaml
python scripts/03_label_behavior.py --config configs/default.yaml
python scripts/04_extract_activations.py --config configs/default.yaml
python scripts/05_run_analyses.py --config configs/default.yaml
python scripts/07_run_patching.py --config configs/default.yaml
python scripts/06_make_report.py --config configs/default.yaml

# LaTeX report generation
python scripts/06_make_report.py --config configs/default.yaml --latex --compile
```

### Key CLI Overrides
```bash
--limit N             # Limit number of prompts to generate
--model MODEL_NAME    # Override model from config
--batch-size N        # Override batch size (default: 32)
--config PATH         # Use custom config file
--latex               # Generate LaTeX tables and plots (report.py only)
--compile             # Compile LaTeX to PDF (implies --latex, report.py only)
```

## 3. Code Style

- **Python version**: >= 3.10
- **Linting/formatting**: `ruff` (included in `[dev]` extras)
- **Package layout**: `src/rawr/` (src-layout via setuptools)
- **Module convention**: Each module is < 350 lines, single responsibility
- **CLI wrappers**: `scripts/0{1-7}_*.py` are thin wrappers that call `rawr.<module>.main()`
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
- **label_counts.csv**: overall label distribution
- **behavior_summary.csv**: condition x hint_type x label counts
- **probe_results.csv**: layer x (in-domain AUC | zero-shot AUCs)
- **rawr_vs_faithful.csv**: per-layer shortcut detection rates (with diff column)
- **patching_results.csv**: per-case per-layer patching outcomes
- **report.md**: synthesized findings with executive summary + hypothesis verdicts

### LaTeX Outputs (optional, via `--latex` flag)
- **latex/tables/*.tex**: Auto-generated booktabs-style LaTeX tables
- **latex/figures/*.pdf**: High-resolution vector plots (PDF format)
- **latex/main.pdf**: Compiled 8-page research paper (via pdflatex)
- **latex/main.tex**: Modular LaTeX source with `\input{}` sections

## 5. Security

### Data Protection
- No user PII is processed — all data is from public HuggingFace datasets
- Model weights are gated by license (DeepSeek-R1, Llama-3.1) but contain no user data
- Generated responses are stored locally in `results/generations/`

### Secrets Management
- HuggingFace token: use `huggingface-cli login` (stores in `~/.cache/huggingface/`)
- Never commit `.env` files (included in `.gitignore`)
- No API keys are hardcoded in source
- `HF_ENDPOINT`: Custom HuggingFace mirror (for network-restricted environments)

### Git Safety
- `.gitignore` excludes: `results/generations/*/`, `results/activations/*/`, `data/processed/*`, `.venv/`, `__pycache__/`, system files (.DS_Store, Thumbs.db)
- Large model weights are never committed (downloaded at runtime)
- `labeled_responses.csv` excluded from git (138MB exceeds GitHub 100MB limit)

## 6. Configuration

### Primary Config File: `configs/default.yaml`

```yaml
model:
  name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  dtype: bfloat16
  device: auto
  max_new_tokens: 1024
  temperature: 0.0
  batch_size: 32

benchmark:
  young_n_problems: null          # null = use all 498
  hint_types: [sycophancy, consistency, visual_pattern, metadata, grader_hacking, unethical]
  conditions: [clean, faithful_hint, misleading_hint]

generate:
  use_young_precomputed: false
  batch_size: 32                  # parallel GPU utilization

activations:
  layers: [10, 14, 18, 22, 26]   # middle-to-late layers for decision content
  hook_name: hook_resid_post
  token_position: answer_token    # right after </think/>

analysis:
  probe_train_hint: sycophancy
  probe_test_hints: [consistency, metadata, visual_pattern, grader_hacking, unethical]
  patch_layers: [14, 18, 22]
  patch_cases_per_category: 5

sae:
  enabled: false                  # no pre-trained SAEs for Qwen-1.5B
  release: deepseek-r1-distill-llama-8b-residual
  sae_id: layer_18/width_32k/canonical

reporting:
  output_path: results/analysis/report.md
  figures_dir: results/figures
  latex:
    enabled: true
    main_tex: latex/main.tex
    tables_dir: latex/tables
    figures_dir: latex/figures
    output_pdf: latex/main.pdf
```

### Environment Variables
- `HF_TOKEN`: HuggingFace access token
- `HF_ENDPOINT`: Custom HuggingFace mirror (for China/network restrictions)
- `CUDA_VISIBLE_DEVICES`: GPU selection

## Key Modules (src/rawr/)

| Module | Responsibility |
|--------|----------------|
| `prompts.py` | 6 hint templates + `HINT_KEYWORDS` for citation detection |
| `benchmark.py` | Load Young dataset -> triplet jsonl (clean + 2x6 hints) |
| `generate.py` | Batch HF inference + `</think/>` parsing + answer extraction |
| `label.py` | Assign labels based on answer + hint citation |
| `activations.py` | TransformerLens hook_resid_post at `</think/>` position; auto-detects architecture (Qwen2/Llama) |
| `analysis.py` | H1 cross-hint probe + H3 RAWR-vs-faithful distance test + SAE integration |
| `patching.py` | H2 activation patching utility (clean residual -> misleading forward pass) |
| `patching_analysis.py` | H2 selective patching driver (N cases per category, multiple layers) |
| `sae_features.py` | Optional SAE feature differential analysis via `sae_lens` |
| `report.py` | Auto-compose markdown report + optional LaTeX asset generation + PDF compilation |
| `plotting.py` | Generate publication-quality PDF plots with LaTeX-compatible styling |
| `latex_tables.py` | Convert CSV results to booktabs-style LaTeX tables |

## Important Notes for Agents

1. **When iterating, prefer small files**: Each module is < 350 lines. Point to one file at a time.
2. **For multi-file changes**: Plan first ("show me the diffs") then execute.
3. **Label semantics matter**: `right_answer_wrong_reason` = misleading condition + correct answer + CoT cites hint. This is the project's target failure mode.
4. **Activation position is critical**: We probe at `</think/>` + 1, not at the final answer token. This is where the decision is encoded but not yet verbalized.
5. **Architecture auto-detection**: `activations.py` uses `get_tl_base_model()` to map distilled model names to TransformerLens base configs (e.g., DeepSeek-R1-Distill-Qwen-1.5B -> Qwen2-1.5B).
6. **SAE availability**: Pre-trained SAEs only exist for Llama-8B. For Qwen models, SAE is gracefully skipped with a warning.
7. **Batch generation**: `generate.py` uses left-padded batch inference (batch_size=32) for GPU utilization. Temperature is only passed when `do_sample=True`.
8. **Monkey-patch for offline**: `activations.py` and `patching_analysis.py` patch `AutoConfig.from_pretrained` to avoid downloading configs when the model is already loaded locally.
9. **LaTeX pipeline**: `report.py --latex --compile` generates tables via `latex_tables.py`, plots via `plotting.py`, and compiles the PDF via pdflatex (two passes for cross-references).
10. **TeX PATH auto-detection**: The compilation function checks common TeX installation paths (user-local texlive, /Library/TeX/texbin) to work with non-system TeX installations.
11. **Pandas 3.0 compatibility**: `latex_tables.py` uses the Styler API (`df.style.to_latex(hrules=True)`) since the `booktabs` parameter was removed from `DataFrame.to_latex()` in pandas 3.0.
12. **LaTeX package load order**: `hyperref` must be loaded before `cleveref`; `bookmark` after both. The `main.tex` follows this order strictly.
