# ============================================================================
# RAWR Pipeline Makefile
# ============================================================================
# One-command pipeline for the Right-Answer Wrong-Reason detection project.
#
# Targets:
#   install         Install package with dev dependencies
#   benchmark       Build triplet benchmark from Young dataset (498 problems)
#   generate        Run model on all prompt variants
#   analyze-behavior Label responses (faithful / RAWR / sycophantic)
#   analyze-mechanistic  Activations + H1 probe + H3 test + SAE + Patching
#   all             Run full pipeline
#   smoke           Quick test with small model and limited data
#   clean           Remove all generated data
# ============================================================================

PYTHON ?= python
CONFIG ?= configs/default.yaml
MODEL  ?= deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

.PHONY: install benchmark generate analyze-behavior analyze-mechanistic clean help all smoke

help:
	@echo "RAWR Pipeline — Right-Answer Wrong-Reason Detection"
	@echo ""
	@echo "Targets:"
	@echo "  install              Install package + deps (including SAE)"
	@echo "  benchmark            Build triplet benchmark (498 problems × 13 variants)"
	@echo "  generate             Run model on all prompts (MODEL=... to override)"
	@echo "  analyze-behavior     Label responses by behavior type"
	@echo "  analyze-mechanistic  Full mechanistic analysis:"
	@echo "                         • TransformerLens activations"
	@echo "                         • H1 cross-hint linear probe"
	@echo "                         • H3 RAWR-vs-faithful test"
	@echo "                         • SAE feature analysis (if enabled)"
	@echo "                         • H2 activation patching (selective)"
	@echo "                         • Auto-generated report"
	@echo "  smoke                Quick dry run (1.5B model, limited data)"
	@echo "  all                  Run the full pipeline end-to-end"
	@echo "  clean                Remove results/* and data/processed/*"
	@echo ""
	@echo "Examples:"
	@echo "  make all                                    # Full pipeline with default model"
	@echo "  make generate MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
	@echo "  make smoke                                  # Quick test"

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------
# Install with [dev,sae] to get all optional dependencies including SAE.
# Why SAE by default: It's part of the standard analysis pipeline now.
# ----------------------------------------------------------------------------
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev,sae]
	@echo ""
	@echo "✔ Install complete!"
	@echo "Next: huggingface-cli login  # for DeepSeek model weights"

# ----------------------------------------------------------------------------
# Pipeline Steps
# ----------------------------------------------------------------------------

# Step 1: Build benchmark from Young dataset
# Output: data/processed/triplet_benchmark.jsonl
# 498 problems × 13 variants = 6474 prompts (full dataset)
benchmark:
	$(PYTHON) scripts/01_build_benchmark.py --config $(CONFIG)

# Step 2: Generate model responses
# Output: results/generations/<model>/responses.jsonl
generate:
	$(PYTHON) scripts/02_generate.py --config $(CONFIG) --model $(MODEL)

# Step 3: Label responses by behavior
# Output: results/analysis/labeled_responses.csv, behavior_summary.csv
analyze-behavior:
	$(PYTHON) scripts/03_label_behavior.py --config $(CONFIG)

# Step 4-7: Full mechanistic analysis
#   4) Extract activations with TransformerLens
#   5) Run H1 probe + H3 test + SAE
#   6) Run H2 activation patching (selective)
#   7) Generate final report
analyze-mechanistic:
	$(PYTHON) scripts/04_extract_activations.py --config $(CONFIG) --model $(MODEL)
	$(PYTHON) scripts/05_run_analyses.py --config $(CONFIG)
	$(PYTHON) scripts/07_run_patching.py --config $(CONFIG) --model $(MODEL)
	$(PYTHON) scripts/06_make_report.py --config $(CONFIG)

# ----------------------------------------------------------------------------
# Quick Smoke Test
# ----------------------------------------------------------------------------
# Uses 1.5B model (CPU-friendly) and limited data for quick validation.
# Why 9 prompts = 3 problems × 3 variants (clean + 2 hints)
# ----------------------------------------------------------------------------
smoke:
	$(PYTHON) scripts/01_build_benchmark.py --config $(CONFIG)
	$(PYTHON) scripts/02_generate.py --config $(CONFIG) --limit 9 \
		--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
	$(PYTHON) scripts/03_label_behavior.py --config $(CONFIG)

# ----------------------------------------------------------------------------
# Full Pipeline
# ----------------------------------------------------------------------------
all: benchmark generate analyze-behavior analyze-mechanistic

# ----------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------
clean:
	rm -rf results/generations/* results/activations/* results/analysis/* results/figures/*
	rm -rf data/processed/*
	@echo "✔ Workspace cleaned"
