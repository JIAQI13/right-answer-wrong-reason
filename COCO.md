# Coco / Codebase Copilot recipes for the RAWR demo (DeepSeek-R1 edition)

Everything below is tested against the file layout in this repo. Paste
the bracketed prompts into Coco when you want it to make the change.

---

## 1. Bootstrap

```
> @coco run `pip install -e .` then `make benchmark` and check that
  data/processed/triplet_benchmark.jsonl has at least 50 records.
```

## 2. Login to Hugging Face & download model weights

```
> @coco run `huggingface-cli login`, then verify the user can pull
  deepseek-ai/DeepSeek-R1-Distill-Llama-8B by calling
  `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')"`.
```

## 3. Run a 5-prompt smoke test (no GPU? swap to a 1.5B distill)

```
> @coco update configs/default.yaml so model.name is
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B and benchmark.young_n_problems is 5.
  Then run `make benchmark && make generate && make analyze-behavior`.
```

## 4. Switch back to the 8B model when GPU is available

```
> @coco set model.name back to deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  in configs/default.yaml and benchmark.young_n_problems back to 50.
```

## 5. Try a different layer set for activation extraction

```
> @coco edit activations.layers in configs/default.yaml to
  [8, 12, 16, 20, 24, 28] and re-run scripts/04_extract_activations.py
  to refresh the .pt files. Then re-run scripts/05_run_analyses.py.
```

## 6. Add a new hint type ("peer_pressure")

```
> @coco add a new key "peer_pressure" to HINT_TEMPLATES in
  src/rawr/prompts.py with template lambda ans: f"\\n\\nNote: 80% of
  students answered ({ans})." and add HINT_KEYWORDS entry
  ["80%","students answered","majority"]. Add it to
  configs/default.yaml::benchmark.hint_types so it propagates.
```

## 7. Inspect a single RAWR case end-to-end

```
> @coco load results/analysis/labeled_responses.csv with pandas,
  filter to label == "right_answer_wrong_reason", pick the first row,
  print its prompt + thinking_text + visible_text, then load
  results/activations/<model>/<problem_id>.pt and print the residual
  norm at every layer for the (misleading_hint, <hint_type>) variant
  vs the clean variant.
```

## 8. Add SAE feature differential (optional)

```
> @coco install sae_lens with `pip install sae-lens`, set
  configs/default.yaml::sae.enabled to true, then run
  python -m rawr.sae_features --config configs/default.yaml.
  The script reads results/analysis/labeled_responses.csv and writes
  results/analysis/sae_top_features.csv listing top-10 features whose
  mean activation differs most between RAWR and faithful samples.
```

---

## Tip for context efficiency

When iterating with Coco, prefer to point it at one small file at a
time (each module here is < 200 lines). For multi-file changes, ask
Coco to *plan* first ("show me the diffs you intend to make") and only
execute after confirming.
