# results/

This directory is populated by the pipeline:

* `generations/<model>/responses.jsonl`  — per-prompt model outputs
* `activations/<model>/<problem_id>.pt`  — TransformerLens caches
* `analysis/labeled_responses.csv`        — every row labeled with one of
  `faithful_reasoning`, `right_answer_wrong_reason`, ...
* `analysis/behavior_summary.csv`         — pivot table by condition × hint × label
* `analysis/probe_results.csv`            — H1 cross-hint probe AUCs
* `analysis/label_counts.csv`             — quick-glance summary
* `analysis/sae_top_features.csv`         — H3 top differential SAE features (optional)
* `figures/`                              — plots produced by analysis scripts
* `analysis/report.md`                    — auto-composed report
