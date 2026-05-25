"""H2 — Activation patching analysis (causal evidence).

WHY selective patching (not full dataset):
  - Full dataset patching = 498 problems × 13 variants × 3 layers = ~20K generations
  - That's expensive and unnecessary
  - We only need representative cases to answer "which layer is causal?"

Strategy:
  1. Select N cases from each category:
     - sycophantic_failure: model fell for the hint (answer = wrong_anchor)
     - right_answer_wrong_reason: correct answer but cites hint
     - faithful_reasoning: correct answer, no hint citation
  2. For each case, patch at layers [14, 18, 22] (where H1/H3 show effects)
  3. Record whether patching changes the answer

Interpretation:
  - If patching layer L flips sycophantic_failure → correct:
    Layer L was causally responsible for the shortcut behavior
  - If patching layer L doesn't change RAWR answers:
    The shortcut may be encoded earlier or RAWR is "lucky" rather than causal

Output: patching_results.csv with columns:
  - problem_id, hint_type, original_label
  - layer, patched_answer, answer_changed
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm


def extract_answer_letter(text: str) -> str | None:
    """Extract answer letter (A/B/C/D) from patched generation."""
    if not text:
        return None
    # Try explicit "Answer: X" pattern first
    m = re.search(r"answer\s*[:=]\s*\(?\s*([A-Da-d])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: last (X) pattern
    m = re.findall(r"\(([A-Da-d])\)", text[-300:])
    return m[-1].upper() if m else None


def select_cases(df: pd.DataFrame, label: str, n: int) -> pd.DataFrame:
    """Select N representative cases for a given label.

    WHY random selection with seed:
      - Ensures reproducibility
      - Avoids bias from ordering (e.g., first N might all be same hint type)
    """
    subset = df[df["label"] == label].copy()
    if len(subset) == 0:
        return subset
    # Sample with fixed seed for reproducibility
    return subset.sample(n=min(n, len(subset)), random_state=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cases-per-category", type=int, default=None,
                    help="Override config's patch_cases_per_category")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    mcfg = cfg["model"]
    model_name = args.model or mcfg["name"]

    # Number of cases to select per category
    n_cases = args.cases_per_category or cfg["analysis"].get("patch_cases_per_category", 5)
    layers = cfg["analysis"].get("patch_layers", [14, 18, 22])

    # Load labeled responses
    label_csv = Path(cfg["label"]["output_path"])
    if not label_csv.exists():
        raise SystemExit(f"missing {label_csv}; run label.py first")
    df = pd.read_csv(label_csv)

    # Load benchmark for prompts
    bench_path = Path(cfg["benchmark"]["output_path"])
    bench = {r["problem_id"]: r for r in (json.loads(l) for l in bench_path.open())}

    # Select representative cases
    # WHY these three categories:
    #   - sycophantic_failure: model clearly used the hint (answer changed)
    #   - RAWR: model got right answer but may have used hint internally
    #   - faithful_reasoning: model ignored the hint (control)
    categories = [
        ("sycophantic_failure", "Model fell for the hint"),
        ("right_answer_wrong_reason", "RAWR: right answer, wrong reason"),
        ("faithful_reasoning", "Faithful: ignored hint"),
    ]

    all_cases = []
    for label, desc in categories:
        cases = select_cases(df, label, n_cases)
        print(f"[info] selected {len(cases)} cases for {label}")
        for _, row in cases.iterrows():
            all_cases.append({
                "problem_id": row["problem_id"],
                "hint_type": row["hint_type"],
                "original_label": label,
                "original_answer": row.get("extracted_answer"),
                "gold_answer": row.get("gold_answer"),
                "wrong_answer": row.get("wrong_answer"),
            })

    if not all_cases:
        print("[warn] no cases available for patching; skipping")
        return

    # Load model for patching
    print(f"[info] loading model {model_name} for patching...")
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=mcfg.get("trust_remote_code", False))
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if mcfg.get("dtype") == "bfloat16" else torch.float16,
        device_map="auto",
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )

    # Auto-detect TransformerLens base model
    from rawr.activations import get_tl_base_model
    tl_base_model = get_tl_base_model(hf_model)
    print(f"[info] using TransformerLens config from {tl_base_model}")

    model = HookedTransformer.from_pretrained(
        tl_base_model,
        hf_model=hf_model,
        tokenizer=hf_tok,
        dtype=torch.bfloat16 if mcfg.get("dtype") == "bfloat16" else torch.float16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    del hf_model
    model.eval()

    # Import patching utility
    from rawr.patching import patch_clean_into_misleading

    # Run patching experiments
    results = []
    for case in tqdm(all_cases, desc="patching experiments"):
        pid = case["problem_id"]
        ht = case["hint_type"]
        rec = bench.get(pid)
        if not rec:
            continue

        # Get clean and misleading prompts
        clean_prompt = rec["variants"]["clean"]["prompt"]
        misleading_prompt = rec["variants"]["misleading_hint"][ht]["prompt"]

        for layer in layers:
            # Run patching
            try:
                patched_output = patch_clean_into_misleading(
                    model,
                    clean_prompt=clean_prompt,
                    misleading_prompt=misleading_prompt,
                    layer=layer,
                    max_new_tokens=256,
                )
                patched_answer = extract_answer_letter(patched_output)

                # Determine if answer changed
                orig_ans = str(case["original_answer"]).upper() if case["original_answer"] else None
                patched_ans = str(patched_answer).upper() if patched_answer else None
                answer_changed = (orig_ans is not None and patched_ans is not None
                                  and orig_ans != patched_ans)
                flipped_to_correct = (patched_ans == str(case["gold_answer"]).upper()
                                      if patched_ans and case["gold_answer"] else False)

                results.append({
                    "problem_id": pid,
                    "hint_type": ht,
                    "original_label": case["original_label"],
                    "original_answer": orig_ans,
                    "gold_answer": case["gold_answer"],
                    "layer": layer,
                    "patched_answer": patched_ans,
                    "answer_changed": answer_changed,
                    "flipped_to_correct": flipped_to_correct,
                })
            except Exception as e:
                print(f"[warn] patching failed for {pid} L{layer}: {e}")
                continue

    # Save results
    out_dir = Path(cfg["analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "patching_results.csv"

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(out_path, index=False)
        print(f"✔ patching results → {out_path}")

        # Print summary
        print("\n=== Patching Summary ===")
        for label, _ in categories:
            label_df = results_df[results_df["original_label"] == label]
            if len(label_df) == 0:
                continue
            n_total = len(label_df) // len(layers)  # Each case tested on N layers
            change_rate = label_df["answer_changed"].mean()
            flip_rate = label_df["flipped_to_correct"].mean()
            print(f"\n{label} ({n_total} cases):")
            print(f"  Answer change rate: {change_rate:.1%}")
            print(f"  Flip to correct rate: {flip_rate:.1%}")

            # Per-layer breakdown
            for layer in layers:
                layer_df = label_df[label_df["layer"] == layer]
                layer_flip = layer_df["flipped_to_correct"].mean()
                print(f"    Layer {layer}: flip-to-correct = {layer_flip:.1%}")
    else:
        print("[warn] no patching results generated")
        pd.DataFrame().to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
