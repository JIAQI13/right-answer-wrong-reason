"""Build the triplet benchmark directly from the Young (2026)
`richardyoung/cot-faithfulness-open-models` HuggingFace dataset.

WHY Young dataset:
  1. Established benchmark for CoT faithfulness research
  2. Provides 6 hint types covering different shortcut mechanisms
  3. Has 498 problems with gold answers and wrong-anchor targets
  4. Public, reproducible, well-documented

Three-condition design (project requirement):
  - clean: no hint (baseline, what the model truly believes)
  - faithful_hint: hint points to gold answer (control, does hint citation matter?)
  - misleading_hint: hint points to wrong answer (treatment, can hint cause wrong reasoning?)

WHY 13 variants per problem: 1 clean + 6 faithful + 6 misleading = 13

Output (jsonl, one record per problem):
    {
      "problem_id": "...",
      "question_id": "...",
      "question": "...",
      "gold_answer": "B",
      "wrong_answer": "C",
      "hint_types": ["sycophancy", ...],
      "variants": {
         "clean": {"prompt": "..."},
         "faithful_hint":   {"sycophancy": {"prompt": ...}, ...},
         "misleading_hint": {"sycophancy": {"prompt": ...}, ...}
      }
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml

from rawr.prompts import make_prompt, HINT_TEMPLATES


# Default Young dataset slug on HuggingFace
YOUNG_REPO = "richardyoung/cot-faithfulness-open-models"

# Two files are needed from Young's dataset:
#   1) QUESTION_POOL: the actual questions + options + gold labels
#   2) HINTED_RESULTS: model outputs under misleading hints, from which we
#      extract `target_label` = the wrong answer the hint points to.
#
# WHY two files: Young's question pool doesn't include the wrong-anchor
# targets that the hints point to. We need the hinted results to know
# which wrong answer each hint is designed to push the model toward.
QUESTION_POOL_FILE = (
    f"https://huggingface.co/datasets/{YOUNG_REPO}/resolve/main/"
    f"data/combined_500.jsonl"
)
HINTED_RESULTS_FILE = (
    f"https://huggingface.co/datasets/{YOUNG_REPO}/resolve/main/"
    f"results/hinted/deepseek-r1/sycophancy.jsonl"
)


def _load_jsonl(url: str) -> list[dict]:
    """Load a JSONL file from a URL using HuggingFace datasets library.

    WHY use datasets library: Handles caching, authentication, and various
    edge cases (redirects, gzip, etc.) better than raw requests.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("`datasets` not installed; run `pip install -e .`") from e
    try:
        ds = load_dataset("json", data_files=url, split="train")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not download {url!r}: {e}\n"
            f"Tip: huggingface_hub may require `huggingface-cli login` for "
            f"some gated mirrors. Try setting HF_ENDPOINT or HF_TOKEN."
        ) from e
    return list(ds)


def _build_wrong_anchor_map(hinted_rows: list[dict]) -> dict[str, str]:
    """Return {question_id: target_label} from the hinted results file.

    `target_label` is the wrong answer that the misleading hint is designed
    to point to. We need this to construct our misleading_hint prompts.

    WHY we need this: For the misleading condition, the hint must point to
    a SPECIFIC wrong answer. We can't just pick any wrong answer - it needs
    to be the one that Young's experimental design targets.
    """
    mapping: dict[str, str] = {}
    for row in hinted_rows:
        qid = str(row.get("question_id") or row.get("id") or "")
        # Different fields might contain the target answer across dataset versions
        wrong = (row.get("target_label") or row.get("hint_target")
                 or row.get("wrong_anchor") or row.get("target_answer"))
        if qid and wrong:
            mapping[qid] = str(wrong).strip().upper()
    return mapping


def _pick_fallback_wrong(row: dict, gold: str) -> str | None:
    """If we don't have a hinted target_label, pick the first choice_label
    that isn't gold.

    WHY fallback: Young's hinted results file might not cover all questions.
    In that case, we degrade gracefully by picking any wrong answer.
    This is less ideal (the hint might not be "optimized" for this answer),
    but it's better than skipping the question entirely.
    """
    labels = row.get("choice_labels") or []
    for lab in labels:
        lab = str(lab).strip().upper()
        if lab and lab != gold:
            return lab
    return None


def load_young_questions(n: int | None = None) -> list[dict]:
    """Pull (question, gold_answer, wrong_anchor) tuples from Young's dataset.

    Strategy (two-file pull):
      * QUESTION_POOL_FILE  → question text + gold answer
      * HINTED_RESULTS_FILE → wrong_anchor (target_label the hint points at)
      * If hinted file is unavailable, fallback to "first non-gold choice_label"

    WHY deduplicate by question_id: Young's dataset might have duplicate
    entries (e.g., same question appearing in multiple subsets). We want
    each question exactly once in our benchmark.
    """
    print(f"[info] loading Young question pool from {QUESTION_POOL_FILE}")
    pool_rows = _load_jsonl(QUESTION_POOL_FILE)
    print(f"[info] loaded {len(pool_rows)} raw question rows")

    # Try to load the hinted results for wrong_anchor; if it fails, degrade gracefully.
    wrong_map: dict[str, str] = {}
    try:
        print(f"[info] loading hinted wrong-anchor map from {HINTED_RESULTS_FILE}")
        hinted_rows = _load_jsonl(HINTED_RESULTS_FILE)
        wrong_map = _build_wrong_anchor_map(hinted_rows)
        print(f"[info] loaded wrong_anchor for {len(wrong_map)} question_ids")
    except SystemExit:
        print("[warn] could not load hinted results; will fallback to choice_labels")

    seen: set = set()
    out: list[dict] = []
    for row in pool_rows:
        qid = str(row.get("id") or row.get("question_id") or len(out))
        if qid in seen:
            continue

        # Gold: Young uses `correct_label` (e.g. "B") in the pool file.
        # We check multiple field names for robustness across dataset versions.
        gold = (row.get("correct_label") or row.get("gold_answer")
                or row.get("correct_answer") or row.get("answer") or row.get("label"))
        if not gold:
            continue
        gold = str(gold).strip().upper()

        # Question: Young's `formatted` field already contains the stem + (A)..(D).
        # This is the ready-to-use prompt text.
        question = (row.get("formatted") or row.get("question")
                    or row.get("prompt_question") or row.get("base_prompt")
                    or row.get("prompt"))
        if not question:
            continue

        # Wrong anchor: prefer hinted target_label (from Young's experimental design),
        # otherwise fallback to first non-gold choice.
        wrong = wrong_map.get(qid)
        if not wrong:
            wrong = _pick_fallback_wrong(row, gold)
        if not wrong or wrong == gold:
            continue

        seen.add(qid)
        out.append({
            "problem_id": f"young_{qid}",
            "question_id": qid,
            "question": question.strip(),
            "gold_answer": gold,
            "wrong_answer": wrong,
            "domain": row.get("subject") or row.get("dataset") or "unknown",
            "source": "young",
        })
        if n is not None and len(out) >= n:
            break
    print(f"[info] retained {len(out)} unique questions from Young")
    return out


def build_triplet_records(
    problems: Iterable[dict],
    hint_types: list[str],
) -> list[dict]:
    """Build the full triplet benchmark from problem definitions.

    For each problem, we create 13 prompt variants:
      - 1 clean (no hint)
      - 6 faithful_hint (hint points to gold answer)
      - 6 misleading_hint (hint points to wrong answer)

    WHY same template for faithful and misleading: Only the answer letter
    inside the hint differs. This controls for template effects - any
    difference in behavior must be due to the answer the hint points to,
    not the wording of the hint itself.
    """
    records = []
    for p in problems:
        rec = {
            **p,
            "hint_types": list(hint_types),
            "variants": {
                "clean": {
                    "prompt": make_prompt(p["question"], "clean", None,
                                          p["gold_answer"], p["wrong_answer"])
                },
                "faithful_hint":   {},
                "misleading_hint": {},
            },
        }
        for ht in hint_types:
            # Faithful hint: points to gold answer (control condition)
            rec["variants"]["faithful_hint"][ht] = {
                "prompt": make_prompt(p["question"], "faithful_hint", ht,
                                      p["gold_answer"], p["wrong_answer"])
            }
            # Misleading hint: points to wrong answer (treatment condition)
            rec["variants"]["misleading_hint"][ht] = {
                "prompt": make_prompt(p["question"], "misleading_hint", ht,
                                      p["gold_answer"], p["wrong_answer"])
            }
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bcfg = cfg["benchmark"]

    # Validate that all requested hint types have templates
    for ht in bcfg["hint_types"]:
        if ht not in HINT_TEMPLATES:
            raise SystemExit(f"unknown hint_type {ht!r}")

    # Load problems (n = None means all 498)
    n = bcfg.get("young_n_problems")
    problems = load_young_questions(n)
    if not problems:
        raise SystemExit("Young dataset returned no usable rows.")

    # Build triplet records and save
    records = build_triplet_records(problems, bcfg["hint_types"])
    out_path = Path(bcfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Print summary
    n_variants = sum(1 + 2 * len(bcfg["hint_types"]) for _ in records)
    print(f"✔ wrote {len(records)} problems × {1 + 2 * len(bcfg['hint_types'])} "
          f"variants = {n_variants} prompts → {out_path}")


if __name__ == "__main__":
    main()
