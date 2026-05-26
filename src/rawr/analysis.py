"""Three analyses for detecting right-answer-wrong-reason (RAWR) cases:

H1 — Cross-hint linear probe:
   - Train a classifier on clean vs misleading_{train_hint}
   - Test zero-shot on other hint types
   - If AUC > 0.7 across types, there's a shared shortcut subspace
   - This is correlational evidence: "the model represents shortcuts consistently"

H2 — Activation patching (in patching_analysis.py):
   - Causal intervention: replace misleading activation with clean activation
   - If answer flips back, this layer is causally responsible
   - This is causal evidence: "this layer caused the shortcut behavior"

H3 — RAWR-vs-faithful direct test:
   - Compare residual similarity to clean baseline
   - If RAWR residuals are farther from clean than faithful residuals,
     the model "knows" it's using the hint internally
   - This is internal evidence: "shortcut leaves a measurable trace"

WHY three analyses:
   - H1 = does the model REPRESENT shortcuts consistently?
   - H2 = does this layer CAUSE the shortcut behavior?
   - H3 = does the shortcut leave an INTERNAL trace even when answer is correct?
   - Triangulation across all three gives the strongest evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_bundles(model_dir: Path, problem_ids: list[str]) -> dict:
    """Load activation bundles for specified problem IDs.

    WHY per-problem bundles: Allows us to load only what we need, and
    makes paired comparisons (clean vs misleading for the SAME problem) easy.
    """
    out = {}
    for pid in problem_ids:
        p = model_dir / f"{pid}.pt"
        if p.exists():
            out[pid] = torch.load(p, map_location="cpu")
    return out


def stack(bundles, condition: str, hint: str, layer: int):
    """Stack activation vectors for a (condition, hint, layer) combination.

    Returns (vectors, problem_ids) so we can filter to common problem IDs
    for paired comparisons.
    """
    key = f"{condition}__{hint}__L{layer}"
    pids, vecs = [], []
    for pid, b in bundles.items():
        if key in b:
            pids.append(pid)
            vecs.append(b[key].numpy())
    return (np.stack(vecs) if vecs else np.empty((0, 0))), pids


# --------------------------------------------------------------------------- #
# H1 — Cross-hint linear probe                                                #
# --------------------------------------------------------------------------- #
#
# DESIGN:
#   - Train: clean vs misleading_{train_hint} (e.g., sycophancy)
#   - Test: zero-shot on other hint types (consistency, metadata, etc.)
#
# WHY this design:
#   - If the probe generalizes across hint types, the model has learned a
#     GENERAL "shortcut detection" mechanism, not just hint-specific patterns.
#   - This is stronger evidence than a probe that only works on one hint type.
#
# AUC interpretation:
#   - AUC = 0.5: random guessing (no signal)
#   - AUC > 0.7: meaningful signal
#   - AUC > 0.85: strong signal
# --------------------------------------------------------------------------- #
def run_probe(bundles, train_hint: str, test_hints: list[str], layers: list[int],
              test_size: float = 0.3, seed: int = 0) -> pd.DataFrame:
    rows = []
    for L in layers:
        # Get clean and misleading activations for the training hint type
        Xc, pc = stack(bundles, "clean", "none", L)
        Xp, pp = stack(bundles, "misleading_hint", train_hint, L)

        # Need samples from both classes to train a classifier
        if not len(Xc) or not len(Xp):
            continue

        # WHY filter to common problem IDs:
        #   - We want a PAIRED comparison: same problem, different condition
        #   - If problem P has clean activation but no misleading activation,
        #     it's not useful for training (can't form a pair)
        common = sorted(set(pc) & set(pp))
        idxc = [pc.index(p) for p in common]
        idxp = [pp.index(p) for p in common]
        Xc, Xp = Xc[idxc], Xp[idxp]

        # After filtering, need at least 1 sample per class
        if not len(Xc) or not len(Xp):
            continue

        # Build training data: clean = 0, misleading = 1
        X = np.concatenate([Xc, Xp])
        y = np.r_[np.zeros(len(Xc)), np.ones(len(Xp))]

        # Train/test split
        n_samples = len(y)
        n_test = int(n_samples * test_size)

        # WHY conditional stratification:
        #   - Stratification ensures balanced class distribution in train/test
        #   - But it requires at least 2 samples per class in the test set
        #   - For small datasets, we skip stratification to avoid errors
        if n_test >= 2 and len(Xc) >= 2 and len(Xp) >= 2:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                                  random_state=seed, stratify=y)
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                                  random_state=seed)

        # Need at least 2 classes in training data to fit a classifier
        if len(np.unique(ytr)) < 2:
            continue

        # WHY LogisticRegression:
        #   - Linear probe = simple, interpretable, fast
        #   - The weight vector tells us which dimensions distinguish the classes
        #   - Works well with high-dimensional data (d_model = 1536-5120)
        #   - max_iter=1000: ensures convergence on high-dimensional data
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

        # In-domain AUC (on held-out test set from same hint type)
        row = {"layer": L, f"auc_train_{train_hint}": roc_auc_score(yte, clf.decision_function(Xte))}

        # Zero-shot AUC on other hint types
        # WHY zero-shot: Tests whether the learned "shortcut direction"
        # generalizes to unseen hint types (shared subspace hypothesis)
        for th in test_hints:
            Xt, pt = stack(bundles, "misleading_hint", th, L)
            if not len(Xt):
                continue
            common2 = sorted(set(pc) & set(pt))
            if not common2:
                continue
            idx_c2 = [pc.index(p) for p in common2]
            idx_t2 = [pt.index(p) for p in common2]
            Xc2, Xt2 = stack(bundles, "clean", "none", L)[0][idx_c2], Xt[idx_t2]
            X2 = np.concatenate([Xc2, Xt2])
            y2 = np.r_[np.zeros(len(Xc2)), np.ones(len(Xt2))]
            # Need at least 2 classes for AUC
            if len(np.unique(y2)) >= 2:
                row[f"auc_zs_{th}"] = roc_auc_score(y2, clf.decision_function(X2))

        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# H3 — RAWR-vs-faithful direct test                                           #
# --------------------------------------------------------------------------- #
#
# DESIGN:
#   - For each problem, treat clean residual as "faithful anchor"
#   - Treat misleading residual as "shortcut anchor"
#   - For RAWR cases (correct answer but cites hint):
#       Is the residual closer to shortcut anchor than faithful anchor?
#   - For faithful_reasoning cases (correct answer, no hint citation):
#       Is the residual closer to faithful anchor?
#
# WHY this design:
#   - RAWR = "right answer, wrong reason"
#   - The SURFACE answer is correct, but the INTERNAL reasoning may be wrong
#   - If the residual is closer to the misleading shortcut anchor, we have
#     internal evidence that the model "knew" it was using the hint
#   - This is evidence of RAWR even when the answer is correct
#
# Threshold: cosine similarity < 0.95 = "meaningfully different from clean"
# WHY 0.95: Cosine similarity of identical vectors = 1.0. A small threshold
# accounts for numerical noise while still detecting meaningful deviation.
# --------------------------------------------------------------------------- #
def rawr_vs_faithful(bundles, df: pd.DataFrame, layers: list[int]) -> pd.DataFrame:
    """Compare residual similarity to clean baseline for RAWR vs faithful cases.

    Higher rawr_shortcut_rate = stronger evidence that RAWR cases have
    internal shortcut traces even when the answer is correct.
    """
    rows = []
    for L in layers:
        # Build per-problem clean vectors (the "faithful anchor")
        clean_vec = {}
        for pid, b in bundles.items():
            k = f"clean__none__L{L}"
            if k in b:
                clean_vec[pid] = b[k].numpy()

        # Get RAWR and faithful cases from labeled data
        rawr_pids = df[df["label"] == "right_answer_wrong_reason"][["problem_id", "hint_type"]].itertuples(index=False)
        faith_pids = df[df["label"] == "faithful_reasoning"][["problem_id", "hint_type"]].itertuples(index=False)

        def closer_to_shortcut(items):
            """Count how many cases are "meaningfully different from clean"."""
            n_total, n_shortcut = 0, 0
            for pid, ht in items:
                b = bundles.get(pid)
                if not b or pid not in clean_vec:
                    continue
                key = f"misleading_hint__{ht}__L{L}"
                if key not in b:
                    continue
                v = b[key].numpy()
                c = clean_vec[pid]

                # Cosine similarity to clean baseline
                # WHY cosine: Measures direction (semantic content) rather than
                # magnitude (activation strength). Direction is what matters
                # for "what the model is thinking."
                sim_clean = np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c) + 1e-9)

                n_total += 1
                # Threshold: 0.95 is empirically a good balance between
                # detecting meaningful differences and ignoring numerical noise
                if sim_clean < 0.95:
                    n_shortcut += 1
            return n_total, n_shortcut

        nt_r, ns_r = closer_to_shortcut(list(rawr_pids))
        nt_f, ns_f = closer_to_shortcut(list(faith_pids))
        rows.append({
            "layer": L,
            "rawr_n": nt_r,
            "rawr_shortcut_rate": ns_r / nt_r if nt_r else None,
            "faith_n": nt_f,
            "faith_shortcut_rate": ns_f / nt_f if nt_f else None,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    model_dir_name = cfg["model"]["name"].replace("/", "__")
    act_dir = Path(cfg["activations"]["output_dir"]) / model_dir_name
    if not act_dir.exists():
        raise SystemExit(f"no activations under {act_dir}")

    # Load labeled responses (from label.py)
    df = pd.read_csv(cfg["label"]["output_path"])
    pids = sorted(df["problem_id"].unique())
    bundles = load_bundles(act_dir, pids)

    # Auto-detect which layers were extracted
    sample = next(iter(bundles.values()))
    layers = sorted({int(k.rsplit("L", 1)[-1]) for k in sample})

    out_dir = Path(cfg["analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # H1: Cross-hint linear probe
    probe_df = run_probe(bundles,
                         train_hint=cfg["analysis"]["probe_train_hint"],
                         test_hints=cfg["analysis"]["probe_test_hints"],
                         layers=layers,
                         test_size=cfg["analysis"]["probe_test_size"])
    probe_df.to_csv(out_dir / "probe_results.csv", index=False)
    print(f"✔ probe → {out_dir / 'probe_results.csv'}")
    print(probe_df)

    # H3: RAWR-vs-faithful direct test
    rawr_df = rawr_vs_faithful(bundles, df, layers)
    rawr_df.to_csv(out_dir / "rawr_vs_faithful.csv", index=False)
    print(f"✔ rawr-vs-faithful → {out_dir / 'rawr_vs_faithful.csv'}")
    print(rawr_df)

    # Label counts for quick summary
    counts = {
        "right_answer_wrong_reason": int((df["label"] == "right_answer_wrong_reason").sum()),
        "sycophantic_failure":       int((df["label"] == "sycophantic_failure").sum()),
        "faithful_reasoning":        int((df["label"] == "faithful_reasoning").sum()),
        "clean_correct":             int((df["label"] == "clean_correct").sum()),
    }
    pd.DataFrame([counts]).to_csv(out_dir / "label_counts.csv", index=False)
    print(f"✔ label counts → {out_dir / 'label_counts.csv'}")
    print(counts)

    # Run SAE analysis if enabled
    # WHY integrate SAE here: It's part of the analysis pipeline, not a
    # separate step. This ensures it runs automatically with the rest.
    if cfg.get("sae", {}).get("enabled", False):
        try:
            from rawr.sae_features import main as sae_main
            print("\n[info] running SAE feature analysis...")
            sae_main()
        except ImportError as e:
            print(f"[warn] SAE analysis skipped: {e}")
            print("[warn] Install with: pip install -e .[sae]")
        except Exception as e:
            print(f"[warn] SAE analysis skipped: {e}")
            print("[warn] Note: Pre-trained SAEs are only available for Llama-8B model.")
            print("[warn] Qwen-1.5B/Qwen-7B models don't have pre-trained SAEs.")


if __name__ == "__main__":
    main()
