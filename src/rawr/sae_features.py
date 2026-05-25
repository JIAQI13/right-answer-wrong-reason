"""Optional SAE (Sparse Autoencoder) feature analysis.

WHAT is an SAE:
  - A sparse autoencoder trained on transformer activations
  - Decomposes a d_model-dimensional vector into ~32K sparse features
  - Each feature corresponds to an interpretable "concept" or "circuit"

WHY use SAE for RAWR detection:
  - H1 probe tells us "there's a shortcut subspace" but not WHAT it is
  - SAE tells us "which specific features fire differently in RAWR vs faithful"
  - Example: Feature 12345 fires when model thinks about "authority" or "leaked info"

Strategy:
  1. Load pre-trained SAE for the target layer
  2. Encode misleading_hint activations for RAWR and faithful cases
  3. Find features with largest mean difference (RAWR_mean - faithful_mean)
  4. Top differential features = candidates for "shortcut features"

Note: Pre-trained SAEs are only available for certain models.
Check sae-lens library for available releases.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Check if SAE is enabled in config
    if not cfg.get("sae", {}).get("enabled", False):
        print("[info] sae.enabled = false in config; skipping.")
        return

    # Try to import sae_lens (optional dependency)
    try:
        from sae_lens import SAE
    except ImportError:
        raise SystemExit("`sae_lens` not installed; run `pip install -e .[sae]`")

    # Load labeled responses to identify RAWR vs faithful cases
    label_csv = Path(cfg["label"]["output_path"])
    df = pd.read_csv(label_csv)

    # Locate activation bundles
    model_name = cfg["model"]["name"].replace("/", "__")
    act_dir = Path(cfg["activations"]["output_dir"]) / model_name

    # Load pre-trained SAE
    # WHY from_pretrained: SAEs are trained on specific model activations;
    # using the wrong SAE gives meaningless features
    sae, _, _ = SAE.from_pretrained(
        release=cfg["sae"]["release"],
        sae_id=cfg["sae"]["sae_id"],
    )
    sae = sae.eval()

    # Extract layer number from sae_id (format: "layer_18/width_32k/canonical")
    # WHY we need the layer: SAE is trained on activations from a specific layer;
    # we must encode activations from the SAME layer
    L = int(cfg["sae"]["sae_id"].split("/")[0].split("_")[1])

    # Encode activations and separate by label
    rawr_acts, faith_acts = [], []
    for _, row in df.iterrows():
        bundle_p = act_dir / f"{row['problem_id']}.pt"
        if not bundle_p.exists():
            continue
        bundle = torch.load(bundle_p, map_location="cpu")

        # We look at misleading_hint activations because that's where
        # the shortcut vs faithful distinction matters
        key = f"misleading__{row['hint_type']}__L{L}"
        if key not in bundle:
            continue

        # Encode activation through SAE to get sparse feature activations
        # WHY detach: SAE encoding is inference, no gradients needed
        feats = sae.encode(bundle[key].unsqueeze(0)).squeeze(0).detach().numpy()

        # Separate into RAWR and faithful groups
        if row["label"] == "right_answer_wrong_reason":
            rawr_acts.append(feats)
        elif row["label"] == "faithful_reasoning":
            faith_acts.append(feats)

    # Need samples from both groups to compute differential
    if not rawr_acts or not faith_acts:
        print("[warn] not enough RAWR or faithful samples — skipping SAE diff.")
        print(f"[warn] RAWR: {len(rawr_acts)}, faithful: {len(faith_acts)}")
        return

    # Compute mean activation per feature for each group
    # WHY mean: Smooths out noise, gives typical activation pattern
    R = np.stack(rawr_acts).mean(axis=0)
    F = np.stack(faith_acts).mean(axis=0)

    # Find features with largest difference (RAWR - faithful)
    # Positive diff = feature fires MORE in RAWR cases = candidate shortcut feature
    diff = R - F
    top = np.argsort(-diff)[: cfg["sae"]["topk"]]

    # Save results
    out_path = Path(cfg["analysis"]["output_dir"]) / "sae_top_features.csv"
    pd.DataFrame({
        "feature": top,
        "rawr_mean": R[top],
        "faith_mean": F[top],
        "diff": diff[top],
    }).to_csv(out_path, index=False)

    print(f"✔ top SAE features → {out_path}")
    print(f"  Analyzed: {len(rawr_acts)} RAWR, {len(faith_acts)} faithful cases")
    print(f"  Layer: {L}, Top-K: {cfg['sae']['topk']}")


if __name__ == "__main__":
    main()
