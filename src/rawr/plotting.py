"""Generate publication-quality PDF plots for LaTeX reports.

WHY PDF: Vector graphics scale perfectly in LaTeX documents.
WHY separate module: Keeps plotting logic isolated from report generation.

All plots use matplotlib with LaTeX-compatible styling (serif fonts, 11pt base size)
and are saved with bbox_inches='tight' to prevent label cutoff.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Match LaTeX document styling for seamless integration
plt.rcParams.update({
    "text.usetex": False,  # Set to True if LaTeX is available in matplotlib env
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Color palette optimized for both color and grayscale readability
PALETTE = {
    "clean_correct": "#2E86AB",
    "faithful_reasoning": "#44AF69",
    "right_answer_wrong_reason": "#F18F01",
    "sycophantic_failure": "#C73E1D",
    "confused": "#8B8B8B",
}


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure with tight bounding box to prevent label cutoff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(label_counts: pd.DataFrame, output_dir: Path) -> Path:
    """Plot bar chart of overall label distribution.

    Returns path to saved PDF.
    """
    if label_counts.empty:
        return Path()

    row = label_counts.iloc[0]
    labels = ["Clean\\_Correct", "Faithful\\_Reasoning", "RAWR", "Sycophantic\\_Failure"]
    values = [
        int(row.get("clean_correct", 0)),
        int(row.get("faithful_reasoning", 0)),
        int(row.get("right_answer_wrong_reason", 0)),
        int(row.get("sycophantic_failure", 0)),
    ]
    colors = [PALETTE["clean_correct"], PALETTE["faithful_reasoning"],
              PALETTE["right_answer_wrong_reason"], PALETTE["sycophantic_failure"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f"{int(height)}",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Count")
    ax.set_title("Distribution of Behavioral Labels")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Highlight RAWR bar
    bars[2].set_hatch("//")
    bars[2].set_alpha(0.8)

    out_path = output_dir / "label_distribution.pdf"
    _save_fig(fig, out_path)
    return out_path


def plot_probe_auc(probe_results: pd.DataFrame, output_dir: Path) -> Path:
    """Plot H1 cross-hint probe AUC across layers.

    Returns path to saved PDF.
    """
    if probe_results.empty:
        return Path()

    zs_cols = [c for c in probe_results.columns if c.startswith("auc_zs_")]
    if not zs_cols:
        return Path()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each zero-shot AUC line
    for col in zs_cols:
        hint_name = col.replace("auc_zs_", "").replace("_", "\\_")
        ax.plot(probe_results["layer"], probe_results[col],
                marker="o", linewidth=2, label=hint_name)

    # Plot in-domain AUC as dashed line
    ax.plot(probe_results["layer"], probe_results["auc_train_sycophancy"],
            marker="s", linestyle="--", linewidth=2, color="black",
            label="In-domain (sycophancy)")

    # Add 0.7 and 0.85 threshold lines
    ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.7, label="AUC = 0.7")
    ax.axhline(y=0.85, color="gray", linestyle=":", alpha=0.7, label="AUC = 0.85")

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.set_title("H1: Cross-Hint Probe Zero-Shot AUC")
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(probe_results["layer"])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    out_path = output_dir / "probe_auc.pdf"
    _save_fig(fig, out_path)
    return out_path


def plot_rawr_vs_faithful(rawr_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot H3 RAWR-vs-faithful shortcut detection rates.

    Returns path to saved PDF.
    """
    if rawr_df.empty:
        return Path()

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(rawr_df["layer"]))
    width = 0.35

    ax.bar([i - width/2 for i in x], rawr_df["rawr_shortcut_rate"] * 100,
           width, label="RAWR", color=PALETTE["right_answer_wrong_reason"],
           edgecolor="black", linewidth=0.5)
    ax.bar([i + width/2 for i in x], rawr_df["faith_shortcut_rate"] * 100,
           width, label="Faithful", color=PALETTE["faithful_reasoning"],
           edgecolor="black", linewidth=0.5)

    # Add difference annotation
    for i, (_, row) in enumerate(rawr_df.iterrows()):
        diff = (row["rawr_shortcut_rate"] - row["faith_shortcut_rate"]) * 100
        y_max = max(row["rawr_shortcut_rate"], row["faith_shortcut_rate"]) * 100
        ax.text(i, y_max + 2, f"+{diff:.1f}\\%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=PALETTE["right_answer_wrong_reason"])

    # Add 50% reference line
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.7, label="50\\% chance")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Shortcut Detection Rate (\\%)")
    ax.set_title("H3: RAWR vs Faithful Shortcut Detection")
    ax.set_xticks(list(x))
    ax.set_xticklabels(rawr_df["layer"])
    ax.set_ylim(0, 50)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    out_path = output_dir / "rawr_vs_faithful.pdf"
    _save_fig(fig, out_path)
    return out_path


def plot_patching_flips(patching_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot H2 activation patching answer flip rates.

    Returns path to saved PDF.
    """
    if patching_df.empty:
        return Path()

    # Compute flip rates per label and layer
    summary_rows = []
    labels = patching_df["original_label"].unique()
    layers = sorted(patching_df["layer"].unique())

    for label in labels:
        label_df = patching_df[patching_df["original_label"] == label]
        for layer in layers:
            layer_df = label_df[label_df["layer"] == layer]
            n_total = len(layer_df)
            n_flipped = layer_df["flipped_to_correct"].sum()
            flip_rate = n_flipped / n_total if n_total > 0 else 0
            display_label = label.replace("_", "\\_")
            summary_rows.append({
                "label": display_label,
                "layer": layer,
                "flip_rate": flip_rate * 100,
            })

    summary = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label in summary["label"].unique():
        label_data = summary[summary["label"] == label]
        ax.plot(label_data["layer"], label_data["flip_rate"],
                marker="o", linewidth=2, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Flip Rate (\\%)")
    ax.set_title("H2: Activation Patching Answer Flip Rates")
    ax.set_xticks(layers)
    ax.set_ylim(0, max(summary["flip_rate"].max() * 1.2, 10))
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    out_path = output_dir / "patching_flips.pdf"
    _save_fig(fig, out_path)
    return out_path


def generate_all_plots(
    label_counts: pd.DataFrame,
    probe_results: pd.DataFrame,
    rawr_df: pd.DataFrame,
    patching_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate all plots and return list of saved file paths.

    Args:
        label_counts: Overall label distribution
        probe_results: H1 cross-hint probe results
        rawr_df: H3 RAWR-vs-faithful results
        patching_df: H2 patching results
        output_dir: Directory to save plots (typically latex/figures/)

    Returns:
        List of paths to generated PDF files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    for plot_func, df in [
        (plot_label_distribution, label_counts),
        (plot_probe_auc, probe_results),
        (plot_rawr_vs_faithful, rawr_df),
        (plot_patching_flips, patching_df),
    ]:
        if not df.empty:
            path = plot_func(df, output_dir)
            if path.exists():
                generated.append(path)
                print(f"  ✔ plot → {path.name}")

    return generated
