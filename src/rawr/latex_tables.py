"""Generate LaTeX tables from CSV analysis artifacts.

WHY separate module: Keeps LaTeX formatting logic isolated from report generation.
WHY booktabs: Produces clean, professional tables without ugly vertical lines.

All tables use pandas to_latex() with booktabs=True and are saved as standalone
.tex files that can be included via \input{} in the main LaTeX document.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _save_tex(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save DataFrame as LaTeX table with sensible defaults.

    Uses Styler API for pandas >= 3.0 compatibility (booktabs parameter removed).
    Outputs only the tabular content (no outer table environment) so it can be
    wrapped in a custom table environment with caption and label in the main .tex file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract column_format if provided (styler uses different param name)
    column_format = kwargs.pop("column_format", None)
    float_format = kwargs.pop("float_format", "%.3f")
    index = kwargs.pop("index", False)

    # Build styler with booktabs-style rules, hide index
    styler = df.style.hide(axis="index")
    if float_format:
        styler = styler.format(precision=3, thousands=None, decimal=".")

    # Convert to LaTeX with booktabs-style horizontal rules
    # We only want the tabular content, not the outer table environment
    tex = styler.to_latex(
        hrules=True,  # booktabs-style \toprule, \midrule, \bottomrule
        column_format=column_format,
        convert_css=True,
    )

    # Remove outer table environment if present (keep only tabular)
    lines = tex.strip().split("\n")
    tabular_lines = []
    in_tabular = False
    for line in lines:
        if "\\begin{tabular}" in line:
            in_tabular = True
        if in_tabular:
            tabular_lines.append(line)
        if "\\end{tabular}" in line:
            in_tabular = False

    if tabular_lines:
        tex = "\n".join(tabular_lines) + "\n"

    path.write_text(tex)


def make_label_counts(label_counts: pd.DataFrame, output_dir: Path) -> Path:
    """Generate label counts table.

    Transposes the single-row CSV for better readability in LaTeX.
    """
    if label_counts.empty:
        return Path()

    # Transpose for vertical layout (labels as rows)
    row = label_counts.iloc[0]
    data = []
    label_map = {
        "clean_correct": "Clean Correct",
        "faithful_reasoning": "Faithful Reasoning",
        "right_answer_wrong_reason": "Right-Answer Wrong-Reason (RAWR)",
        "sycophantic_failure": "Sycophantic Failure",
    }
    for col, display in label_map.items():
        if col in row:
            data.append({"Label": display, "Count": int(row[col])})

    df = pd.DataFrame(data)
    out_path = output_dir / "label_counts.tex"
    _save_tex(df, out_path, column_format="lr")
    return out_path


def make_behavior_summary(behavior_summary: pd.DataFrame, output_dir: Path) -> Path:
    """Generate behavior summary table (condition x hint_type x label).

    Filters to key columns and renames for readability.
    """
    if behavior_summary.empty:
        return Path()

    # Select and rename key columns
    key_cols = ["condition", "hint_type", "faithful_reasoning",
                "right_answer_wrong_reason", "sycophantic_failure", "confused"]
    df = behavior_summary[[c for c in key_cols if c in behavior_summary.columns]].copy()

    # Rename columns for display
    df = df.rename(columns={
        "condition": "Condition",
        "hint_type": "Hint Type",
        "faithful_reasoning": "Faithful",
        "right_answer_wrong_reason": "RAWR",
        "sycophantic_failure": "Sycophantic",
        "confused": "Confused",
    })

    # Replace underscores in hint types
    if "Hint Type" in df.columns:
        df["Hint Type"] = df["Hint Type"].fillna("—").str.replace("_", " ")

    # Replace underscores in conditions
    df["Condition"] = df["Condition"].str.replace("_", " ")

    out_path = output_dir / "behavior_summary.tex"
    _save_tex(df, out_path, column_format="llrrrr")
    return out_path


def make_probe_results(probe_results: pd.DataFrame, output_dir: Path) -> Path:
    """Generate H1 cross-hint probe results table.

    Formats AUC values as percentages for readability.
    """
    if probe_results.empty:
        return Path()

    df = probe_results.copy()

    # Format AUC columns as percentages
    auc_cols = [c for c in df.columns if c.startswith("auc_")]
    for col in auc_cols:
        df[col] = df[col].apply(lambda x: f"{x*100:.1f}\\%")

    # Rename columns for display
    rename_map = {"layer": "Layer"}
    for col in auc_cols:
        if col == "auc_train_sycophancy":
            rename_map[col] = "In-domain (sycophancy)"
        else:
            hint_name = col.replace("auc_zs_", "").replace("_", " ").title()
            rename_map[col] = f"Zero-shot: {hint_name}"

    df = df.rename(columns=rename_map)

    out_path = output_dir / "probe_results.tex"
    _save_tex(df, out_path, column_format="l" + "c" * (len(df.columns) - 1))
    return out_path


def make_rawr_vs_faithful(rawr_df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate H3 RAWR-vs-faithful results table.

    Adds difference column and formats as percentages.
    """
    if rawr_df.empty:
        return Path()

    df = rawr_df.copy()
    df["diff"] = df["rawr_shortcut_rate"] - df["faith_shortcut_rate"]

    # Format percentages
    df["rawr_shortcut_rate"] = df["rawr_shortcut_rate"].apply(lambda x: f"{x*100:.1f}\\%")
    df["faith_shortcut_rate"] = df["faith_shortcut_rate"].apply(lambda x: f"{x*100:.1f}\\%")
    df["diff"] = df["diff"].apply(lambda x: f"{x*100:+.1f}\\%")

    # Rename columns
    df = df.rename(columns={
        "layer": "Layer",
        "rawr_n": "RAWR (n)",
        "rawr_shortcut_rate": "RAWR Rate",
        "faith_n": "Faithful (n)",
        "faith_shortcut_rate": "Faithful Rate",
        "diff": "Difference",
    })

    out_path = output_dir / "rawr_vs_faithful.tex"
    _save_tex(df, out_path, column_format="lrcrcc")
    return out_path


def make_patching_results(patching_df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate H2 activation patching summary table.

    Aggregates per-label, per-layer flip rates.
    """
    if patching_df.empty:
        return Path()

    # Aggregate results
    summary_rows = []
    labels = patching_df["original_label"].unique()
    layers = sorted(patching_df["layer"].unique())

    for label in labels:
        label_df = patching_df[patching_df["original_label"] == label]
        n_cases = label_df["problem_id"].nunique()

        for layer in layers:
            layer_df = label_df[label_df["layer"] == layer]
            n_total = len(layer_df)
            n_flipped = layer_df["flipped_to_correct"].sum()
            flip_rate = n_flipped / n_total if n_total > 0 else 0

            display_label = label.replace("_", " ").title()
            summary_rows.append({
                "Original Label": display_label,
                "Cases": n_cases,
                "Layer": layer,
                "Total": n_total,
                "Flipped": n_flipped,
                "Flip Rate": f"{flip_rate*100:.1f}\\%",
            })

    df = pd.DataFrame(summary_rows)

    out_path = output_dir / "patching_results.tex"
    _save_tex(df, out_path, column_format="lrcrrc")
    return out_path


def generate_all_tables(
    label_counts: pd.DataFrame,
    behavior_summary: pd.DataFrame,
    probe_results: pd.DataFrame,
    rawr_df: pd.DataFrame,
    patching_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate all LaTeX tables and return list of saved file paths.

    Args:
        label_counts: Overall label distribution
        behavior_summary: Condition x hint_type x label breakdown
        probe_results: H1 cross-hint probe results
        rawr_df: H3 RAWR-vs-faithful results
        patching_df: H2 patching results
        output_dir: Directory to save tables (typically latex/tables/)

    Returns:
        List of paths to generated .tex files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    for table_func, df in [
        (make_label_counts, label_counts),
        (make_behavior_summary, behavior_summary),
        (make_probe_results, probe_results),
        (make_rawr_vs_faithful, rawr_df),
        (make_patching_results, patching_df),
    ]:
        if not df.empty:
            path = table_func(df, output_dir)
            if path.exists():
                generated.append(path)
                print(f"  ✔ table → {path.name}")

    return generated
