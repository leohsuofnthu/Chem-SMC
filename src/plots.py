"""
Plotting utilities for GPT2-Zinc+SMC vs SmileyLlama experiments.
Focuses on constraint-based generation metrics visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import summarise_group
from .utils import ensure_directory


def _load(path: str, temperature: Optional[float]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if temperature is not None and "Temperature" in df.columns:
        df = df[np.isclose(df["Temperature"], temperature)]
    return df


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Valid"]].copy() if "Valid" in df.columns else df


def plot_histograms(
    model_frames: Dict[str, pd.DataFrame],
    properties: Iterable[str],
    out_dir: Path,
    ref_df: Optional[pd.DataFrame] = None,
    dataset_label: str = "Reference",
) -> None:
    """Plot property histograms for generated molecules (optional reference for comparison)."""
    for prop in properties:
        plt.figure(figsize=(6, 4))
        # Optional reference distribution
        if ref_df is not None:
            ref_vals = ref_df[prop].dropna()
            if not ref_vals.empty:
                plt.hist(ref_vals, bins=50, density=True, alpha=0.35, label=f"{dataset_label} reference", color="#4C72B0")
        # Generated molecule distributions
        for label, frame in model_frames.items():
            vals = frame[prop].dropna()
            if vals.empty:
                continue
            plt.hist(vals, bins=50, density=True, alpha=0.35, label=label)
        plt.xlabel(prop)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        filename = f"{prop.lower()}_histogram.png"
        if ref_df is not None:
            filename = f"{prop.lower()}_histogram_{dataset_label.lower()}.png"
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def plot_qed(model_frames: Dict[str, pd.DataFrame], out_dir: Path, ref_df: Optional[pd.DataFrame] = None, dataset_label: str = "Reference") -> None:
    """Plot QED distribution for generated molecules (optional reference for comparison)."""
    plt.figure(figsize=(6, 4))
    # Optional reference distribution
    if ref_df is not None:
        ref_vals = ref_df["QED"].dropna()
        if not ref_vals.empty:
            plt.hist(ref_vals, bins=50, density=True, alpha=0.35, label=f"{dataset_label} reference", color="#4C72B0")
    # Generated molecule distributions
    for label, frame in model_frames.items():
        vals = frame["QED"].dropna()
        if vals.empty:
            continue
        plt.hist(vals, bins=50, density=True, alpha=0.35, label=label)
    plt.xlabel("QED")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    filename = "qed_distribution.png"
    if ref_df is not None:
        filename = f"qed_distribution_{dataset_label.lower()}.png"
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_bars(metrics: pd.DataFrame, out_dir: Path, filename: str = "model_bars.png") -> None:
    """Plot bar charts for the 4 core constraint-based metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    metrics = metrics.copy()
    labels = metrics["Model"].tolist()
    
    # Adherence % (most important for constraint-based generation)
    axes[0].bar(labels, metrics["Adherence %"], color="#55A868")
    axes[0].set_title("Adherence %", fontweight="bold")
    axes[0].set_ylim(0, 110)
    axes[0].tick_params(axis="x", rotation=20)
    
    # Valid %
    axes[1].bar(labels, metrics["Valid %"], color="#4C72B0")
    axes[1].set_title("Validity %")
    axes[1].set_ylim(0, 110)
    axes[1].tick_params(axis="x", rotation=20)
    
    # Distinct %
    if "Distinct %" in metrics.columns:
        axes[2].bar(labels, metrics["Distinct %"], color="#DD8452")
        axes[2].set_title("Distinct %")
        axes[2].set_ylim(0, 110)
        axes[2].tick_params(axis="x", rotation=20)
    else:
        axes[2].axis("off")

    # Diversity
    axes[3].bar(labels, metrics["Diversity"], color="#C44E52")
    axes[3].set_title("Diversity (1 - mean Tanimoto)")
    axes[3].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot constraint-based generation metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plots the 4 core constraint-based metrics:
  1. Adherence % - Constraint adherence
  2. Valid % - SMILES validity
  3. Distinct % - Molecule uniqueness
  4. Diversity - Tanimoto diversity

Optional reference dataset for property distribution comparison.

Examples:
  # Plot metrics (no reference needed)
  python -m src.plots

  # Plot with optional reference for comparison
  python -m src.plots --reference data/zinc250k/test.csv
        """
    )
    parser.add_argument("--baseline", type=str, default="results/baseline_results.csv")
    parser.add_argument("--smc", type=str, default="results/smc_results.csv")
    parser.add_argument("--smiley", type=str, default="results/smiley_results.csv")
    parser.add_argument("--reference", type=str, default=None, help="Optional: reference dataset for distribution comparison (CSV or parquet)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--figures", type=str, default="figures")
    args = parser.parse_args()

    baseline = _load(args.baseline, args.temperature)
    smc = _load(args.smc, args.temperature)
    smiley = _load(args.smiley, args.temperature)

    model_frames = {
        "GPT2-Zinc baseline": _valid(baseline),
        "GPT2-Zinc+SMC": _valid(smc),
        "SmileyLlama": _valid(smiley),
    }

    ensure_directory(args.figures)
    fig_dir = Path(args.figures)

    # Optional reference dataset (for visualization only - not required for metrics)
    ref_df = None
    dataset_label = ""
    if args.reference:
        ref_path = Path(args.reference)
        print(f"Loading reference dataset from {ref_path} (optional, for visualization only)...")
        try:
            if ref_path.suffix == ".parquet":
                ref_df = pd.read_parquet(ref_path)
            else:
                # CSV - load and compute properties if needed
                from .utils import compute_properties_df
                ref_df = pd.read_csv(ref_path)
                if "smiles" in ref_df.columns:
                    ref_df = ref_df.rename(columns={"smiles": "SMILES"})
                # Compute properties if missing
                if "Valid" not in ref_df.columns or any(p not in ref_df.columns for p in ["MW", "logP", "QED"]):
                    print("  Computing properties for reference dataset...")
                    prop_df = compute_properties_df(ref_df["SMILES"].tolist())
                    ref_df = prop_df
            ref_df = ref_df[ref_df["Valid"]].copy() if "Valid" in ref_df.columns else ref_df
            ref_df = _valid(ref_df)
            dataset_label = ref_path.stem.replace("_", " ").title()
        except Exception as e:
            print(f"  Warning: Could not load reference dataset: {e}")
            print("  Continuing without reference visualization...")

    print(f"\n{'='*80}")
    print("Constraint-Based Generation Visualization")
    print(f"{'='*80}")

    # Plot property distributions (optional reference)
    plot_histograms(model_frames, properties=["MW", "logP", "TPSA"], out_dir=fig_dir, ref_df=ref_df, dataset_label=dataset_label)
    plot_qed(model_frames, out_dir=fig_dir, ref_df=ref_df, dataset_label=dataset_label)

    # Compute metrics (no reference needed)
    summary_rows = []
    for label, frame in model_frames.items():
        summary = summarise_group(frame)
        summary["Model"] = label
        summary_rows.append(summary)
    metrics = pd.DataFrame(summary_rows)
    
    # Plot bar charts
    plot_bars(metrics, fig_dir, filename="model_bars.png")
    print(f"\nSaved plots to {fig_dir}/")
    print("  - Property histograms (MW, logP, TPSA)")
    print("  - QED distribution")
    print("  - Model comparison bars (Adherence, Valid, Distinct, Diversity)")


if __name__ == "__main__":
    main()
