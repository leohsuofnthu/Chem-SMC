"""
Plotting utilities for GPT2-Zinc baseline vs SmileyLlama experiments.
Focuses on constraint-based generation metrics visualization.

Automatically loads individual experiment result files (baseline_*_results.csv, smiley_*_results.csv)
or combined files if available.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import summarise_group
from .utils import ensure_directory


def _load(path: str, temperature: Optional[float]) -> pd.DataFrame:
    """Load a single results CSV file."""
    df = pd.read_csv(path)
    if temperature is not None and "Temperature" in df.columns:
        df = df[np.isclose(df["Temperature"], temperature)]
    return df


def _load_results_by_pattern(pattern: str, temperature: Optional[float] = None) -> pd.DataFrame:
    """
    Load all results files matching a glob pattern and combine them.
    
    Args:
        pattern: Glob pattern to match result files (e.g., 'results/baseline_*_results.csv')
        temperature: Optional temperature filter
    
    Returns:
        Combined DataFrame, or empty DataFrame if no files found.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        df = _load(f, temperature)
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


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
    if metrics.empty:
        print("Warning: No metrics data to plot. Skipping bar chart.")
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    metrics = metrics.copy()
    labels = metrics["Model"].tolist()
    
    # Adherence % (most important for constraint-based generation)
    if "Adherence %" in metrics.columns:
        axes[0].bar(labels, metrics["Adherence %"], color="#55A868")
        axes[0].set_title("Adherence %", fontweight="bold")
        axes[0].set_ylim(0, 110)
        axes[0].tick_params(axis="x", rotation=20)
    else:
        axes[0].axis("off")
    
    # Valid %
    if "Valid %" in metrics.columns:
        axes[1].bar(labels, metrics["Valid %"], color="#4C72B0")
        axes[1].set_title("Validity %")
        axes[1].set_ylim(0, 110)
        axes[1].tick_params(axis="x", rotation=20)
    else:
        axes[1].axis("off")
    
    # Distinct %
    if "Distinct %" in metrics.columns:
        axes[2].bar(labels, metrics["Distinct %"], color="#DD8452")
        axes[2].set_title("Distinct %")
        axes[2].set_ylim(0, 110)
        axes[2].tick_params(axis="x", rotation=20)
    else:
        axes[2].axis("off")

    # Diversity
    if "Diversity" in metrics.columns:
        axes[3].bar(labels, metrics["Diversity"], color="#C44E52")
        axes[3].set_title("Diversity (1 - mean Tanimoto)")
        axes[3].tick_params(axis="x", rotation=20)
    else:
        axes[3].axis("off")

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
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline results file or pattern. Default: auto-detect results/baseline_*_results.csv"
    )
    parser.add_argument(
        "--smc",
        type=str,
        default=None,
        help="SMC results file or pattern. Default: auto-detect results/smc_*_results.csv"
    )
    parser.add_argument(
        "--smiley",
        type=str,
        default=None,
        help="SmileyLlama results file or pattern. Default: auto-detect results/smiley_*_results.csv"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result files (default: results)"
    )
    parser.add_argument("--reference", type=str, default=None, help="Optional: reference dataset for distribution comparison (CSV or parquet)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--figures", type=str, default="figures")
    args = parser.parse_args()

    # Auto-detect result files if not specified
    results_dir = args.results_dir
    
    if args.baseline is None:
        # Try combined file first, then individual files
        baseline_combined = Path(results_dir) / "baseline_results.csv"
        if baseline_combined.exists():
            baseline = _load(str(baseline_combined), args.temperature)
        else:
            baseline = _load_results_by_pattern(f"{results_dir}/baseline_*_results.csv", args.temperature)
    else:
        if "*" in args.baseline:
            baseline = _load_results_by_pattern(args.baseline, args.temperature)
        else:
            baseline = _load(args.baseline, args.temperature)
    
    if args.smc is None:
        # Try combined file first, then individual files
        smc_combined = Path(results_dir) / "smc_results.csv"
        if smc_combined.exists():
            smc = _load(str(smc_combined), args.temperature)
        else:
            smc = _load_results_by_pattern(f"{results_dir}/smc_*_results.csv", args.temperature)
    else:
        if "*" in args.smc:
            smc = _load_results_by_pattern(args.smc, args.temperature)
        else:
            smc = _load(args.smc, args.temperature)
    
    if args.smiley is None:
        # Try combined file first, then individual files
        smiley_combined = Path(results_dir) / "smiley_results.csv"
        if smiley_combined.exists():
            smiley = _load(str(smiley_combined), args.temperature)
        else:
            smiley = _load_results_by_pattern(f"{results_dir}/smiley_*_results.csv", args.temperature)
    else:
        if "*" in args.smiley:
            smiley = _load_results_by_pattern(args.smiley, args.temperature)
        else:
            smiley = _load(args.smiley, args.temperature)

    # Build model frames dictionary (only include models with data)
    model_frames = {}
    if not baseline.empty:
        model_frames["GPT2-Zinc baseline"] = _valid(baseline)
    if not smc.empty:
        model_frames["GPT2-Zinc+SMC"] = _valid(smc)
    if not smiley.empty:
        model_frames["SmileyLlama"] = _valid(smiley)
    
    if not model_frames:
        print("Error: No result files found!")
        print(f"  Looked in: {results_dir}/")
        print("  Expected files: baseline_*_results.csv, smc_*_results.csv, or smiley_*_results.csv")
        return

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
    
    # Print loaded data info
    if not baseline.empty:
        baseline_files = glob.glob(f"{results_dir}/baseline_*_results.csv") if args.baseline is None else []
        file_count = len(baseline_files) if baseline_files else (1 if args.baseline else 0)
        print(f"Loaded {len(baseline)} baseline molecules from {file_count} file(s)")
    if not smc.empty:
        smc_files = glob.glob(f"{results_dir}/smc_*_results.csv") if args.smc is None else []
        file_count = len(smc_files) if smc_files else (1 if args.smc else 0)
        print(f"Loaded {len(smc)} SMC molecules from {file_count} file(s)")
    if not smiley.empty:
        smiley_files = glob.glob(f"{results_dir}/smiley_*_results.csv") if args.smiley is None else []
        file_count = len(smiley_files) if smiley_files else (1 if args.smiley else 0)
        print(f"Loaded {len(smiley)} SmileyLlama molecules from {file_count} file(s)")

    # Plot property distributions (optional reference)
    plot_histograms(model_frames, properties=["MW", "logP", "TPSA"], out_dir=fig_dir, ref_df=ref_df, dataset_label=dataset_label)
    plot_qed(model_frames, out_dir=fig_dir, ref_df=ref_df, dataset_label=dataset_label)

    # Compute metrics (no reference needed)
    # Note: summarise_group expects the full dataframe with Adherence column
    # We need to use the original combined dataframes, not just the valid subset
    summary_rows = []
    for label, frame in model_frames.items():
        # Find the corresponding full dataframe (not just valid molecules)
        if "GPT2-Zinc baseline" in label:
            full_frame = baseline if not baseline.empty else pd.DataFrame()
        elif "GPT2-Zinc+SMC" in label:
            full_frame = smc if not smc.empty else pd.DataFrame()
        elif "SmileyLlama" in label:
            full_frame = smiley if not smiley.empty else pd.DataFrame()
        else:
            full_frame = frame
        
        if full_frame.empty:
            continue
            
        # Use full frame for metrics computation (summarise_group handles Valid column internally)
        summary = summarise_group(full_frame)
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
