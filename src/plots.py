"""
Plotting utilities for GPT2-Zinc baseline vs SmileyLlama experiments.
Focuses on constraint-based generation metrics visualization.

Automatically loads individual experiment result files (baseline_*_results.csv, smiley_*_results.csv)
or combined files if available.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import summarise_group
from .utils import (
    ensure_directory,
    load_results_file,
    load_results_by_pattern,
    filter_valid_molecules,
    load_experiment_results,
    count_result_files,
    compute_properties_df,
)


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
        # Special handling for QED to match original filename
        if prop == "QED":
            filename = "qed_distribution.png"
            if ref_df is not None:
                filename = f"qed_distribution_{dataset_label.lower()}.png"
        else:
            filename = f"{prop.lower()}_histogram.png"
            if ref_df is not None:
                filename = f"{prop.lower()}_histogram_{dataset_label.lower()}.png"
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def plot_qed(model_frames: Dict[str, pd.DataFrame], out_dir: Path, ref_df: Optional[pd.DataFrame] = None, dataset_label: str = "Reference") -> None:
    """Plot QED distribution for generated molecules (optional reference for comparison)."""
    # Reuse plot_histograms for consistency
    plot_histograms(model_frames, properties=["QED"], out_dir=out_dir, ref_df=ref_df, dataset_label=dataset_label)


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
    
    # Use consolidated loading function
    baseline, smc_gradual_df, smc_range_df, smiley_gradual_df, smiley_range_df, smc, smiley = load_experiment_results(
        results_dir=results_dir,
        baseline=args.baseline,
        smc=args.smc,
        smiley=args.smiley,
        temperature=args.temperature,
    )

    # Build model frames dictionary (only include models with data)
    # Distinguish between gradual and range-based constraints if ConstraintLevel is available
    model_frames = {}
    if not baseline.empty:
        model_frames["GPT2-Zinc-87M (range-based)"] = filter_valid_molecules(baseline)
    if not smc.empty:
        # Distinguish between gradual and range-based based on loaded files or ConstraintLevel
        if not smc_gradual_df.empty or not smc_range_df.empty:
            # Both types loaded separately - use the separate dataframes
            if not smc_gradual_df.empty:
                model_frames["GPT2-Zinc-87M+SMC (gradual)"] = filter_valid_molecules(smc_gradual_df)
            if not smc_range_df.empty:
                model_frames["GPT2-Zinc-87M+SMC (range-based)"] = filter_valid_molecules(smc_range_df)
        elif "ConstraintLevel" in smc.columns:
            # Check ConstraintLevel to determine type (for combined files or individual files)
            constraint_levels = smc["ConstraintLevel"].unique()
            has_loosen = "loosen" in constraint_levels
            has_loose = "loose" in constraint_levels
            
            if has_loosen and has_loose:
                # Both types in combined file - split by unique identifiers
                # Gradual: "loosen", "tight", "ultra_tight"; Range-based: "loose", "tight", "ultra_tight"
                # Note: "tight" and "ultra_tight" appear in both, so we assign them to gradual if "loosen" exists
                smc_gradual_split = smc[smc["ConstraintLevel"].isin(["loosen", "tight", "ultra_tight"])]
                smc_range_split = smc[smc["ConstraintLevel"] == "loose"]
                if not smc_gradual_split.empty:
                    model_frames["GPT2-Zinc-87M+SMC (gradual)"] = filter_valid_molecules(smc_gradual_split)
                if not smc_range_split.empty:
                    model_frames["GPT2-Zinc-87M+SMC (range-based)"] = filter_valid_molecules(smc_range_split)
            elif has_loosen:
                # Only gradual
                model_frames["GPT2-Zinc-87M+SMC (gradual)"] = filter_valid_molecules(smc)
            elif has_loose:
                # Only range-based
                model_frames["GPT2-Zinc-87M+SMC (range-based)"] = filter_valid_molecules(smc)
            else:
                # Unknown - use generic
                model_frames["GPT2-Zinc-87M+SMC"] = filter_valid_molecules(smc)
        else:
            # No ConstraintLevel column - use generic label
            model_frames["GPT2-Zinc-87M+SMC"] = filter_valid_molecules(smc)
    
    if not smiley.empty:
        # Distinguish between gradual and range-based based on loaded files or ConstraintLevel
        if not smiley_gradual_df.empty or not smiley_range_df.empty:
            # Both types loaded separately - use the separate dataframes
            if not smiley_gradual_df.empty:
                model_frames["SmileyLlama-8B (gradual)"] = filter_valid_molecules(smiley_gradual_df)
            if not smiley_range_df.empty:
                model_frames["SmileyLlama-8B (range-based)"] = filter_valid_molecules(smiley_range_df)
        elif "ConstraintLevel" in smiley.columns:
            # Check ConstraintLevel to determine type
            constraint_levels = smiley["ConstraintLevel"].unique()
            has_loosen = "loosen" in constraint_levels
            has_loose = "loose" in constraint_levels
            
            if has_loosen and has_loose:
                # Both types in combined file - split by unique identifiers
                smiley_gradual_split = smiley[smiley["ConstraintLevel"].isin(["loosen", "tight", "ultra_tight"])]
                smiley_range_split = smiley[smiley["ConstraintLevel"] == "loose"]
                if not smiley_gradual_split.empty:
                    model_frames["SmileyLlama-8B (gradual)"] = filter_valid_molecules(smiley_gradual_split)
                if not smiley_range_split.empty:
                    model_frames["SmileyLlama-8B (range-based)"] = filter_valid_molecules(smiley_range_split)
            elif has_loosen:
                # Only gradual
                model_frames["SmileyLlama-8B (gradual)"] = filter_valid_molecules(smiley)
            elif has_loose:
                # Only range-based
                model_frames["SmileyLlama-8B (range-based)"] = filter_valid_molecules(smiley)
            else:
                # Unknown - use generic
                model_frames["SmileyLlama-8B"] = filter_valid_molecules(smiley)
        else:
            # No ConstraintLevel column - use generic label
            model_frames["SmileyLlama-8B"] = filter_valid_molecules(smiley)
    
    if not model_frames:
        print("Error: No result files found!")
        print(f"  Looked in: {results_dir}/")
        print("  Expected files:")
        print("    - baseline_results.csv or baseline_*_results.csv")
        print("    - smc_all_results.csv, smc_gradual_results.csv, smc_range_results.csv, or smc_*_results.csv")
        print("    - smiley_all_results.csv, smiley_gradual_results.csv, smiley_range_results.csv, or smiley_*_results.csv")
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
                ref_df = pd.read_csv(ref_path)
                if "smiles" in ref_df.columns:
                    ref_df = ref_df.rename(columns={"smiles": "SMILES"})
                # Compute properties if missing
                if "Valid" not in ref_df.columns or any(p not in ref_df.columns for p in ["MW", "logP", "QED"]):
                    print("  Computing properties for reference dataset...")
                    prop_df = compute_properties_df(ref_df["SMILES"].tolist())
                    ref_df = prop_df
            ref_df = filter_valid_molecules(ref_df)
            dataset_label = ref_path.stem.replace("_", " ").title()
        except Exception as e:
            print(f"  Warning: Could not load reference dataset: {e}")
            print("  Continuing without reference visualization...")

    print(f"\n{'='*80}")
    print("Constraint-Based Generation Visualization")
    print(f"{'='*80}")
    
    # Print loaded data info
    baseline_count, smc_count, smiley_count = count_result_files(
        results_dir=results_dir,
        baseline=args.baseline,
        smc=args.smc,
        smiley=args.smiley,
    )
    
    if not baseline.empty:
        print(f"Loaded {len(baseline)} baseline molecules from {baseline_count} file(s)")
    if not smc.empty:
        print(f"Loaded {len(smc)} SMC molecules from {smc_count} file(s)")
    if not smiley.empty:
        print(f"Loaded {len(smiley)} SmileyLlama molecules from {smiley_count} file(s)")

    # Plot property distributions (optional reference)
    plot_histograms(model_frames, properties=["MW", "logP", "TPSA"], out_dir=fig_dir, ref_df=ref_df, dataset_label=dataset_label)
    plot_qed(model_frames, out_dir=fig_dir, ref_df=ref_df, dataset_label=dataset_label)

    # Compute metrics (no reference needed)
    # Note: summarise_group expects the full dataframe with Adherence column
    # We need to use the original combined dataframes, not just the valid subset
    summary_rows = []
    for label, frame in model_frames.items():
        # Find the corresponding full dataframe (not just valid molecules)
        if "GPT2-Zinc-87M (range-based)" in label:
            full_frame = baseline if not baseline.empty else pd.DataFrame()
        elif "GPT2-Zinc-87M+SMC (gradual)" in label:
            # Use the separate gradual dataframe if available, otherwise extract from combined
            if not smc_gradual_df.empty:
                full_frame = smc_gradual_df
            elif "ConstraintLevel" in smc.columns:
                full_frame = smc[smc["ConstraintLevel"].isin(["loosen", "tight", "ultra_tight"])] if not smc.empty else pd.DataFrame()
            else:
                full_frame = smc if not smc.empty else pd.DataFrame()
        elif "GPT2-Zinc-87M+SMC (range-based)" in label:
            # Use the separate range-based dataframe if available, otherwise extract from combined
            if not smc_range_df.empty:
                full_frame = smc_range_df
            elif "ConstraintLevel" in smc.columns:
                full_frame = smc[smc["ConstraintLevel"] == "loose"] if not smc.empty else pd.DataFrame()
            else:
                full_frame = smc if not smc.empty else pd.DataFrame()
        elif "GPT2-Zinc-87M+SMC" in label:
            full_frame = smc if not smc.empty else pd.DataFrame()
        elif "SmileyLlama-8B (gradual)" in label:
            # Use the separate gradual dataframe if available, otherwise extract from combined
            if not smiley_gradual_df.empty:
                full_frame = smiley_gradual_df
            elif "ConstraintLevel" in smiley.columns:
                full_frame = smiley[smiley["ConstraintLevel"].isin(["loosen", "tight", "ultra_tight"])] if not smiley.empty else pd.DataFrame()
            else:
                full_frame = smiley if not smiley.empty else pd.DataFrame()
        elif "SmileyLlama-8B (range-based)" in label:
            # Use the separate range-based dataframe if available, otherwise extract from combined
            if not smiley_range_df.empty:
                full_frame = smiley_range_df
            elif "ConstraintLevel" in smiley.columns:
                full_frame = smiley[smiley["ConstraintLevel"] == "loose"] if not smiley.empty else pd.DataFrame()
            else:
                full_frame = smiley if not smiley.empty else pd.DataFrame()
        elif "SmileyLlama-8B" in label:
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
