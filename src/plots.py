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
    """Plot property histograms for generated molecules (optional reference for comparison) with improved styling."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Color palette for models
    model_colors = [
        "#2E7D32", "#1976D2", "#F57C00", "#C62828", "#7B1FA2", 
        "#00897B", "#E64A19", "#5D4037", "#455A64", "#D32F2F"
    ]
    
    for prop in properties:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Optional reference distribution
        if ref_df is not None:
            ref_vals = ref_df[prop].dropna()
            if not ref_vals.empty:
                ax.hist(ref_vals, bins=50, density=True, alpha=0.4, 
                       label=f"{dataset_label} reference", color="#4C72B0", 
                       edgecolor='white', linewidth=0.5, linestyle='--')
        
        # Generated molecule distributions
        color_idx = 0
        for label, frame in model_frames.items():
            vals = frame[prop].dropna()
            if vals.empty:
                continue
            color = model_colors[color_idx % len(model_colors)]
            ax.hist(vals, bins=50, density=True, alpha=0.6, label=label,
                   color=color, edgecolor='white', linewidth=0.5)
            color_idx += 1
        
        ax.set_xlabel(prop, fontsize=12, fontweight='bold')
        ax.set_ylabel("Density", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color('#CCCCCC')
        ax.spines["bottom"].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=10)
        
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
        plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def plot_qed(model_frames: Dict[str, pd.DataFrame], out_dir: Path, ref_df: Optional[pd.DataFrame] = None, dataset_label: str = "Reference") -> None:
    """Plot QED distribution for generated molecules (optional reference for comparison)."""
    # Reuse plot_histograms for consistency
    plot_histograms(model_frames, properties=["QED"], out_dir=out_dir, ref_df=ref_df, dataset_label=dataset_label)


def plot_heatmap_comparison(panel_table: pd.DataFrame, out_dir: Path, filename: str = "metrics_heatmap.png") -> None:
    """Create a beautiful heatmap showing all metrics across models and constraint levels."""
    if panel_table.empty:
        print("Warning: No panel table data to plot. Skipping heatmap.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Prepare data for heatmap
    metrics = ['Adherence %', 'Valid %', 'Distinct %', 'Diversity', 'QED']
    available_metrics = [m for m in metrics if m in panel_table.columns]
    
    if not available_metrics:
        print("Warning: No metrics found in panel table. Skipping heatmap.")
        return
    
    # Create model+type+level identifier
    panel_table['ModelType'] = panel_table['Model'].astype(str)
    if 'ConstraintType' in panel_table.columns:
        panel_table['ModelType'] = panel_table['ModelType'] + ' (' + panel_table['ConstraintType'].astype(str) + ')'
    
    # Create a pivot table for each metric
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Metrics Heatmap: Performance Across Models and Constraint Levels', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    level_order = ['loose', 'tight', 'ultra_tight']
    level_labels = {'loose': 'Loose', 'tight': 'Tight', 'ultra_tight': 'Ultra Tight'}
    
    for idx, metric in enumerate(available_metrics[:4]):  # Max 4 metrics
        ax = axes[idx // 2, idx % 2]
        
        # Create pivot table
        pivot_data = panel_table.pivot_table(
            values=metric,
            index='ModelType',
            columns='ConstraintLevel',
            aggfunc='mean'
        )
        
        # Reorder columns
        existing_levels = [l for l in level_order if l in pivot_data.columns]
        pivot_data = pivot_data[existing_levels]
        
        # Rename columns for better display
        pivot_data.columns = [level_labels.get(col, col) for col in pivot_data.columns]
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100 if '%' in metric else 1.0)
        
        # Set ticks and labels
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels(pivot_data.columns, fontsize=10, fontweight='bold')
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels([mt.replace(' (gradual)', '\n(gradual)').replace(' (range-based)', '\n(range)') 
                           for mt in pivot_data.index], fontsize=9)
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.iloc[i, j]
                if not pd.isna(value):
                    text_color = 'white' if value < 50 else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                           color=text_color, fontsize=9, fontweight='bold')
        
        ax.set_title(metric, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Constraint Level', fontsize=11, fontweight='bold')
        if idx % 2 == 0:
            ax.set_ylabel('Model (Constraint Type)', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric, fontsize=9, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(available_metrics), 4):
        axes[idx // 2, idx % 2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_constraint_level_comparison(panel_table: pd.DataFrame, out_dir: Path, filename: str = "constraint_level_comparison.png") -> None:
    """Plot metrics comparison across constraint levels with beautiful styling. Saves each metric as a separate file."""
    if panel_table.empty or "ConstraintLevel" not in panel_table.columns:
        print("Warning: No constraint level data to plot. Skipping constraint level comparison.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Define constraint level order
    level_order = ['loose', 'tight', 'ultra_tight']
    level_labels = {'loose': 'Loose', 'tight': 'Tight', 'ultra_tight': 'Ultra Tight'}
    
    # Color scheme: different colors for each model+type combination
    model_colors = {
        'GPT2-Zinc-87M': '#4C72B0',
        'GPT2-Zinc-87M+SMC (gradual)': '#55A868',
        'GPT2-Zinc-87M+SMC (range-based)': '#64B5CD',
        'SmileyLlama-8B (gradual)': '#F57C00',
        'SmileyLlama-8B (range-based)': '#FFB74D',
    }
    
    # Define metrics to plot with their y-axis ranges
    metrics_to_plot = []
    if 'Adherence %' in panel_table.columns:
        metrics_to_plot.append(('Adherence %', 0, 110))
    if 'Valid %' in panel_table.columns:
        metrics_to_plot.append(('Valid %', 0, 110))
    if 'Distinct %' in panel_table.columns:
        metrics_to_plot.append(('Distinct %', 0, 110))
    if 'Diversity' in panel_table.columns:
        metrics_to_plot.append(('Diversity', 0.8, 1.0))
    if 'QED' in panel_table.columns:
        metrics_to_plot.append(('QED', 0, 1.0))
    
    # Create a separate plot for each metric
    for metric_name, y_min, y_max in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create model+type identifier (create a copy to avoid modifying original)
        plot_data = panel_table.copy()
        plot_data['ModelType'] = plot_data['Model'].astype(str)
        if 'ConstraintType' in plot_data.columns:
            plot_data['ModelType'] = plot_data['ModelType'] + ' (' + plot_data['ConstraintType'].astype(str) + ')'
        
        # Plot each model+type combination
        x_positions = {}
        x_pos = 0
        for model_type in sorted(plot_data['ModelType'].unique()):
            x_positions[model_type] = x_pos
            x_pos += 1
        
        width = 0.18
        level_x_offsets = {'loose': -0.5*width, 'tight': 0.5*width, 'ultra_tight': 1.5*width}
        level_colors = {'loose': '#87CEEB', 'tight': '#FFA500', 'ultra_tight': '#FF6347'}
        
        # Track which levels we've plotted for legend
        legend_added = set()
        
        for model_type in sorted(plot_data['ModelType'].unique()):
            model_data = plot_data[plot_data['ModelType'] == model_type]
            base_x = x_positions[model_type]
            
            for level in level_order:
                level_data = model_data[model_data['ConstraintLevel'] == level]
                if not level_data.empty:
                    value = level_data[metric_name].iloc[0]
                    color = level_colors.get(level, '#999999')
                    offset = level_x_offsets.get(level, 0)
                    
                    label = f'{level_labels[level]}' if level not in legend_added else ''
                    if label:
                        legend_added.add(level)
                    
                    bar = ax.bar(base_x + offset, value, width=width, 
                                color=color, alpha=0.85, edgecolor='white', linewidth=1.5,
                                label=label)
                    
                    # Add value label with appropriate formatting
                    if metric_name in ['Diversity', 'QED']:
                        label_text = f'{value:.3f}'
                    else:
                        label_text = f'{value:.1f}'
                    ax.text(base_x + offset, value + (y_max - y_min) * 0.015,
                           label_text, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([mt.replace(' (gradual)', '\n(gradual)').replace(' (range-based)', '\n(range)') 
                           for mt in sorted(x_positions.keys())], 
                          rotation=0, ha='center', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color('#CCCCCC')
        ax.spines["bottom"].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=10)
        
        # Add legend for constraint levels
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicates while preserving order
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, loc='upper left', fontsize=9, 
                 framealpha=0.95, title='Constraint Level', title_fontsize=10,
                 edgecolor='#CCCCCC', fancybox=True)
        
        ax.set_title(f'{metric_name} Across Constraint Levels', fontsize=14, fontweight='bold', pad=15)
        
        # Save each metric as a separate file
        metric_filename = filename.replace('.png', f'_{metric_name.replace(" ", "_").replace("%", "pct")}.png')
        plt.tight_layout()
        plt.savefig(out_dir / metric_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def plot_bars(metrics: pd.DataFrame, out_dir: Path, filename: str = "model_bars.png") -> None:
    """Plot bar charts for the core constraint-based metrics with improved styling. Saves each metric as a separate file."""
    if metrics.empty:
        print("Warning: No metrics data to plot. Skipping bar chart.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    metrics = metrics.copy()
    
    # Create labels with better formatting
    labels = metrics["Model"].tolist()
    # Truncate long labels and add line breaks if needed
    formatted_labels = []
    for label in labels:
        if len(label) > 20:
            # Try to break at natural points
            if "(" in label:
                parts = label.split("(")
                if len(parts) == 2:
                    formatted_labels.append(f"{parts[0]}\n({parts[1]}")
                else:
                    formatted_labels.append(label[:20] + "...")
            else:
                formatted_labels.append(label[:20] + "...")
        else:
            formatted_labels.append(label)
    
    # Color palette - modern, accessible colors
    colors = {
        "Adherence %": "#2E7D32",  # Green
        "Valid %": "#1976D2",       # Blue
        "Distinct %": "#F57C00",    # Orange
        "Diversity": "#C62828",     # Red
        "QED": "#7B1FA2",           # Purple
    }
    
    # Plot each metric separately
    metric_configs = [
        ("Adherence %", 0, 110, "Percentage (%)", lambda v: f'{v:.1f}%'),
        ("Valid %", 0, 110, "Percentage (%)", lambda v: f'{v:.1f}%'),
        ("Distinct %", 0, 110, "Percentage (%)", lambda v: f'{v:.1f}%'),
        ("Diversity", None, None, "Diversity Score", lambda v: f'{v:.3f}'),
        ("QED", 0, 1.0, "QED Score", lambda v: f'{v:.3f}'),
    ]
    
    for metric_name, y_min, y_max, ylabel, format_func in metric_configs:
        if metric_name not in metrics.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(labels)), metrics[metric_name], 
                      color=colors[metric_name], alpha=0.85, edgecolor='white', linewidth=1.5)
        
        ax.set_title(metric_name, fontweight="bold", fontsize=14, pad=10)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics[metric_name])):
            height = bar.get_height()
            offset = (y_max - y_min) * 0.01 if y_min is not None and y_max is not None else height * 0.01
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    format_func(val), ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Improve appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color('#CCCCCC')
        ax.spines["bottom"].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=10)
        
        # Save each metric as a separate file
        metric_filename = filename.replace('.png', f'_{metric_name.replace(" ", "_").replace("%", "pct")}.png')
        plt.tight_layout()
        plt.savefig(out_dir / metric_filename, dpi=300, bbox_inches='tight', facecolor='white')
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
            # Check ConstraintType to determine type (for combined files or individual files)
            if "ConstraintType" in smc.columns:
                smc_gradual_split = smc[smc["ConstraintType"] == "gradual"]
                smc_range_split = smc[smc["ConstraintType"] == "range-based"]
                if not smc_gradual_split.empty:
                    model_frames["GPT2-Zinc-87M+SMC (gradual)"] = filter_valid_molecules(smc_gradual_split)
                if not smc_range_split.empty:
                    model_frames["GPT2-Zinc-87M+SMC (range-based)"] = filter_valid_molecules(smc_range_split)
            else:
                # Fallback: assume all is gradual if we can't determine
                if not smc.empty:
                    model_frames["GPT2-Zinc-87M+SMC (gradual)"] = filter_valid_molecules(smc)
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
            # Check ConstraintType to determine type
            if "ConstraintType" in smiley.columns:
                smiley_gradual_split = smiley[smiley["ConstraintType"] == "gradual"]
                smiley_range_split = smiley[smiley["ConstraintType"] == "range-based"]
                if not smiley_gradual_split.empty:
                    model_frames["SmileyLlama-8B (gradual)"] = filter_valid_molecules(smiley_gradual_split)
                if not smiley_range_split.empty:
                    model_frames["SmileyLlama-8B (range-based)"] = filter_valid_molecules(smiley_range_split)
            else:
                # Fallback: assume all is gradual if we can't determine
                if not smiley.empty:
                    model_frames["SmileyLlama-8B (gradual)"] = filter_valid_molecules(smiley)
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

    # Load panel table for constraint level comparison
    panel_table_path = Path(results_dir) / "panel_table.csv"
    if panel_table_path.exists():
        panel_table = pd.read_csv(panel_table_path)
        plot_constraint_level_comparison(panel_table, fig_dir, filename="constraint_level_comparison.png")
        # Heatmap plots removed - simplified to focus on key metrics
    
    print(f"\nSaved plots to {fig_dir}/")
    print("  - Constraint level comparison (Adherence, Valid, Distinct, Diversity, QED) - saved separately")


if __name__ == "__main__":
    main()
