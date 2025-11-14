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
    
    level_order = ['loosen', 'loose', 'tight', 'ultra_tight']
    level_labels = {'loosen': 'Loosen', 'loose': 'Loose', 'tight': 'Tight', 'ultra_tight': 'Ultra Tight'}
    
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
    """Plot metrics comparison across constraint levels with beautiful styling."""
    if panel_table.empty or "ConstraintLevel" not in panel_table.columns:
        print("Warning: No constraint level data to plot. Skipping constraint level comparison.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Define constraint level order
    level_order = ['loosen', 'loose', 'tight', 'ultra_tight']
    level_labels = {'loosen': 'Loosen', 'loose': 'Loose', 'tight': 'Tight', 'ultra_tight': 'Ultra Tight'}
    
    # Color scheme: different colors for each model+type combination
    model_colors = {
        'GPT2-Zinc-87M': '#4C72B0',
        'GPT2-Zinc-87M+SMC (gradual)': '#55A868',
        'GPT2-Zinc-87M+SMC (range-based)': '#64B5CD',
        'SmileyLlama-8B (gradual)': '#F57C00',
        'SmileyLlama-8B (range-based)': '#FFB74D',
    }
    
    # Determine layout based on available metrics
    has_diversity = 'Diversity' in panel_table.columns
    has_qed = 'QED' in panel_table.columns
    
    if has_diversity and has_qed:
        # 2x3 layout for all 5 metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metrics_to_plot = [
            ('Adherence %', axes[0, 0], 0, 110),
            ('Valid %', axes[0, 1], 0, 110),
            ('Distinct %', axes[0, 2], 0, 110),
            ('Diversity', axes[1, 0], 0.8, 1.0),
            ('QED', axes[1, 1], 0, 1.0),
        ]
        axes[1, 2].axis('off')
    elif has_diversity or has_qed:
        # 2x2 layout for 4 metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics_to_plot = [
            ('Adherence %', axes[0, 0], 0, 110),
            ('Valid %', axes[0, 1], 0, 110),
            ('Distinct %', axes[1, 0], 0, 110),
        ]
        if has_diversity:
            metrics_to_plot.append(('Diversity', axes[1, 1], 0.8, 1.0))
        elif has_qed:
            metrics_to_plot.append(('QED', axes[1, 1], 0, 1.0))
    else:
        # 2x2 layout for 3 metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics_to_plot = [
            ('Adherence %', axes[0, 0], 0, 110),
            ('Valid %', axes[0, 1], 0, 110),
            ('Distinct %', axes[1, 0], 0, 110),
        ]
        axes[1, 1].axis('off')
    
    fig.suptitle('Performance Metrics Across Constraint Levels', fontsize=16, fontweight='bold', y=0.995)
    
    for metric_name, ax, y_min, y_max in metrics_to_plot:
        if metric_name not in panel_table.columns:
            ax.axis('off')
            continue
        
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
        level_x_offsets = {'loosen': -1.5*width, 'loose': -0.5*width, 'tight': 0.5*width, 'ultra_tight': 1.5*width}
        level_colors = {'loosen': '#90EE90', 'loose': '#87CEEB', 'tight': '#FFA500', 'ultra_tight': '#FF6347'}
        
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
                    
                    # Add value label
                    ax.text(base_x + offset, value + (y_max - y_min) * 0.015,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
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
        
        # Add legend for constraint levels (only on first subplot)
        if metric_name == 'Adherence %':
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
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_bars(metrics: pd.DataFrame, out_dir: Path, filename: str = "model_bars.png") -> None:
    """Plot bar charts for the core constraint-based metrics with improved styling."""
    if metrics.empty:
        print("Warning: No metrics data to plot. Skipping bar chart.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Determine number of subplots based on available metrics
    available_metrics = []
    if "Adherence %" in metrics.columns:
        available_metrics.append("Adherence %")
    if "Valid %" in metrics.columns:
        available_metrics.append("Valid %")
    if "Distinct %" in metrics.columns:
        available_metrics.append("Distinct %")
    if "Diversity" in metrics.columns:
        available_metrics.append("Diversity")
    if "QED" in metrics.columns:
        available_metrics.append("QED")
    
    n_plots = len(available_metrics)
    if n_plots == 0:
        print("Warning: No metrics found to plot. Skipping bar chart.")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), sharey=False)
    if n_plots == 1:
        axes = [axes]  # Make it iterable
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
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
    
    # Plot available metrics in order
    ax_idx = 0
    
    # Adherence % (most important for constraint-based generation)
    if "Adherence %" in metrics.columns:
        bars = axes[ax_idx].bar(range(len(labels)), metrics["Adherence %"], 
                          color=colors["Adherence %"], alpha=0.85, edgecolor='white', linewidth=1.5)
        axes[ax_idx].set_title("Adherence %", fontweight="bold", fontsize=14, pad=10)
        axes[ax_idx].set_ylim(0, 110)
        axes[ax_idx].set_xticks(range(len(labels)))
        axes[ax_idx].set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
        axes[ax_idx].set_ylabel("Percentage (%)", fontsize=11)
        axes[ax_idx].grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics["Adherence %"])):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax_idx += 1
    
    # Valid %
    if "Valid %" in metrics.columns:
        bars = axes[ax_idx].bar(range(len(labels)), metrics["Valid %"], 
                          color=colors["Valid %"], alpha=0.85, edgecolor='white', linewidth=1.5)
        axes[ax_idx].set_title("Validity %", fontweight="bold", fontsize=14, pad=10)
        axes[ax_idx].set_ylim(0, 110)
        axes[ax_idx].set_xticks(range(len(labels)))
        axes[ax_idx].set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
        axes[ax_idx].set_ylabel("Percentage (%)", fontsize=11)
        axes[ax_idx].grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics["Valid %"])):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax_idx += 1
    
    # Distinct %
    if "Distinct %" in metrics.columns:
        bars = axes[ax_idx].bar(range(len(labels)), metrics["Distinct %"], 
                          color=colors["Distinct %"], alpha=0.85, edgecolor='white', linewidth=1.5)
        axes[ax_idx].set_title("Distinct %", fontweight="bold", fontsize=14, pad=10)
        axes[ax_idx].set_ylim(0, 110)
        axes[ax_idx].set_xticks(range(len(labels)))
        axes[ax_idx].set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
        axes[ax_idx].set_ylabel("Percentage (%)", fontsize=11)
        axes[ax_idx].grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics["Distinct %"])):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax_idx += 1

    # Diversity
    if "Diversity" in metrics.columns:
        bars = axes[ax_idx].bar(range(len(labels)), metrics["Diversity"], 
                          color=colors["Diversity"], alpha=0.85, edgecolor='white', linewidth=1.5)
        axes[ax_idx].set_title("Diversity\n(1 - mean Tanimoto)", fontweight="bold", fontsize=14, pad=10)
        axes[ax_idx].set_xticks(range(len(labels)))
        axes[ax_idx].set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
        axes[ax_idx].set_ylabel("Diversity Score", fontsize=11)
        axes[ax_idx].grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics["Diversity"])):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax_idx += 1

    # QED
    if "QED" in metrics.columns:
        bars = axes[ax_idx].bar(range(len(labels)), metrics["QED"], 
                          color=colors["QED"], alpha=0.85, edgecolor='white', linewidth=1.5)
        axes[ax_idx].set_title("QED\n(Quantitative Estimate of Drug-likeness)", fontweight="bold", fontsize=14, pad=10)
        axes[ax_idx].set_ylim(0, 1.0)
        axes[ax_idx].set_xticks(range(len(labels)))
        axes[ax_idx].set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
        axes[ax_idx].set_ylabel("QED Score", fontsize=11)
        axes[ax_idx].grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics["QED"])):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Improve overall appearance
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color('#CCCCCC')
        ax.spines["bottom"].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=10)

    plt.tight_layout(pad=2.0)
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
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
    
    # Load panel table for constraint level comparison
    panel_table_path = Path(results_dir) / "panel_table.csv"
    if panel_table_path.exists():
        panel_table = pd.read_csv(panel_table_path)
        plot_constraint_level_comparison(panel_table, fig_dir, filename="constraint_level_comparison.png")
        plot_heatmap_comparison(panel_table, fig_dir, filename="metrics_heatmap.png")
    
    print(f"\nSaved plots to {fig_dir}/")
    print("  - Property histograms (MW, logP, TPSA)")
    print("  - QED distribution")
    print("  - Model comparison bars (Adherence, Valid, Distinct, Diversity)")
    print("  - Constraint level comparison (across all levels)")
    print("  - Metrics heatmap (visual summary)")


if __name__ == "__main__":
    main()
