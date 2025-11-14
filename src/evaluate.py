"""
Evaluation utilities for GPT2-Zinc baseline vs SmileyLlama experiments.
Focuses on constraint-based generation metrics: Adherence, Validity, Distinctness, Diversity.

Automatically loads individual experiment result files (baseline_*_results.csv, smiley_*_results.csv)
or combined files if available.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Conditional rdkit import for test compatibility
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    RDKIT_AVAILABLE = True
    # Suppress RDKit SMILES parse error messages
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    DataStructs = None
    RDLogger = None

from .utils import (
    ensure_directory,
    load_results_file,
    load_results_by_pattern,
    filter_valid_molecules,
    load_experiment_results,
    count_result_files,
)


def tanimoto_diversity(smiles: Iterable[str]) -> float:
    """Calculate Tanimoto diversity (1 - mean similarity) for SMILES strings."""
    if not RDKIT_AVAILABLE:
        # Return 0.0 if rdkit is not available (can't calculate diversity)
        return 0.0
    
    fps: List[DataStructs.ExplicitBitVect] = []
    for s in smiles:
        # Filter out NaN and non-string values (pandas Series can contain NaN as float)
        if not isinstance(s, str):
            continue
        if pd.isna(s):
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    if len(fps) < 2:
        return 0.0
    sims: List[float] = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
    if not sims:
        return 0.0
    mean_sim = float(np.mean(sims))
    return 1.0 - mean_sim


def summarise_group(df: pd.DataFrame) -> pd.Series:
    """
    Summarize constraint-based generation metrics.
    
    Focuses on core metrics for controlled generation:
    1. Adherence % - Percentage of molecules meeting property constraints
    2. Valid % - Percentage of valid SMILES
    3. Distinct % - Percentage of unique molecules
    4. Diversity - Tanimoto diversity (1 - mean similarity)
    5. QED - Mean QED (Quantitative Estimate of Drug-likeness) for valid molecules
    """
    total = len(df)
    valid_df = filter_valid_molecules(df)
    metrics = {
        "Adherence %": 100.0 * df["Adherence"].mean() if "Adherence" in df else 0.0,
        "Valid %": 100.0 * df["Valid"].mean() if total else 0.0,
        "Distinct %": 100.0 * valid_df["SMILES"].nunique() / total if total else 0.0,
        "Diversity": tanimoto_diversity(valid_df["SMILES"]) if not valid_df.empty else 0.0,
        "QED": valid_df["QED"].mean() if not valid_df.empty and "QED" in valid_df.columns else 0.0,
    }
    return pd.Series(metrics)


def load_runtime_data(results_dir: str = "results") -> pd.DataFrame:
    """
    Load runtime data from all summary files and combine into a single dataframe.
    
    Returns:
        DataFrame with Model, ConstraintLevel, ConstraintType, Prompt, and Runtime columns
    """
    
    summary_files = glob.glob(str(Path(results_dir) / "*_summary.csv"))
    if not summary_files:
        return pd.DataFrame()
    
    runtime_rows = []
    for summary_file in summary_files:
        try:
            summary_df = pd.read_csv(summary_file)
            # Check if runtime columns exist
            if "Runtime_formatted" in summary_df.columns or "Runtime_seconds" in summary_df.columns:
                # Extract model, constraint level, and type from filename
                filename = Path(summary_file).stem
                
                # Determine model from filename
                if "baseline" in filename:
                    model = "GPT2-Zinc-87M"
                    constraint_type = "range-based"
                elif "smc" in filename:
                    model = "GPT2-Zinc-87M+SMC"
                    if "gradual" in filename:
                        constraint_type = "gradual"
                    else:
                        constraint_type = "range-based"
                elif "smiley" in filename:
                    model = "SmileyLlama-8B"
                    if "gradual" in filename:
                        constraint_type = "gradual"
                    else:
                        constraint_type = "range-based"
                else:
                    continue
                
                # Extract constraint level from filename
                # Check longer/more specific strings first to avoid substring matches
                constraint_level = None
                for level in ["ultra_tight", "loosen", "loose", "tight"]:
                    if level in filename:
                        constraint_level = level
                        break
                
                if constraint_level is None:
                    continue
                
                # Get runtime data from summary
                for _, row in summary_df.iterrows():
                    runtime_row = {
                        "Model": model,
                        "ConstraintType": constraint_type,
                        "ConstraintLevel": constraint_level,
                        "Prompt": row.get("Prompt", ""),
                    }
                    
                    # Handle temperature - convert to float if present, or use None
                    temp = row.get("Temperature", None)
                    if temp is not None and pd.notna(temp):
                        try:
                            runtime_row["Temperature"] = float(temp)
                        except (ValueError, TypeError):
                            runtime_row["Temperature"] = None
                    else:
                        runtime_row["Temperature"] = None
                    
                    # Add runtime formatted column if available
                    if "Runtime_formatted" in row and pd.notna(row.get("Runtime_formatted")):
                        runtime_row["Runtime_formatted"] = str(row["Runtime_formatted"])
                    
                    runtime_rows.append(runtime_row)
        except Exception as e:
            # Skip files that can't be read
            continue
    
    if not runtime_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(runtime_rows)


def build_tables(
    combined: pd.DataFrame,
    temperature_filter: Optional[float] = None,
    results_dir: str = "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build evaluation tables focusing on constraint-based metrics.
    No reference dataset needed - only evaluates adherence to constraints.
    """
    df = combined.copy()
    if temperature_filter is not None and "Temperature" in df.columns:
        df = df[np.isclose(df["Temperature"], temperature_filter)]

    # Ensure ConstraintType is set for all rows
    if "ConstraintType" not in df.columns:
        df["ConstraintType"] = None
    
    # Fill missing ConstraintType based on ConstraintLevel
    if "ConstraintLevel" in df.columns:
        mask_missing = df["ConstraintType"].isna() | (df["ConstraintType"] == "")
        if mask_missing.any():
            # Infer from ConstraintLevel: "loosen" = gradual, "loose" = range-based
            df.loc[mask_missing & (df["ConstraintLevel"] == "loosen"), "ConstraintType"] = "gradual"
            df.loc[mask_missing & (df["ConstraintLevel"] == "loose"), "ConstraintType"] = "range-based"
            
            # For "tight" and "ultra_tight", try to infer from Model or Prompt
            # Baseline always uses range-based
            df.loc[mask_missing & (df["Model"] == "GPT2-Zinc-87M"), "ConstraintType"] = "range-based"
            
            # For SMC and SmileyLlama with tight/ultra_tight, check if we have both types
            # If we can't determine, leave as None (will be grouped separately)
            # But try to infer from existing data: if we have "loosen" entries, likely gradual
            # If we have "loose" entries, likely range-based
            for model in ["GPT2-Zinc-87M+SMC", "SmileyLlama-8B"]:
                model_mask = mask_missing & (df["Model"] == model)
                if model_mask.any():
                    # Check if this model has any gradual entries
                    has_gradual = ((df["Model"] == model) & (df["ConstraintType"] == "gradual")).any()
                    has_range = ((df["Model"] == model) & (df["ConstraintType"] == "range-based")).any()
                    
                    # If we have gradual but no range, assume ambiguous ones are gradual
                    if has_gradual and not has_range:
                        df.loc[model_mask, "ConstraintType"] = "gradual"
                    # If we have range but no gradual, assume ambiguous ones are range-based
                    elif has_range and not has_gradual:
                        df.loc[model_mask, "ConstraintType"] = "range-based"

    # Group by Model and ConstraintLevel (if available) to distinguish between gradual and range-based
    group_cols = ["Model"]
    if "ConstraintLevel" in df.columns and df["ConstraintLevel"].nunique() > 1:
        # Don't add ConstraintLevel to group_cols for summary - we want to see model performance across all levels
        # But we can use it to distinguish between gradual (loosen/tight/ultra_tight) and range-based (loose/tight/ultra_tight)
        pass
    if "Temperature" in df.columns and df["Temperature"].nunique() > 1:
        group_cols.append("Temperature")

    summary_rows: List[pd.Series] = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        summary = summarise_group(group)
        for col, value in zip(group_cols, keys):
            summary[col] = value
        summary_rows.append(summary)
    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table[group_cols + ["Adherence %", "Valid %", "Distinct %", "Diversity", "QED"]]

    # Panel table groups by Model, ConstraintType (if available), ConstraintLevel (if available), and Prompt
    panel_rows: List[pd.Series] = []
    panel_group_cols = group_cols.copy()
    
    # Add ConstraintType if available to distinguish gradual vs range-based
    if "ConstraintType" in df.columns and df["ConstraintType"].notna().any():
        panel_group_cols.append("ConstraintType")
    
    if "ConstraintLevel" in df.columns:
        panel_group_cols.append("ConstraintLevel")
    panel_group_cols.append("Prompt")
    
    for keys, group in df.groupby(panel_group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        summary = summarise_group(group)
        for col, value in zip(panel_group_cols, keys):
            summary[col] = value
        panel_rows.append(summary)
    panel_table = pd.DataFrame(panel_rows)
    panel_table = panel_table[panel_group_cols + ["Adherence %", "Valid %", "Distinct %", "Diversity", "QED"]]
    
    # Merge runtime data from summary files
    runtime_df = load_runtime_data(results_dir)
    if not runtime_df.empty:
        # Merge on Model, ConstraintType, ConstraintLevel, and Prompt
        merge_cols = ["Model"]
        if "ConstraintType" in panel_table.columns and "ConstraintType" in runtime_df.columns:
            merge_cols.append("ConstraintType")
        if "ConstraintLevel" in panel_table.columns and "ConstraintLevel" in runtime_df.columns:
            merge_cols.append("ConstraintLevel")
        if "Prompt" in panel_table.columns and "Prompt" in runtime_df.columns:
            merge_cols.append("Prompt")
        if "Temperature" in panel_table.columns and "Temperature" in runtime_df.columns:
            merge_cols.append("Temperature")
        
        # Merge runtime data (only Runtime_formatted)
        # Don't use Temperature as merge key since panel_table doesn't include it
        if "Runtime_formatted" in runtime_df.columns:
            merge_cols_final = [c for c in merge_cols if c != "Temperature"]
            panel_table = panel_table.merge(
                runtime_df[merge_cols_final + ["Runtime_formatted"]],
                on=merge_cols_final,
                how="left"
            )
            
            # Reorder columns to put runtime at the end
            metric_cols = ["Adherence %", "Valid %", "Distinct %", "Diversity", "QED"]
            other_cols = [col for col in panel_table.columns 
                         if col not in panel_group_cols + metric_cols + ["Runtime_formatted"]]
            panel_table = panel_table[panel_group_cols + metric_cols + other_cols + ["Runtime_formatted"]]
    
    return summary_table.sort_values(group_cols), panel_table.sort_values(panel_group_cols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate constraint-based molecule generation outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluates 4 core metrics for controlled generation:
  1. Adherence % - Percentage meeting property constraints
  2. Valid % - Percentage of valid SMILES
  3. Distinct % - Percentage of unique molecules
  4. Diversity - Tanimoto diversity (1 - mean similarity)

No reference dataset needed - only evaluates constraint adherence.

Examples:
  # Evaluate all models
  python -m src.evaluate

  # Filter by temperature
  python -m src.evaluate --temperature 1.0
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
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature filter.")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory for tables")
    parser.add_argument("--summary-table", type=str, default="results/summary_table.csv", help="Summary table output")
    parser.add_argument("--panel-table", type=str, default="results/panel_table.csv", help="Panel table output")
    args = parser.parse_args()

    # Auto-detect result files if not specified
    results_dir = args.results_dir
    
    # Use consolidated loading function
    baseline, smc_gradual, smc_range, smiley_gradual, smiley_range, smc, smiley = load_experiment_results(
        results_dir=results_dir,
        baseline=args.baseline,
        smc=args.smc,
        smiley=args.smiley,
        temperature=args.temperature,
    )
    
    # Combine all results
    if baseline.empty and smc.empty and smiley.empty:
        print("Error: No result files found!")
        print(f"  Looked in: {results_dir}/")
        print("  Expected files:")
        print("    - baseline_results.csv or baseline_*_results.csv")
        print("    - smc_all_results.csv, smc_gradual_results.csv, smc_range_results.csv, or smc_*_results.csv")
        print("    - smiley_all_results.csv, smiley_gradual_results.csv, smiley_range_results.csv, or smiley_*_results.csv")
        return
    
    dfs_to_combine = []
    baseline_count, smc_count, smiley_count = count_result_files(
        results_dir=results_dir,
        baseline=args.baseline,
        smc=args.smc,
        smiley=args.smiley,
    )
    
    if not baseline.empty:
        dfs_to_combine.append(baseline)
        print(f"Loaded {len(baseline)} baseline molecules from {baseline_count} file(s)")
    
    if not smc.empty:
        dfs_to_combine.append(smc)
        print(f"Loaded {len(smc)} SMC molecules from {smc_count} file(s)")
    
    if not smiley.empty:
        dfs_to_combine.append(smiley)
        print(f"Loaded {len(smiley)} SmileyLlama molecules from {smiley_count} file(s)")
    
    combined = pd.concat(dfs_to_combine, ignore_index=True) if dfs_to_combine else pd.DataFrame()

    ensure_directory(Path(args.out_dir).as_posix())
    
    print(f"\n{'='*80}")
    print("Constraint-Based Generation Evaluation")
    print(f"{'='*80}")
    print("Metrics: Adherence %, Valid %, Distinct %, Diversity")
    print(f"Total molecules: {len(combined)}")
    print(f"Valid molecules: {combined['Valid'].sum() if 'Valid' in combined.columns else 'N/A'}")
    
    summary_table, panel_table = build_tables(combined, temperature_filter=args.temperature, results_dir=results_dir)
    
    # Save tables
    summary_path = Path(args.summary_table)
    panel_path = Path(args.panel_table)
    
    summary_table.to_csv(summary_path, index=False)
    panel_table.to_csv(panel_path, index=False)
    
    print(f"\nSummary Table (by Model):")
    print(summary_table.to_string(index=False))
    print(f"\nPanel Table (by Model + Prompt):")
    print(panel_table.to_string(index=False))
    print(f"\nSaved tables to:")
    print(f"  {summary_path}")
    print(f"  {panel_path}")


if __name__ == "__main__":
    main()
