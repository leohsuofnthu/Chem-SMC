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
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

from .utils import ensure_directory

# Suppress RDKit SMILES parse error messages
RDLogger.DisableLog("rdApp.*")


def _load_results(path: str) -> pd.DataFrame:
    """Load a single results CSV file."""
    df = pd.read_csv(path)
    if "Valid" not in df.columns:
        df["Valid"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    return df


def _load_results_by_pattern(pattern: str) -> pd.DataFrame:
    """
    Load all results files matching a glob pattern and combine them.
    
    Args:
        pattern: Glob pattern to match result files (e.g., 'results/baseline_*_results.csv')
    
    Returns:
        Combined DataFrame, or empty DataFrame if no files found.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        df = _load_results(f)
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def _valid_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Valid"]].copy()


def tanimoto_diversity(smiles: Iterable[str]) -> float:
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
    
    Focuses on 4 core metrics for controlled generation:
    1. Adherence % - Percentage of molecules meeting property constraints
    2. Valid % - Percentage of valid SMILES
    3. Distinct % - Percentage of unique molecules
    4. Diversity - Tanimoto diversity (1 - mean similarity)
    """
    total = len(df)
    valid_df = _valid_df(df)
    metrics = {
        "Adherence %": 100.0 * df["Adherence"].mean() if "Adherence" in df else 0.0,
        "Valid %": 100.0 * df["Valid"].mean() if total else 0.0,
        "Distinct %": 100.0 * valid_df["SMILES"].nunique() / total if total else 0.0,
        "Diversity": tanimoto_diversity(valid_df["SMILES"]) if not valid_df.empty else 0.0,
    }
    return pd.Series(metrics)


def build_tables(
    combined: pd.DataFrame,
    temperature_filter: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build evaluation tables focusing on constraint-based metrics.
    No reference dataset needed - only evaluates adherence to constraints.
    """
    df = combined.copy()
    if temperature_filter is not None and "Temperature" in df.columns:
        df = df[np.isclose(df["Temperature"], temperature_filter)]

    group_cols = ["Model"]
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
    summary_table = summary_table[group_cols + ["Adherence %", "Valid %", "Distinct %", "Diversity"]]

    panel_rows: List[pd.Series] = []
    panel_group_cols = group_cols + ["Prompt"]
    for keys, group in df.groupby(panel_group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        summary = summarise_group(group)
        for col, value in zip(panel_group_cols, keys):
            summary[col] = value
        panel_rows.append(summary)
    panel_table = pd.DataFrame(panel_rows)
    panel_table = panel_table[panel_group_cols + ["Adherence %", "Valid %", "Distinct %", "Diversity"]]
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
    
    if args.baseline is None:
        # Try combined file first, then individual files
        baseline_combined = Path(results_dir) / "baseline_results.csv"
        if baseline_combined.exists():
            baseline = _load_results(str(baseline_combined))
        else:
            baseline = _load_results_by_pattern(f"{results_dir}/baseline_*_results.csv")
    else:
        if "*" in args.baseline:
            baseline = _load_results_by_pattern(args.baseline)
        else:
            baseline = _load_results(args.baseline)
    
    if args.smc is None:
        # Try combined file first, then individual files
        smc_combined = Path(results_dir) / "smc_results.csv"
        if smc_combined.exists():
            smc = _load_results(str(smc_combined))
        else:
            smc = _load_results_by_pattern(f"{results_dir}/smc_*_results.csv")
    else:
        if "*" in args.smc:
            smc = _load_results_by_pattern(args.smc)
        else:
            smc = _load_results(args.smc)
    
    if args.smiley is None:
        # Try combined file first, then individual files
        smiley_combined = Path(results_dir) / "smiley_results.csv"
        if smiley_combined.exists():
            smiley = _load_results(str(smiley_combined))
        else:
            smiley = _load_results_by_pattern(f"{results_dir}/smiley_*_results.csv")
    else:
        if "*" in args.smiley:
            smiley = _load_results_by_pattern(args.smiley)
        else:
            smiley = _load_results(args.smiley)
    
    # Combine all results
    if baseline.empty and smc.empty and smiley.empty:
        print("Error: No result files found!")
        print(f"  Looked in: {results_dir}/")
        print("  Expected files: baseline_*_results.csv, smc_*_results.csv, or smiley_*_results.csv")
        return
    
    dfs_to_combine = []
    if not baseline.empty:
        dfs_to_combine.append(baseline)
        if args.baseline is None:
            baseline_files = glob.glob(f"{results_dir}/baseline_*_results.csv")
            file_count = len(baseline_files) if baseline_files else 0
        else:
            file_count = 1
        print(f"Loaded {len(baseline)} baseline molecules from {file_count} file(s)")
    
    if not smc.empty:
        dfs_to_combine.append(smc)
        if args.smc is None:
            smc_files = glob.glob(f"{results_dir}/smc_*_results.csv")
            file_count = len(smc_files) if smc_files else 0
        else:
            file_count = 1
        print(f"Loaded {len(smc)} SMC molecules from {file_count} file(s)")
    
    if not smiley.empty:
        dfs_to_combine.append(smiley)
        if args.smiley is None:
            smiley_files = glob.glob(f"{results_dir}/smiley_*_results.csv")
            file_count = len(smiley_files) if smiley_files else 0
        else:
            file_count = 1
        print(f"Loaded {len(smiley)} SmileyLlama molecules from {file_count} file(s)")
    
    combined = pd.concat(dfs_to_combine, ignore_index=True) if dfs_to_combine else pd.DataFrame()

    ensure_directory(Path(args.out_dir).as_posix())
    
    print(f"\n{'='*80}")
    print("Constraint-Based Generation Evaluation")
    print(f"{'='*80}")
    print("Metrics: Adherence %, Valid %, Distinct %, Diversity")
    print(f"Total molecules: {len(combined)}")
    print(f"Valid molecules: {combined['Valid'].sum() if 'Valid' in combined.columns else 'N/A'}")
    
    summary_table, panel_table = build_tables(combined, temperature_filter=args.temperature)
    
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
