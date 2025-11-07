"""
Evaluation utilities for GPT2-Zinc+SMC vs SmileyLlama experiments.
Focuses on constraint-based generation metrics: Adherence, Validity, Distinctness, Diversity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

# Suppress RDKit SMILES parse error messages
RDLogger.DisableLog("rdApp.*")


def _load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Valid" not in df.columns:
        df["Valid"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    return df


def _valid_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Valid"]].copy()


def tanimoto_diversity(smiles: Iterable[str]) -> float:
    fps: List[DataStructs.ExplicitBitVect] = []
    for s in smiles:
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
    parser.add_argument("--baseline", type=str, default="results/baseline_results.csv")
    parser.add_argument("--smc", type=str, default="results/smc_results.csv")
    parser.add_argument("--smiley", type=str, default="results/smiley_results.csv")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature filter.")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory for tables")
    parser.add_argument("--summary-table", type=str, default="results/summary_table.csv", help="Summary table output")
    parser.add_argument("--panel-table", type=str, default="results/panel_table.csv", help="Panel table output")
    args = parser.parse_args()

    # Load generation results (SMC is optional - not in simplified pipeline)
    baseline = _load_results(args.baseline)
    smiley = _load_results(args.smiley)
    
    # Try to load SMC if it exists (optional)
    if Path(args.smc).exists():
        smc = _load_results(args.smc)
        combined = pd.concat([baseline, smc, smiley], ignore_index=True)
    else:
        # SMC results not available (simplified pipeline doesn't include it)
        combined = pd.concat([baseline, smiley], ignore_index=True)

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
