"""
Analyze and visualize property distributions in training data from ZINC and ChEMBL datasets.

This script loads train.csv files from both datasets, computes molecular properties,
and creates visualizations to help identify property ranges for LLM testing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import compute_properties_df, percentile_ranges, PROPERTY_COLUMNS, ensure_directory, compute_percentile_constraints


def load_zinc_train(path: str) -> pd.DataFrame:
    """Load ZINC train.csv and normalize column names."""
    # Use chunksize for large files with progress bar
    chunk_list = []
    chunk_size = 10000
    
    # Read file in chunks with progress bar
    pbar = tqdm(desc="  Loading ZINC", unit=" rows", ncols=80, unit_scale=True)
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            chunk_list.append(chunk)
            pbar.update(len(chunk))
    finally:
        pbar.close()
    
    df = pd.concat(chunk_list, ignore_index=True)
    
    # Normalize column names: mw -> MW, logp -> logP, rotb -> RotB
    df = df.rename(columns={
        "mw": "MW",
        "logp": "logP",
        "rotb": "RotB",
    })
    # Ensure SMILES column is named correctly
    if "smiles" in df.columns:
        df = df.rename(columns={"smiles": "SMILES"})
    return df


def load_chembl_train(path: str) -> pd.DataFrame:
    """Load ChEMBL train.csv (only has SMILES)."""
    # Use chunksize for large files with progress bar
    chunk_list = []
    chunk_size = 10000
    
    # Read file in chunks with progress bar
    pbar = tqdm(desc="  Loading ChEMBL", unit=" rows", ncols=80, unit_scale=True)
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            chunk_list.append(chunk)
            pbar.update(len(chunk))
    finally:
        pbar.close()
    
    df = pd.concat(chunk_list, ignore_index=True)
    
    # Normalize SMILES column name
    if "smiles" in df.columns:
        df = df.rename(columns={"smiles": "SMILES"})
    return df


def compute_properties_df_with_progress(smiles_list: list[str], desc: str = "Computing properties") -> pd.DataFrame:
    """Compute properties with progress bar."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    from rdkit import RDLogger
    
    # Suppress RDKit SMILES parse error messages
    RDLogger.DisableLog("rdApp.*")
    
    PROPERTY_FNS = {
        "MW": Descriptors.MolWt,
        "logP": Crippen.MolLogP,
        "RotB": Lipinski.NumRotatableBonds,
        "HBD": Lipinski.NumHDonors,
        "HBA": Lipinski.NumHAcceptors,
        "TPSA": Descriptors.TPSA,
        "QED": QED.qed,
        "Fsp3": rdMolDescriptors.CalcFractionCSP3,
    }
    
    PROPERTY_COLUMNS = ["MW", "logP", "RotB", "HBD", "HBA", "TPSA", "QED"]
    
    records = []
    for s in tqdm(smiles_list, desc=desc, unit=" mols", ncols=80):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            records.append(
                {
                    "SMILES": s,
                    "Valid": False,
                    **{k: np.nan for k in PROPERTY_COLUMNS},
                    "Fsp3": np.nan,
                }
            )
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        record = {"SMILES": canonical, "Valid": True}
        for key, fn in PROPERTY_FNS.items():
            record[key] = fn(mol)
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_missing_properties(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Compute missing properties for a dataset."""
    # Check which properties need to be computed
    existing_props = set(df.columns) & set(PROPERTY_COLUMNS + ["RotB"])
    missing_props = set(PROPERTY_COLUMNS + ["RotB"]) - existing_props
    
    if not missing_props:
        # All properties exist, just need to validate SMILES and compute Valid flag
        print(f"[{dataset_name}] All properties already exist. Validating SMILES...")
        props_df = compute_properties_df_with_progress(df["SMILES"].tolist(), f"  [{dataset_name}] Validating")
        # Merge with existing properties
        for prop in existing_props:
            if prop in df.columns:
                props_df[prop] = df[prop].values
        return props_df
    
    print(f"[{dataset_name}] Computing missing properties: {missing_props}")
    # Compute all properties from SMILES with progress bar
    props_df = compute_properties_df_with_progress(df["SMILES"].tolist(), f"  [{dataset_name}] Computing")
    
    # If some properties already exist, use them instead of recomputed ones
    for prop in existing_props:
        if prop in df.columns:
            props_df[prop] = df[prop].values
    
    return props_df


def plot_property_distributions(
    zinc_df: pd.DataFrame,
    chembl_df: pd.DataFrame,
    properties: list[str],
    out_dir: Path,
) -> None:
    """Create distribution plots comparing ZINC and ChEMBL datasets."""
    # Filter to valid molecules only
    zinc_valid = zinc_df[zinc_df["Valid"]].copy() if "Valid" in zinc_df.columns else zinc_df
    chembl_valid = chembl_df[chembl_df["Valid"]].copy() if "Valid" in chembl_df.columns else chembl_df
    
    # Create a 2x2 subplot for the 4 properties
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = {"ZINC": "#4C72B0", "ChEMBL": "#55A868"}
    
    for idx, prop in enumerate(properties):
        ax = axes[idx]
        
        # Get data for both datasets
        zinc_vals = zinc_valid[prop].dropna()
        chembl_vals = chembl_valid[prop].dropna()
        
        if len(zinc_vals) > 0:
            ax.hist(
                zinc_vals,
                bins=50,
                density=True,
                alpha=0.6,
                label=f"ZINC (n={len(zinc_vals):,})",
                color=colors["ZINC"],
                edgecolor="black",
                linewidth=0.5,
            )
        
        if len(chembl_vals) > 0:
            ax.hist(
                chembl_vals,
                bins=50,
                density=True,
                alpha=0.6,
                label=f"ChEMBL (n={len(chembl_vals):,})",
                color=colors["ChEMBL"],
                edgecolor="black",
                linewidth=0.5,
            )
        
        ax.set_xlabel(prop, fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{prop} Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_dir / "train_data_distributions.png", dpi=300, bbox_inches="tight")
    print(f"Saved distribution plot to {out_dir / 'train_data_distributions.png'}")
    plt.close()


def print_summary_statistics(
    zinc_df: pd.DataFrame,
    chembl_df: pd.DataFrame,
    properties: list[str],
) -> None:
    """Print summary statistics for both datasets."""
    zinc_valid = zinc_df[zinc_df["Valid"]].copy() if "Valid" in zinc_df.columns else zinc_df
    chembl_valid = chembl_df[chembl_df["Valid"]].copy() if "Valid" in chembl_df.columns else chembl_df
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for prop in properties:
        print(f"\n{prop}:")
        print("-" * 80)
        
        zinc_vals = zinc_valid[prop].dropna()
        chembl_vals = chembl_valid[prop].dropna()
        
        if len(zinc_vals) > 0:
            print(f"  ZINC (n={len(zinc_vals):,}):")
            print(f"    Min:    {zinc_vals.min():.2f}")
            print(f"    5th %:  {zinc_vals.quantile(0.05):.2f}")
            print(f"    25th %: {zinc_vals.quantile(0.25):.2f}")
            print(f"    Median: {zinc_vals.median():.2f}")
            print(f"    Mean:   {zinc_vals.mean():.2f}")
            print(f"    75th %: {zinc_vals.quantile(0.75):.2f}")
            print(f"    95th %: {zinc_vals.quantile(0.95):.2f}")
            print(f"    Max:    {zinc_vals.max():.2f}")
        
        if len(chembl_vals) > 0:
            print(f"  ChEMBL (n={len(chembl_vals):,}):")
            print(f"    Min:    {chembl_vals.min():.2f}")
            print(f"    5th %:  {chembl_vals.quantile(0.05):.2f}")
            print(f"    25th %: {chembl_vals.quantile(0.25):.2f}")
            print(f"    Median: {chembl_vals.median():.2f}")
            print(f"    Mean:   {chembl_vals.mean():.2f}")
            print(f"    75th %: {chembl_vals.quantile(0.75):.2f}")
            print(f"    95th %: {chembl_vals.quantile(0.95):.2f}")
            print(f"    Max:    {chembl_vals.max():.2f}")


def save_property_ranges(
    zinc_df: pd.DataFrame,
    chembl_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save property ranges for all constraint levels (loose, tight, ultra-tight) for all datasets.
    
    This function computes and saves:
    - Loose: 5th-95th percentile
    - Tight: 25th-75th percentile
    - Ultra-tight: 40th-60th percentile
    """
    
    zinc_valid = zinc_df[zinc_df["Valid"]].copy() if "Valid" in zinc_df.columns else zinc_df
    chembl_valid = chembl_df[chembl_df["Valid"]].copy() if "Valid" in chembl_df.columns else chembl_df
    
    # Properties to analyze (focusing on the 4 key ones: MW, logP, RB, QED)
    target_props = ["MW", "logP", "RotB", "QED"]
    
    # Combine datasets for combined ranges
    combined_valid = pd.concat([zinc_valid, chembl_valid], ignore_index=True)
    
    # Define constraint levels
    constraint_levels = {
        "loose": (0.05, 0.95),      # 5th-95th percentile
        "tight": (0.25, 0.75),      # 25th-75th percentile
        "ultra_tight": (0.40, 0.60), # 40th-60th percentile
    }
    
    # Compute ranges for each dataset and constraint level
    output_dict = {}
    
    for dataset_name, df in [("ZINC", zinc_valid), ("ChEMBL", chembl_valid), ("Combined", combined_valid)]:
        dataset_ranges = {}
        
        for level_name, (q_low, q_high) in constraint_levels.items():
            level_ranges = compute_percentile_constraints(df, target_props, q_low, q_high)
            # Convert to list format for JSON
            dataset_ranges[level_name] = {
                prop: [float(low), float(high)]
                for prop, (low, high) in level_ranges.items()
            }
        
        output_dict[dataset_name] = dataset_ranges
    
    # Also save a flat structure for backward compatibility (just loose ranges)
    ranges_dict_flat = {
        "ZINC": output_dict["ZINC"]["loose"],
        "ChEMBL": output_dict["ChEMBL"]["loose"],
        "Combined": output_dict["Combined"]["loose"],
    }
    
    # Save the full structure with all constraint levels
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nSaved property ranges (all constraint levels) to {output_path}")
    print("\nProperty Ranges by Constraint Level:")
    print("=" * 80)
    
    for dataset_name in ["ZINC", "ChEMBL", "Combined"]:
        print(f"\n{dataset_name}:")
        for level_name in ["loose", "tight", "ultra_tight"]:
            print(f"  {level_name}:")
            ranges = output_dict[dataset_name][level_name]
            for prop, (low, high) in ranges.items():
                print(f"    {prop:>4s}: {low:>8.2f} - {high:>8.2f}")
    
    # Also save flat version for backward compatibility
    flat_output_path = output_path.parent / f"{output_path.stem}_flat.json"
    with open(flat_output_path, "w", encoding="utf-8") as f:
        json.dump(ranges_dict_flat, f, indent=2)
    print(f"\nAlso saved flat structure (loose only) to {flat_output_path} for backward compatibility")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze property distributions in ZINC and ChEMBL training data."
    )
    parser.add_argument(
        "--zinc-train",
        type=str,
        default="data/zinc250k/train.csv",
        help="Path to ZINC train.csv",
    )
    parser.add_argument(
        "--chembl-train",
        type=str,
        default="data/chembl/train.csv",
        help="Path to ChEMBL train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--ranges-json",
        type=str,
        default="data/train_property_ranges.json",
        help="Path to save property ranges JSON",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for ChEMBL (for faster processing). If None, use all data.",
    )
    args = parser.parse_args()
    
    # Load datasets
    print("\n" + "=" * 80)
    print("STEP 1: Loading Training Data")
    print("=" * 80)
    print("\nLoading ZINC training data...")
    zinc_df = load_zinc_train(args.zinc_train)
    print(f"\n  ✓ Loaded {len(zinc_df):,} molecules from ZINC")
    
    print("\nLoading ChEMBL training data...")
    chembl_df = load_chembl_train(args.chembl_train)
    if args.sample_size and len(chembl_df) > args.sample_size:
        print(f"\n  Sampling {args.sample_size:,} molecules from {len(chembl_df):,} total...")
        chembl_df = chembl_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    print(f"\n  ✓ Loaded {len(chembl_df):,} molecules from ChEMBL")
    
    # Compute missing properties
    print("\n" + "=" * 80)
    print("STEP 2: Computing Molecular Properties")
    print("=" * 80)
    print("\nComputing properties for ZINC...")
    zinc_df = compute_missing_properties(zinc_df, "ZINC")
    
    print("\nComputing properties for ChEMBL...")
    chembl_df = compute_missing_properties(chembl_df, "ChEMBL")
    
    # Print validity statistics
    if "Valid" in zinc_df.columns:
        zinc_valid_pct = 100.0 * zinc_df["Valid"].sum() / len(zinc_df)
        print(f"\nZINC: {zinc_df['Valid'].sum():,}/{len(zinc_df):,} valid molecules ({zinc_valid_pct:.1f}%)")
    
    if "Valid" in chembl_df.columns:
        chembl_valid_pct = 100.0 * chembl_df["Valid"].sum() / len(chembl_df)
        print(f"ChEMBL: {chembl_df['Valid'].sum():,}/{len(chembl_df):,} valid molecules ({chembl_valid_pct:.1f}%)")
    
    # Create output directory
    ensure_directory(args.output_dir)
    output_dir = Path(args.output_dir)
    
    # Properties to visualize
    properties = ["MW", "logP", "RotB", "QED"]
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("STEP 3: Generating Visualizations")
    print("=" * 80)
    print("\nGenerating distribution plots...")
    with tqdm(total=1, desc="  Creating plots", unit=" plot", ncols=80) as pbar:
        plot_property_distributions(zinc_df, chembl_df, properties, output_dir)
        pbar.update(1)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("STEP 4: Computing Statistics")
    print("=" * 80)
    print_summary_statistics(zinc_df, chembl_df, properties)
    
    # Save property ranges
    print("\n" + "=" * 80)
    print("STEP 5: Saving Results")
    print("=" * 80)
    ensure_directory(Path(args.ranges_json).parent)
    save_property_ranges(zinc_df, chembl_df, Path(args.ranges_json))
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

