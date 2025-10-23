"""
Data preparation pipeline for ZINC-250k.

Loads SMILES, computes RDKit descriptors, writes parquet splits, and records
empirical property ranges for later prompt construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import PROPERTY_COLUMNS, compute_properties_df, percentile_ranges


def _load_zinc_dataframe(input_path: Optional[Path]) -> pd.DataFrame:
    if input_path is None:
        # Default search order within data directory
        candidates = [
            Path("data/zinc250k.csv"),
            Path("data/zinc250k.smi"),
            Path("data/zinc250k.tsv"),
            Path("data/zinc.csv"),
        ]
        for cand in candidates:
            if cand.exists():
                input_path = cand
                break
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            "ZINC-250k file not found. Place a CSV/SMI/TSV with a 'smiles' column under data/ "
            "or supply --input <path>."
        )

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix in {".csv", ".smi"}:
        df = pd.read_csv(input_path)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(input_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    smiles_col = None
    for col in df.columns:
        if col.lower() in {"smiles", "smile", "smiles_string"}:
            smiles_col = col
            break
    if smiles_col is None:
        raise KeyError("Input file must contain a SMILES column.")
    df = df[[smiles_col]].rename(columns={smiles_col: "SMILES"})
    return df


def prepare_data(input_path: Optional[str], seed: int, ref_size: int) -> pd.DataFrame:
    df = _load_zinc_dataframe(Path(input_path) if input_path else None)
    if len(df) < ref_size + 1:
        raise ValueError(f"Dataset has {len(df)} rows, need at least {ref_size + 1}.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)
    df_props = compute_properties_df(df["SMILES"])
    return df_props


def run_data_prep(
    input_path: Optional[str],
    output_dir: str,
    ranges_path: str,
    seed: int = 42,
    ref_size: int = 240_000,
) -> None:
    df = prepare_data(input_path, seed=seed, ref_size=ref_size)

    ref = df.iloc[:ref_size].reset_index(drop=True)
    eval_df = df.iloc[ref_size : ref_size + 10_000].reset_index(drop=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ref_path = Path(output_dir) / "zinc_ref.parquet"
    eval_path = Path(output_dir) / "zinc_eval.parquet"
    ref.to_parquet(ref_path, index=False)
    eval_df.to_parquet(eval_path, index=False)

    ranges = percentile_ranges(ref, PROPERTY_COLUMNS)
    with open(ranges_path, "w", encoding="utf-8") as fp:
        json.dump(ranges, fp, indent=2)

    print("Prepared datasets:")
    print(f"  Reference: {ref_path} ({len(ref)} molecules)")
    print(f"  Eval:      {eval_path} ({len(eval_df)} molecules)")
    print("Property ranges (5th-95th percentile):")
    for key, (low, high) in ranges.items():
        print(f"  {key:>4s}: {low:.2f} – {high:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ZINC splits and compute property ranges.")
    parser.add_argument("--input", type=str, default=None, help="Optional path to ZINC-250k file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-size", type=int, default=240_000)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--ranges-path", type=str, default="data/zinc_property_ranges.json")
    args = parser.parse_args()
    run_data_prep(args.input, args.output_dir, args.ranges_path, seed=args.seed, ref_size=args.ref_size)


if __name__ == "__main__":
    main()
