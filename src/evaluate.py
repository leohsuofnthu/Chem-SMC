"""
Evaluation utilities for GPT2-Zinc+SMC vs SmileyLlama experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from .utils import PROMPTS, PROMPT_MAP, ensure_directory


PROPS_FOR_KL = ["MW", "logP", "RotB", "TPSA"]


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


def kl_divergence(sample: np.ndarray, reference: np.ndarray, bins: int = 50) -> float:
    if len(sample) < 2 or len(reference) < 2:
        return float("nan")
    combined = np.concatenate([sample, reference])
    lo, hi = combined.min(), combined.max()
    if np.isclose(lo, hi):
        return 0.0
    hist_s, bin_edges = np.histogram(sample, bins=bins, range=(lo, hi), density=False)
    hist_r, _ = np.histogram(reference, bins=bin_edges, density=False)
    hist_s = hist_s.astype(float) + 1e-8
    hist_r = hist_r.astype(float) + 1e-8
    hist_s /= hist_s.sum()
    hist_r /= hist_r.sum()
    return float(np.sum(hist_s * np.log(hist_s / hist_r)))


def property_kl(sample_df: pd.DataFrame, ref_df: pd.DataFrame) -> float:
    values: List[float] = []
    for prop in PROPS_FOR_KL:
        if prop not in sample_df or prop not in ref_df:
            continue
        sample = sample_df[prop].dropna().to_numpy()
        reference = ref_df[prop].dropna().to_numpy()
        if len(sample) == 0 or len(reference) == 0:
            continue
        values.append(kl_divergence(sample, reference))
    return float(np.nan) if not values else float(np.nanmean(values))


def summarise_group(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.Series:
    total = len(df)
    valid_df = _valid_df(df)
    metrics = {
        "QED": valid_df["QED"].mean(skipna=True),
        "Valid %": 100.0 * df["Valid"].mean() if total else 0.0,
        "Distinct %": 100.0 * valid_df["SMILES"].nunique() / total if total else 0.0,
        "Adherence %": 100.0 * df["Adherence"].mean() if "Adherence" in df else 0.0,
        "Diversity": tanimoto_diversity(valid_df["SMILES"]) if not valid_df.empty else 0.0,
        "KL": property_kl(valid_df, ref_df),
    }
    return pd.Series(metrics)


def build_tables(
    combined: pd.DataFrame,
    ref_df: pd.DataFrame,
    temperature_filter: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = combined.copy()
    if temperature_filter is not None and "Temperature" in df.columns:
        df = df[np.isclose(df["Temperature"], temperature_filter)]

    group_cols = ["Model"]
    if "Temperature" in df.columns and df["Temperature"].nunique() > 1:
        group_cols.append("Temperature")

    summary_rows: List[pd.Series] = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        summary = summarise_group(group, ref_df)
        for col, value in zip(group_cols, keys):
            summary[col] = value
        summary_rows.append(summary)
    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table[group_cols + ["QED", "Valid %", "Distinct %", "Diversity", "KL"]]

    panel_rows: List[pd.Series] = []
    panel_group_cols = group_cols + ["Prompt"]
    for keys, group in df.groupby(panel_group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        summary = summarise_group(group, ref_df)
        for col, value in zip(panel_group_cols, keys):
            summary[col] = value
        panel_rows.append(summary)
    panel_table = pd.DataFrame(panel_rows)
    panel_table = panel_table[panel_group_cols + ["Adherence %", "Valid %", "Distinct %"]]
    return summary_table.sort_values(group_cols), panel_table.sort_values(panel_group_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate molecule generation outputs.")
    parser.add_argument("--baseline", type=str, default="results/baseline_results.csv")
    parser.add_argument("--smc", type=str, default="results/smc_results.csv")
    parser.add_argument("--smiley", type=str, default="results/smiley_results.csv")
    parser.add_argument("--reference", type=str, default="data/zinc_ref.parquet")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature filter.")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--qed_table", type=str, default="results/qed_table.csv")
    parser.add_argument("--panel_table", type=str, default="results/panel_table.csv")
    args = parser.parse_args()

    baseline = _load_results(args.baseline)
    smc = _load_results(args.smc)
    smiley = _load_results(args.smiley)
    combined = pd.concat([baseline, smc, smiley], ignore_index=True)

    ref_df = pd.read_parquet(args.reference)
    ref_df = ref_df[ref_df["Valid"]].copy() if "Valid" in ref_df.columns else ref_df

    summary_table, panel_table = build_tables(combined, ref_df, temperature_filter=args.temperature)
    ensure_directory(Path(args.out_dir).as_posix())
    summary_table.to_csv(args.qed_table, index=False)
    panel_table.to_csv(args.panel_table, index=False)

    print("Table 1 – ZINC-like (QED / Validity / Diversity / KL)")
    print(summary_table.to_string(index=False))
    print("\nTable 2 – Property-range adherence (panel)")
    print(panel_table.to_string(index=False))


if __name__ == "__main__":
    main()
