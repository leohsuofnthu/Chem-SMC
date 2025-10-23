"""
Plotting utilities for ChemGPT+SMC vs SmileyLlama experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import summarise_group
from .utils import ensure_directory


def _load(path: str, temperature: Optional[float]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if temperature is not None and "Temperature" in df.columns:
        df = df[np.isclose(df["Temperature"], temperature)]
    return df


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Valid"]].copy() if "Valid" in df.columns else df


def plot_histograms(
    ref_df: pd.DataFrame,
    model_frames: Dict[str, pd.DataFrame],
    properties: Iterable[str],
    out_dir: Path,
) -> None:
    for prop in properties:
        plt.figure(figsize=(6, 4))
        ref_vals = ref_df[prop].dropna()
        if not ref_vals.empty:
            plt.hist(ref_vals, bins=50, density=True, alpha=0.35, label="ZINC reference", color="#4C72B0")
        for label, frame in model_frames.items():
            vals = frame[prop].dropna()
            if vals.empty:
                continue
            plt.hist(vals, bins=50, density=True, alpha=0.35, label=label)
        plt.xlabel(prop)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prop.lower()}_histogram.png", dpi=200)
        plt.close()


def plot_qed(ref_df: pd.DataFrame, model_frames: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    ref_vals = ref_df["QED"].dropna()
    if not ref_vals.empty:
        plt.hist(ref_vals, bins=50, density=True, alpha=0.35, label="ZINC reference", color="#4C72B0")
    for label, frame in model_frames.items():
        vals = frame["QED"].dropna()
        if vals.empty:
            continue
        plt.hist(vals, bins=50, density=True, alpha=0.35, label=label)
    plt.xlabel("QED")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "qed_distribution.png", dpi=200)
    plt.close()


def plot_bars(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    metrics = metrics.copy()
    labels = metrics["Model"].tolist()
    axes[0].bar(labels, metrics["Valid %"])
    axes[0].set_title("Validity %")
    axes[0].set_ylim(0, 110)
    axes[0].tick_params(axis="x", rotation=20)

    if "Adherence %" in metrics.columns:
        axes[1].bar(labels, metrics["Adherence %"], color="#55A868")
        axes[1].set_title("Adherence %")
        axes[1].set_ylim(0, 110)
        axes[1].tick_params(axis="x", rotation=20)
    else:
        axes[1].axis("off")

    axes[2].bar(labels, metrics["Diversity"], color="#C44E52")
    axes[2].set_title("Diversity (1 - mean Tanimoto)")
    axes[2].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_dir / "model_bars.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation figures.")
    parser.add_argument("--baseline", type=str, default="results/baseline_results.csv")
    parser.add_argument("--smc", type=str, default="results/smc_results.csv")
    parser.add_argument("--smiley", type=str, default="results/smiley_results.csv")
    parser.add_argument("--reference", type=str, default="data/zinc_ref.parquet")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--figures", type=str, default="figures")
    args = parser.parse_args()

    baseline = _load(args.baseline, args.temperature)
    smc = _load(args.smc, args.temperature)
    smiley = _load(args.smiley, args.temperature)
    combined = pd.concat([baseline, smc, smiley], ignore_index=True)

    ref_df = pd.read_parquet(args.reference)
    ref_valid = _valid(ref_df)

    model_frames = {
        "ChemGPT baseline": _valid(baseline),
        "ChemGPT+SMC": _valid(smc),
        "SmileyLlama": _valid(smiley),
    }

    ensure_directory(args.figures)
    fig_dir = Path(args.figures)

    plot_histograms(ref_valid, model_frames, properties=["MW", "logP", "TPSA"], out_dir=fig_dir)
    plot_qed(ref_valid, model_frames, out_dir=fig_dir)

    summary_rows = []
    for label, frame in model_frames.items():
        summary = summarise_group(frame, ref_valid)
        summary["Model"] = label
        summary_rows.append(summary)
    metrics = pd.DataFrame(summary_rows)
    plot_bars(metrics, fig_dir)


if __name__ == "__main__":
    main()
