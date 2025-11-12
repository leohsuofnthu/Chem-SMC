"""
Constraint-aware SmileyLlama generation using constraint variants.
Runs experiments with loose, tight, and ultra_tight constraints.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from .smiley_generate import (
    load_model,
    generate_for_prompt,
    _format_smiley_prompt,
    SMILEY_SYSTEM_MSG,
)
from .utils import (
    SMILEY_PROMPTS,
    SMILEY_PROMPT_MAP,
    PromptSpec,
    load_property_ranges,
    create_constraint_variant,
    ensure_directory,
    summarise_adherence,
    compute_properties_df,
    annotate_adherence,
)


def run_constraint_experiment(
    constraint_level: str = "loose",  # "loose", "tight", "ultra_tight"
    property_ranges_path: str = "data/train_property_ranges.json",
    dataset: str = "Combined",
    base_prompt_name: str = "mw_logp_rotb",  # Use this prompt as base
    n: int = 1_000,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    batch_size: int = 50,
    seed: int = 0,
    device: Optional[str] = None,
    quantize: bool = True,
    out_csv: str = "results/smiley_results.csv",
    summary_csv: str = "results/smiley_summary.csv",
) -> pd.DataFrame:
    """
    Run a single constraint-level experiment with SmileyLlama.
    Uses constraint variants to modify the prompt constraints.
    """
    start_time = time.time()
    
    # Load constraint ranges for this level
    constraint_ranges = load_property_ranges(property_ranges_path, dataset, constraint_level)
    
    # Get base prompt
    base_prompt = SMILEY_PROMPT_MAP[base_prompt_name]
    
    # Create constraint variant for ALL levels (including loose) to ensure prompt text matches evaluation constraints
    constraint_prompt = create_constraint_variant(base_prompt, constraint_ranges, tightness=constraint_level)
    prompt_variant_name = f"{base_prompt_name}_{constraint_level}" if constraint_level != "loose" else base_prompt_name
    
    # Load model
    tokenizer, model = load_model(device=device, quantize=quantize)
    
    # Format prompt for SmileyLlama
    formatted_prompt = _format_smiley_prompt(constraint_prompt.text)
    
    # Generate molecules
    from .smiley_generate import _gather_smiles, _seed_all
    _seed_all(seed)
    
    smiles = _gather_smiles(
        tokenizer,
        model,
        formatted_prompt,
        target_n=n,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    
    # Compute properties and annotate adherence
    df = compute_properties_df(smiles[:n])
    df = annotate_adherence(df, constraint_prompt)
    df["Model"] = "SmileyLlama-8B"
    df["Prompt"] = prompt_variant_name
    df["ConstraintLevel"] = constraint_level
    df["Temperature"] = temperature
    df["TopP"] = top_p
    df["Weight"] = 1.0
    
    # Save results
    ensure_directory(Path(out_csv).parent.as_posix())
    columns = [
        "SMILES", "Valid", "QED", "MW", "logP", "RotB", "TPSA", "HBD", "HBA",
        "Adherence", "Weight", "Model", "Prompt", "ConstraintLevel",
        "Temperature", "TopP",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[columns]
    df.to_csv(out_csv, index=False)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Create summary
    summary = summarise_adherence(df)
    summary["ConstraintLevel"] = constraint_level
    summary["Model"] = "SmileyLlama-8B"
    summary["Prompt"] = prompt_variant_name
    summary["Temperature"] = temperature
    summary["Runtime_seconds"] = elapsed_time
    summary["Runtime_minutes"] = elapsed_time / 60.0
    summary["Runtime_formatted"] = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_csv, index=False)
    
    # Print timing info
    print(f"  Completed in {summary['Runtime_formatted']} ({elapsed_time:.1f} seconds)")
    
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint-aware SmileyLlama generation.")
    parser.add_argument("--constraint-level", type=str, default="loose", choices=["loose", "tight", "ultra_tight"])
    parser.add_argument("--property-ranges", type=str, default="data/train_property_ranges.json")
    parser.add_argument("--dataset", type=str, default="Combined", choices=["ZINC", "ChEMBL", "Combined"])
    parser.add_argument("--base-prompt", type=str, default="mw_logp_rotb", choices=[p.name for p in SMILEY_PROMPTS])
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--out-csv", type=str, default="results/smiley_results.csv")
    parser.add_argument("--summary-csv", type=str, default="results/smiley_summary.csv")
    args = parser.parse_args()

    quantize_flag = args.quantize or not args.no_quantize

    run_constraint_experiment(
        constraint_level=args.constraint_level,
        property_ranges_path=args.property_ranges,
        dataset=args.dataset,
        base_prompt_name=args.base_prompt,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        quantize=quantize_flag,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
    )


if __name__ == "__main__":
    main()

