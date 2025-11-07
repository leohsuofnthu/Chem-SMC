"""
Constraint-aware baseline generation with GPT2-Zinc.
Samples from all available prefixes and filters by constraint adherence.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm

from .utils import (
    GPT_ZINC_PROMPTS,
    PromptSpec,
    annotate_adherence,
    compute_properties_df,
    ensure_directory,
    load_property_ranges,
    create_constraint_variant,
    summarise_adherence,
)

MODEL_NAME = "entropy/gpt2_zinc_87M"
_MODEL_CACHE: Dict[tuple, tuple] = {}


def _load_model(device: str = "cpu") -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    key = (device, str(torch_dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set padding side to left for decoder-only models (fixes generation warning)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _decode_generations(
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    generated_ids: torch.Tensor,
) -> List[str]:
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned: List[str] = []
    for text, prompt in zip(decoded, prompt_texts):
        if text.startswith(prompt):
            text = text[len(prompt) :]
        cleaned.append(text.strip())
    return cleaned


def generate_with_multi_prefix(
    prefix_prompts: List[str],
    constraint_spec: Optional[object] = None,  # PromptSpec with constraints
    n: int = 1_000,
    T: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    batch_size: int = 64,
    device: Optional[str] = None,
    seed: int = 42,
    filter_by_constraints: bool = True,
) -> pd.DataFrame:
    """
    Generate molecules by sampling from multiple prefixes randomly.
    
    Args:
        prefix_prompts: List of prefix strings to sample from
        constraint_spec: PromptSpec with constraints for filtering (optional)
        n: Target number of molecules to generate
        filter_by_constraints: If True, only keep molecules meeting constraints
        ... other args same as before
    """
    set_seed(seed)
    random.seed(seed)
    tokenizer, model = _load_model(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device

    outputs: List[str] = []
    prefix_used: List[str] = []  # Track which prefix was used for each output
    
    # Generate more if we're filtering (expect ~30-50% to pass constraints)
    target_generate = n * 3 if filter_by_constraints else n
    
    # Create progress bar
    pbar = tqdm(
        total=n,
        desc="Generating with multi-prefix",
        unit="mol",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    try:
        while len(outputs) < target_generate:
            current_batch = min(batch_size, target_generate - len(outputs))
            # Randomly sample prefixes for this batch
            batch_prompts = [random.choice(prefix_prompts) for _ in range(current_batch)]
            
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    temperature=T,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoded = _decode_generations(tokenizer, batch_prompts, generated)
            
            # Store outputs with corresponding prefixes
            for text, prefix in zip(decoded, batch_prompts):
                outputs.append(text)
                prefix_used.append(prefix)
            
            # If filtering, check adherence periodically and update progress
            if filter_by_constraints and constraint_spec and len(outputs) >= batch_size:
                # Check every batch_size molecules
                temp_df = compute_properties_df(outputs)
                temp_df = annotate_adherence(temp_df, constraint_spec)
                valid_count = temp_df["Adherence"].sum() if "Adherence" in temp_df.columns else 0
                pbar.n = min(valid_count, n)
                pbar.refresh()
                pbar.set_postfix({
                    "Valid": f"{valid_count}",
                    "Generated": f"{len(outputs)}",
                    "Temp": f"{T:.1f}"
                })
                
                # Break early if we have enough valid molecules
                if valid_count >= n:
                    break
            else:
                pbar.update(len(decoded))
                pbar.set_postfix({
                    "Generated": f"{len(outputs)}",
                    "Temp": f"{T:.1f}"
                })
    
    finally:
        pbar.close()

    # Compute properties and filter
    df = compute_properties_df(outputs)
    df["Prefix"] = prefix_used[:len(df)]
    
    # Filter by constraints if requested
    if filter_by_constraints and constraint_spec:
        df = annotate_adherence(df, constraint_spec)
        # Keep only molecules meeting constraints
        df = df[df["Adherence"] == True].head(n).copy()
        if len(df) < n:
            print(f"Warning: Only {len(df)}/{n} molecules met constraints. Consider increasing generation target.")
    
    df["Temperature"] = T
    df["TopP"] = top_p
    df["Weight"] = 1.0
    df["Model"] = "GPT2-Zinc-87M"
    return df


def run_constraint_experiment(
    constraint_level: str = "loose",  # "loose", "tight", "ultra_tight"
    property_ranges_path: str = "data/train_property_ranges.json",
    dataset: str = "Combined",
    n: int = 1_000,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    batch_size: int = 64,
    device: Optional[str] = None,
    seed: int = 42,
    out_csv: str = "results/baseline_results.csv",
    summary_csv: str = "results/baseline_summary.csv",
) -> pd.DataFrame:
    """
    Run a single constraint-level experiment using all available prefixes.
    """
    start_time = time.time()
    
    # Load constraint ranges for this level
    constraint_ranges = load_property_ranges(property_ranges_path, dataset, constraint_level)
    
    # Create a constraint spec using one of the base prompts as template
    # We'll use the first prompt as template, then override with constraint ranges
    base_prompt = GPT_ZINC_PROMPTS[0]
    constraint_spec = create_constraint_variant(base_prompt, constraint_ranges, tightness=constraint_level)
    # Create a new spec just for constraint checking (prefix doesn't matter)
    constraint_spec = PromptSpec(
        name=f"multi_prefix_{constraint_level}",
        text="",  # Prefix doesn't matter for constraint checking
        constraints=constraint_spec.constraints,
    )
    
    # Get all prefix texts
    all_prefixes = [p.text for p in GPT_ZINC_PROMPTS]
    
    # Generate with multi-prefix
    df = generate_with_multi_prefix(
        prefix_prompts=all_prefixes,
        constraint_spec=constraint_spec,
        n=n,
        T=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        seed=seed,
        filter_by_constraints=True,
    )
    
    # Add metadata
    df["ConstraintLevel"] = constraint_level
    df["Prompt"] = "multi_prefix"  # All prefixes used
    
    # Save results
    ensure_directory(Path(out_csv).parent.as_posix())
    columns = [
        "SMILES", "Valid", "QED", "MW", "logP", "RotB", "TPSA", "HBD", "HBA",
        "Adherence", "Weight", "Model", "Prompt", "ConstraintLevel",
        "Temperature", "TopP", "Prefix",
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
    summary["Model"] = "GPT2-Zinc-87M"
    summary["Prompt"] = "multi_prefix"
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
    parser = argparse.ArgumentParser(description="Constraint-aware baseline GPT2-Zinc generation.")
    parser.add_argument("--constraint-level", type=str, default="loose", choices=["loose", "tight", "ultra_tight"])
    parser.add_argument("--property-ranges", type=str, default="data/train_property_ranges.json")
    parser.add_argument("--dataset", type=str, default="Combined", choices=["ZINC", "ChEMBL", "Combined"])
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="results/baseline_results.csv")
    parser.add_argument("--summary-csv", type=str, default="results/baseline_summary.csv")
    args = parser.parse_args()

    run_constraint_experiment(
        constraint_level=args.constraint_level,
        property_ranges_path=args.property_ranges,
        dataset=args.dataset,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
    )


if __name__ == "__main__":
    main()

