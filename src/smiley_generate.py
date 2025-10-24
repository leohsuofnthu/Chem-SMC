"""
SmileyLlama inference with property evaluation.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from .utils import (
    PROMPTS,
    PROMPT_MAP,
    annotate_adherence,
    all_valid_smiles,
    compute_properties_df,
    ensure_directory,
    summarise_adherence,
)

MODEL_NAME = "THGLab/Llama-3.1-8B-SmileyLlama-1.1"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    device: Optional[str] = None,
    quantize: bool = True,
    low_cpu_mem_usage: bool = True,
    max_memory: Optional[dict] = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
    }
    if device.startswith("cuda"):
        model_kwargs["device_map"] = "auto"
        if max_memory:
            model_kwargs["max_memory"] = max_memory
    if quantize and device.startswith("cuda"):
        try:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception:
        # Fallback to full precision if quantized load fails.
        model_kwargs.pop("quantization_config", None)
        model_kwargs["torch_dtype"] = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return tokenizer, model


def _gather_smiles(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    target_n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    device = next(model.parameters()).device
    collected: List[str] = []
    seen = set()
    
    # Create progress bar for Smiley generation
    pbar = tqdm(
        total=target_n,
        desc="Generating SmileyLlama molecules",
        unit="mol",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    batch_count = 0
    try:
        while len(collected) < target_n:
            batch_count += 1
            prompts = [prompt] * batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            valid_count = 0
            batch_molecules = []
            for text in decoded:
                body = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
                smiles_list = all_valid_smiles(body)
                for smi in smiles_list:
                    if smi not in seen:
                        seen.add(smi)
                        collected.append(smi)
                        batch_molecules.append(smi)
                        valid_count += 1
                        if len(collected) >= target_n:
                            break
                if len(collected) >= target_n:
                    break
            
            # Update progress bar with batch of molecules
            if batch_molecules:
                pbar.update(len(batch_molecules))
            
            # Update progress bar with current status
            pbar.set_postfix({
                "Batch": f"{batch_count}",
                "Valid": f"{valid_count}",
                "Temp": f"{temperature:.1f}",
                "Total": f"{len(collected)}/{target_n}"
            })
    finally:
        pbar.close()
    
    return collected[:target_n]


def generate_for_prompt(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt_name: str,
    n: int = 1_000,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    batch_size: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    _seed_all(seed)
    spec = PROMPT_MAP[prompt_name]
    smiles = _gather_smiles(
        tokenizer,
        model,
        spec.text,
        target_n=n,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    df = compute_properties_df(smiles[:n])
    df = annotate_adherence(df, spec)
    df["Model"] = "SmileyLlama-8B"
    df["Prompt"] = spec.name
    df["Temperature"] = temperature
    df["TopP"] = top_p
    df["Weight"] = 1.0
    return df


def run_experiment(
    prompt_names: Iterable[str],
    out_csv: str,
    summary_csv: str,
    temperatures: Optional[Iterable[float]] = None,
    n: int = 1_000,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    batch_size: int = 50,
    seed: int = 0,
    device: Optional[str] = None,
    quantize: bool = True,
    low_cpu_mem_usage: bool = True,
    max_memory: Optional[dict] = None,
) -> pd.DataFrame:
    tokenizer, model = load_model(
        device=device,
        quantize=quantize,
        low_cpu_mem_usage=low_cpu_mem_usage,
        max_memory=max_memory,
    )
    frames: List[pd.DataFrame] = []
    summaries: List[pd.Series] = []
    temp_list = list(temperatures or [1.0])
    
    # Create progress bar for experiment
    total_experiments = len(prompt_names) * len(temp_list)
    exp_pbar = tqdm(
        total=total_experiments,
        desc="Running SmileyLlama experiments",
        unit="exp",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    try:
        for idx, prompt_name in enumerate(prompt_names):
            for temp in temp_list:
                exp_pbar.set_postfix({"Prompt": prompt_name, "Temp": f"{temp:.1f}"})
                
                df = generate_for_prompt(
                    tokenizer,
                    model,
                    prompt_name,
                    n=n,
                    temperature=temp,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                    seed=seed + idx,
                )
                frames.append(df)
                summary = summarise_adherence(df)
                summary["Prompt"] = prompt_name
                summary["Model"] = "SmileyLlama-8B"
                summary["Temperature"] = temp
                summaries.append(summary)
                
                exp_pbar.update(1)
    finally:
        exp_pbar.close()

    results = pd.concat(frames, ignore_index=True)
    ensure_directory(Path(out_csv).parent.as_posix())
    results.to_csv(out_csv, index=False)

    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SmileyLlama molecular generation.")
    parser.add_argument("--prompts", nargs="+", default=[p.name for p in PROMPTS])
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--out_csv", type=str, default="results/smiley_results.csv")
    parser.add_argument("--summary_csv", type=str, default="results/smiley_summary.csv")
    args = parser.parse_args()

    quantize = args.quantize or not args.no_quantize

    set_seed(args.seed)
    run_experiment(
        prompt_names=args.prompts,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
        temperatures=args.temperatures,
        n=args.n,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        quantize=quantize,
    )


if __name__ == "__main__":
    main()
