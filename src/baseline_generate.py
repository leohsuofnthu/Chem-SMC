"""
Baseline generation with ChemGPT (entropy/gpt2_zinc_87M).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .utils import (
    PROMPTS,
    PROMPT_MAP,
    annotate_adherence,
    compute_properties_df,
    ensure_directory,
    summarise_adherence,
)

MODEL_NAME = "entropy/gpt2_zinc_87M"
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def _load_model(device: str = "cpu") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    key = (device, str(torch_dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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


def generate_baseline(
    prompt: str,
    n: int = 1_000,
    T: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    batch_size: int = 64,
    device: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    set_seed(seed)
    tokenizer, model = _load_model(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device

    outputs: List[str] = []
    while len(outputs) < n:
        current_batch = min(batch_size, n - len(outputs))
        prompts = [prompt] * current_batch
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
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
        decoded = _decode_generations(tokenizer, prompts, generated)
        outputs.extend(decoded)

    df = compute_properties_df(outputs[:n])
    df["Temperature"] = T
    df["TopP"] = top_p
    df["Weight"] = 1.0
    return df


def _attach_metadata(df: pd.DataFrame, prompt_name: str, model_name: str) -> pd.DataFrame:
    df = df.copy()
    df["Model"] = model_name
    df["Prompt"] = prompt_name
    return df


def generate_for_prompt(prompt_name: str, **kwargs) -> pd.DataFrame:
    spec = PROMPT_MAP[prompt_name]
    df = generate_baseline(spec.text, **kwargs)
    df = annotate_adherence(df, spec)
    df = _attach_metadata(df, spec.name, "ChemGPT-87M")
    return df


def run_experiment(
    prompt_names: Iterable[str],
    out_csv: str,
    summary_csv: str,
    temperatures: Optional[Iterable[float]] = None,
    **gen_kwargs,
) -> pd.DataFrame:
    frames = []
    summaries = []
    default_temp = gen_kwargs.pop("T", 1.0)
    base_kwargs = gen_kwargs
    for name in prompt_names:
        temps = list(temperatures or [default_temp])
        for temp in temps:
            frame = generate_for_prompt(name, T=temp, **base_kwargs)
            frames.append(frame)
            summary = summarise_adherence(frame)
            summary["Prompt"] = name
            summary["Model"] = "ChemGPT-87M"
            summary["Temperature"] = temp
            summaries.append(summary)

    results = pd.concat(frames, ignore_index=True)
    columns = [
        "SMILES",
        "Valid",
        "QED",
        "MW",
        "logP",
        "RotB",
        "TPSA",
        "HBD",
        "HBA",
        "Adherence",
        "Weight",
        "Model",
        "Prompt",
        "Temperature",
        "TopP",
    ]
    for col in columns:
        if col not in results.columns:
            results[col] = pd.NA
    results = results[columns]
    ensure_directory(Path(out_csv).parent.as_posix())
    results.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_csv, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline ChemGPT generation.")
    parser.add_argument("--prompts", type=str, nargs="+", default=[p.name for p in PROMPTS])
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="results/baseline_results.csv")
    parser.add_argument("--summary_csv", type=str, default="results/baseline_summary.csv")
    args = parser.parse_args()

    run_experiment(
        prompt_names=args.prompts,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
        n=args.n,
        T=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
