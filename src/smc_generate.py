"""
Sequential Monte Carlo generation using GenLM's AWRS controller.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from transformers import set_seed

from .utils import (
    PROMPTS,
    PROMPT_MAP,
    PromptSpec,
    annotate_adherence,
    check_constraints,
    compute_properties,
    compute_properties_df,
    ensure_directory,
    summarise_adherence,
    first_valid_smiles,
)

try:
    from genlm.control import AWRS, PromptedLLM

    GENLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GENLM_AVAILABLE = False

DEFAULT_MODEL_NAME = "entropy/gpt2_zinc_87M"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_potential(prompt: PromptSpec):
    """Create GenLM potential with optimized ZINC-like reward function."""

    def potential(seq: bytes) -> float:
        decoded = seq.decode("utf-8", errors="ignore")
        smiles = first_valid_smiles(decoded)
        if not smiles:
            return 0.001

        # Encourage concise sequences that primarily contain SMILES.
        if len(decoded.strip()) > len(smiles) + 20:
            return 0.001

        props = compute_properties(smiles)
        if props is None:
            return 0.001

        mw = props.get("MW", 0)
        logp = props.get("logP", 0)
        rotb = props.get("RotB", 0)

        # Hard cutoff for completely invalid molecules
        if mw > 600 or logp > 6 or rotb > 15:
            return 0.001

        # Optimized ZINC-like scoring (same as our previous implementation)
        import math
        
        # MW scoring: Gaussian-like with steeper falloff
        mw_center = 350  # Target center
        mw_penalty = abs(mw - mw_center) / 50  # Normalized distance
        mw_score = math.exp(-mw_penalty**2)  # Gaussian-like
        
        # LogP scoring: More sensitive to deviations
        logp_center = 2.0  # Target center
        logp_penalty = abs(logp - logp_center) / 1.0  # Normalized distance
        logp_score = math.exp(-logp_penalty**2)  # Gaussian-like
        
        # Rotatable bonds: Exponential penalty for high values
        rotb_score = math.exp(-max(0, rotb - 5) / 2.0)  # Penalty starts at 5 rotb
        
        # Ring bonus: Encourage ring structures (drug-like)
        mol = Chem.MolFromSmiles(smiles)
        rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        ring_bonus = 1.0 + 0.2 * min(rings, 4)  # Reduced bonus to avoid over-weighting
        
        # Size bonus: Penalize very small molecules
        size_bonus = 1.0 + 0.1 * min(mw / 100, 3)  # Bonus for larger molecules
        
        # Combine scores with weights
        base_score = mw_score * logp_score * rotb_score
        final_score = base_score * ring_bonus * size_bonus
        
        # Scale to reasonable range
        return float(max(final_score * 5.0, 0.001))

    return potential


@dataclass
class GenLMSMCSampler:
    model_name: str = DEFAULT_MODEL_NAME
    particles: int = 10
    ess_threshold: float = 0.5
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if not GENLM_AVAILABLE:
            raise ImportError(
                "genlm-control is required for SMC generation. Install with `pip install genlm-control`."
            )
        llm_kwargs = {
            "temperature": self.temperature,
        }
        sampler_kwargs = {"top_p": self.top_p}
        try:
            llm_kwargs["sampler_kwargs"] = sampler_kwargs
            self.llm = PromptedLLM.from_name(self.model_name, **llm_kwargs)
        except TypeError:
            # Older genlm-control versions may not support sampler_kwargs.
            llm_kwargs.pop("sampler_kwargs", None)
            self.llm = PromptedLLM.from_name(self.model_name, **llm_kwargs)

    async def _sample_once(self, prompt: PromptSpec) -> Dict[str, float]:
        potential = _make_potential(prompt)
        self.llm.set_prompt_from_str(prompt.text.strip())
        sampler = AWRS(self.llm, potential)
        sequences = await sampler.smc(
            n_particles=self.particles,
            ess_threshold=self.ess_threshold,
            max_tokens=self.max_new_tokens,
            verbosity=0,
        )
        posterior = getattr(sequences, "decoded_posterior", {})
        return {str(k): float(v) for k, v in posterior.items()}

    def sample(self, prompt: PromptSpec, n: int, max_batches: Optional[int] = None) -> Dict[str, float]:
        collected: Dict[str, float] = {}
        batches = 0
        target_batches = max_batches or max(4, math.ceil(n / max(self.particles, 1)))
        while len(collected) < n and batches < target_batches:
            posterior = asyncio.run(self._sample_once(prompt))
            batches += 1
            if not posterior:
                continue
            for sequence, weight in sorted(posterior.items(), key=lambda x: x[1], reverse=True):
                if sequence.startswith(prompt.text):
                    candidate = sequence[len(prompt.text) :].strip()
                else:
                    candidate = sequence.strip()
                smiles = first_valid_smiles(candidate)
                if not smiles:
                    continue
                if smiles not in collected:
                    collected[smiles] = weight
                if len(collected) >= n:
                    break
        return collected


def _samples_to_dataframe(samples: Dict[str, float]) -> pd.DataFrame:
    smiles = list(samples.keys())
    df = compute_properties_df(smiles)
    df["Weight"] = df["SMILES"].map(samples).astype(float)
    cols = [
        "SMILES",
        "Valid",
        "QED",
        "MW",
        "logP",
        "RotB",
        "TPSA",
        "HBD",
        "HBA",
        "Weight",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def generate_for_prompt(
    prompt_name: str,
    n: int = 1_000,
    temperature: float = 1.0,
    top_p: float = 0.9,
    particles: int = 10,
    ess_threshold: float = 0.5,
    max_new_tokens: int = 128,
    seed: int = 0,
) -> pd.DataFrame:
    _seed_everything(seed)
    spec = PROMPT_MAP[prompt_name]
    sampler = GenLMSMCSampler(
        model_name=DEFAULT_MODEL_NAME,
        particles=particles,
        ess_threshold=ess_threshold,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    collected = sampler.sample(spec, n=n)
    df = _samples_to_dataframe(collected)
    df = annotate_adherence(df, spec)
    df["Model"] = "GPT2-Zinc-87M+SMC"
    df["Prompt"] = spec.name
    df["Temperature"] = temperature
    df["TopP"] = top_p
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
        if col not in df.columns:
            df[col] = pd.NA
    df = df[columns]
    return df


def run_experiment(
    prompt_names: Iterable[str],
    out_csv: str,
    summary_csv: str,
    n: int = 1_000,
    temperatures: Optional[Iterable[float]] = None,
    top_p: float = 0.9,
    particles: int = 10,
    ess_threshold: float = 0.5,
    max_new_tokens: int = 128,
    seed: int = 0,
) -> pd.DataFrame:
    temperatures = list(temperatures or [1.0])
    frames: List[pd.DataFrame] = []
    summaries: List[pd.Series] = []
    for prompt_name in prompt_names:
        for temp in temperatures:
            df = generate_for_prompt(
                prompt_name,
                n=n,
                temperature=temp,
                top_p=top_p,
                particles=particles,
                ess_threshold=ess_threshold,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            frames.append(df)
            summary = summarise_adherence(df)
            summary["Prompt"] = prompt_name
            summary["Model"] = "GPT2-Zinc-87M+SMC"
            summary["Temperature"] = temp
            summaries.append(summary)

    if not frames:
        raise RuntimeError("No SMC samples were generated.")

    results = pd.concat(frames, ignore_index=True)
    ensure_directory(Path(out_csv).parent.as_posix())
    results.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_csv, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GenLM-based SMC generation.")
    parser.add_argument("--prompts", nargs="+", default=[p.name for p in PROMPTS])
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="results/smc_results.csv")
    parser.add_argument("--summary_csv", type=str, default="results/smc_summary.csv")
    args = parser.parse_args()

    if not GENLM_AVAILABLE:
        raise SystemExit("genlm-control must be installed to run SMC generation.")

    set_seed(args.seed)
    run_experiment(
        prompt_names=args.prompts,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
        n=args.n,
        temperatures=args.temperatures,
        top_p=args.top_p,
        particles=args.particles,
        ess_threshold=args.ess_threshold,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
