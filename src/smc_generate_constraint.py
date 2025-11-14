"""
SMC-guided generation with GPT2-Zinc using GenLM Control.
Uses Sequential Monte Carlo to guide generation based on constraint satisfaction.
Follows the same pattern as baseline_generate_constraint.py for consistency.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from transformers import set_seed
from tqdm import tqdm

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

from .utils import (
    GPT_ZINC_PROMPTS,
    GPT_ZINC_PROMPT_MAP,
    PromptSpec,
    annotate_adherence,
    check_constraints,
    compute_properties,
    compute_properties_df,
    ensure_directory,
    load_property_ranges,
    create_constraint_variant,
    create_gradual_constraint_prompt,
    summarise_adherence,
    first_valid_smiles,
    PROPERTY_FNS,
)

try:
    from genlm.control import AWRS, PromptedLLM, Potential
    GENLM_AVAILABLE = True
except ImportError:
    GENLM_AVAILABLE = False

DEFAULT_MODEL_NAME = "entropy/gpt2_zinc_87M"


# ============================================================
# Constraint-based Molecular Potential (adapts to constraint levels)
# ============================================================

class ConstraintBasedMolecularConstraint:
    """
    Reward function that adapts to different constraint levels (loose/tight/ultra_tight).
    Uses the constraint_spec to dynamically score molecules based on property constraints.
    Reward scales with the number of constraints: more constraints = higher reward for satisfaction.
    """
    def __init__(self, constraint_spec: PromptSpec):
        self.constraint_spec = constraint_spec
        self.constraints = constraint_spec.constraints
        # Count number of active constraints (non-None bounds)
        self.num_constraints = sum(
            1 for lower, upper in self.constraints.values()
            if lower is not None or upper is not None
        )
        # Base reward scales with number of constraints
        # 1 constraint: base 10.0, 2 constraints: base 15.0, 3+ constraints: base 20.0
        self.base_reward = 5.0 + (self.num_constraints * 5.0)
        
    def __call__(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.001
        
        # Compute properties using the existing utils
        props = compute_properties(smiles)
        if props is None:
            return 0.001
        
        # Check if all constraints are satisfied
        satisfied = check_constraints(props, self.constraints)
        
        if satisfied:
            # Perfect satisfaction - return high score that scales with constraint count
            # More constraints satisfied = higher reward
            try:
                # Ring bonus (encourages more complex molecules)
                rings = Chem.rdMolDescriptors.CalcNumRings(mol)
                ring_bonus = 1.0 + 0.2 * min(rings, 3)
                
                # Scale reward with number of constraints
                reward = self.base_reward * ring_bonus
                return float(max(reward, 0.001))
            except:
                return float(self.base_reward)
        
        # Partial satisfaction - compute soft score
        # Count how many constraints are satisfied
        satisfied_count = 0
        total_constraints = 0
        score = 0.0
        total_weight = 0.0
        
        for prop_name, (lower, upper) in self.constraints.items():
            if prop_name not in props:
                continue
                
            value = props.get(prop_name)
            if value is None or np.isnan(value):
                continue
            
            total_constraints += 1
            weight = 1.0
            total_weight += weight
            
            # Check if within bounds
            constraint_satisfied = False
            if lower is not None and upper is not None:
                # Range constraint
                center = (lower + upper) / 2.0
                width = (upper - lower) / 2.0
                if width > 0:
                    penalty = abs(value - center) / width
                    constraint_score = math.exp(-penalty**2)
                    score += weight * constraint_score
                    if constraint_score > 0.9:  # Consider satisfied if very close
                        constraint_satisfied = True
                else:
                    if lower <= value <= upper:
                        score += weight
                        constraint_satisfied = True
            elif lower is not None:
                # Lower bound only
                if value >= lower:
                    score += weight
                    constraint_satisfied = True
                else:
                    penalty = max(0, 1.0 - (lower - value) / (abs(lower) + 1e-6))
                    score += weight * penalty * 0.3
            elif upper is not None:
                # Upper bound only
                if value <= upper:
                    score += weight
                    constraint_satisfied = True
                else:
                    penalty = max(0, 1.0 - (value - upper) / (upper + 1e-6))
                    score += weight * penalty * 0.3
            
            if constraint_satisfied:
                satisfied_count += 1
        
        if total_weight == 0:
            return 0.001
        
        # Normalize and scale
        normalized_score = score / total_weight
        
        # Scale reward based on:
        # 1. How many constraints are satisfied (more = better)
        # 2. How close to satisfying all constraints
        constraint_bonus = 1.0 + (satisfied_count / max(total_constraints, 1)) * 0.5
        scaled_score = normalized_score * (self.base_reward * 0.5) * constraint_bonus
        
        return float(max(scaled_score, 0.001))


# ============================================================
# SMILES sanitization
# ============================================================

def _extract_partial_smiles(text: str) -> str:
    """Extract the longest potential SMILES substring from text."""
    import re
    pattern = r"[A-Za-z0-9@+\-\[\]\(\)=#\\/]+"
    matches = re.findall(pattern, text)
    if matches:
        return max(matches, key=len)
    return ""


def _is_valid_partial_smiles(smiles: str) -> bool:
    """Check if a partial SMILES string is potentially valid."""
    if not smiles:
        return False
    
    # Basic checks for partial SMILES validity
    paren_count = smiles.count('(') - smiles.count(')')
    bracket_count = smiles.count('[') - smiles.count(']')
    
    # Allow some imbalance for partial strings
    if abs(paren_count) > 2 or abs(bracket_count) > 2:
        return False
    
    # Check for obviously invalid patterns
    invalid_patterns = ['@@', '##', '++', '--', '==']
    for pattern in invalid_patterns:
        if pattern in smiles:
            return False
    
    return True


# ============================================================
# GenLM Potential Classes
# ============================================================

class MolecularPotential(Potential):
    """GenLM Potential for molecular generation with constraint-based scoring."""
    
    def __init__(self, constraint_spec: PromptSpec):
        self.constraint = ConstraintBasedMolecularConstraint(constraint_spec)
        # Initialize with vocabulary of bytes (0-255) as required by GenLM
        super().__init__(vocabulary=list(range(256)))
    
    async def prefix(self, context: bytes) -> float:
        """Score partial sequences during generation."""
        decoded = context.decode("utf-8", errors="ignore")
        
        # Progressive validation: check if we're building a valid SMILES
        smiles = first_valid_smiles(decoded)
        if not smiles:
            # Check if we're in the middle of a potentially valid SMILES
            partial_smiles = _extract_partial_smiles(decoded)
            if partial_smiles and _is_valid_partial_smiles(partial_smiles):
                # Progressive reward based on partial SMILES quality
                length_bonus = min(len(partial_smiles) / 20.0, 1.0)
                return 0.1 + 0.1 * length_bonus + random.uniform(0, 0.05)
            return 0.001
        
        # Encourage concise sequences
        if len(decoded.strip()) > len(smiles) + 20:
            return 0.001
        
        # Use constraint-based scoring for complete valid SMILES
        return self.constraint(smiles)
    
    async def complete(self, context: bytes) -> float:
        """Score complete sequences."""
        return await self.prefix(context)
    
    async def batch_prefix(self, contexts):
        """Batch version of prefix for performance."""
        return [await self.prefix(ctx) for ctx in contexts]
    
    async def batch_complete(self, contexts):
        """Batch version of complete for performance."""
        return [await self.complete(ctx) for ctx in contexts]


class SMILESConstraintPotential(Potential):
    """Boolean constraint potential for SMILES validity - works with AWRS."""
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))
    
    async def prefix(self, context: bytes) -> float:
        decoded = context.decode("utf-8", errors="ignore")
        # Encourage continuation even when not yet valid
        if len(decoded) < 15:
            return 0.0
        smiles = first_valid_smiles(decoded)
        if smiles:
            return 0.0
        partial_smiles = _extract_partial_smiles(decoded)
        if partial_smiles and _is_valid_partial_smiles(partial_smiles):
            return 0.0
        return -np.inf
    
    async def complete(self, context: bytes) -> float:
        decoded = context.decode("utf-8", errors="ignore")
        smiles = first_valid_smiles(decoded)
        if smiles is not None:
            return 0.0
        else:
            return -np.inf
    
    async def batch_prefix(self, contexts):
        return [await self.prefix(ctx) for ctx in contexts]
    
    async def batch_complete(self, contexts):
        return [await self.complete(ctx) for ctx in contexts]


# ============================================================
# SMC Sampler
# ============================================================

@dataclass
class GenLMSMCSampler:
    model_name: str = DEFAULT_MODEL_NAME
    particles: int = 50
    ess_threshold: float = 0.3
    max_new_tokens: int = 60
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    
    def __post_init__(self) -> None:
        if not GENLM_AVAILABLE:
            raise ImportError(
                "genlm-control is required for SMC generation. Install with `pip install genlm-control`."
            )
        try:
            self.llm = PromptedLLM.from_name(self.model_name, temperature=self.temperature)
        except Exception as e:
            self.llm = PromptedLLM.from_name(self.model_name)
    
    async def _sample_once(self, prompt: PromptSpec, constraint_spec: PromptSpec) -> Dict[str, float]:
        """
        Run a single Sequential Monte Carlo batch using GenLM's AWRS controller.
        """
        # 1. Create the boolean SMILES constraint potential
        constraint_potential = SMILESConstraintPotential()
        self.llm.set_prompt_from_str(prompt.text.strip())
        
        # 2. Coerce the constraint potential to work with the LLM's token type
        try:
            coerced_constraint = constraint_potential.coerce(self.llm, f=b"".join)
        except Exception as e:
            print(f"⚠️ Constraint coercion failed ({e}); using direct constraint.")
            coerced_constraint = constraint_potential
        
        # 3. Create the scoring potential as a critic (uses constraint_spec)
        scoring_potential = MolecularPotential(constraint_spec)
        try:
            coerced_scoring = scoring_potential.coerce(self.llm, f=b"".join)
        except Exception as e:
            print(f"⚠️ Scoring coercion failed ({e}); using direct scoring.")
            coerced_scoring = scoring_potential
        
        # 4. Create AWRS sampler with boolean constraint
        try:
            sampler = AWRS(self.llm, coerced_constraint)
        except Exception as e:
            print(f"⚠️ AWRS constraint failed ({e}); falling back to direct sampling.")
            from genlm.control import direct_token_sampler
            sampler = direct_token_sampler(self.llm)
        
        # 5. CUDA-safe inference mode
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 6. Run SMC sampling with critic
        sequences = await sampler.smc(
            n_particles=self.particles,
            ess_threshold=self.ess_threshold,
            max_tokens=self.max_new_tokens,
            verbosity=0,
            critic=coerced_scoring,
        )
        
        # 7. Collect posterior weights
        posterior = getattr(sequences, "decoded_posterior", {})
        return {str(k): float(v) for k, v in posterior.items()}
    
    def sample(
        self,
        prompt: PromptSpec,
        constraint_spec: PromptSpec,
        n: int,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """Sample molecules using SMC."""
        collected: Dict[str, float] = {}
        batches = 0
        target_batches = max_batches or max(4, math.ceil(n / max(self.particles, 1)))
        
        pbar = tqdm(
            total=n,
            desc="SMC generation",
            unit="mol",
            ncols=80,
        )
        
        try:
            while len(collected) < n and batches < target_batches:
                posterior = asyncio.run(self._sample_once(prompt, constraint_spec))
                batches += 1
                
                if not posterior:
                    pbar.set_postfix({"Batch": f"{batches}/{target_batches}", "Valid": "0"})
                    continue
                
                valid_count = 0
                for sequence, weight in sorted(posterior.items(), key=lambda x: x[1], reverse=True):
                    if sequence.startswith(prompt.text):
                        candidate = sequence[len(prompt.text):].strip()
                    else:
                        candidate = sequence.strip()
                    
                    smiles = first_valid_smiles(candidate)
                    if not smiles:
                        continue
                    
                    if smiles not in collected:
                        collected[smiles] = weight
                        valid_count += 1
                        pbar.update(1)
                    
                    if len(collected) >= n:
                        break
                
                pbar.set_postfix({
                    "Batch": f"{batches}/{target_batches}",
                    "Valid": f"{valid_count}",
                    "Total": f"{len(collected)}/{n}"
                })
        finally:
            pbar.close()
        
        return collected


def _seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main Generation Functions
# ============================================================

def _samples_to_dataframe(
    samples: Dict[str, float],
    constraint_spec: PromptSpec,
    prompt_name: str,
    temperature: float,
    top_p: float,
) -> pd.DataFrame:
    """Convert SMC samples to DataFrame."""
    smiles = list(samples.keys())
    df = compute_properties_df(smiles)
    df["Weight"] = df["SMILES"].map(samples).astype(float)
    df = annotate_adherence(df, constraint_spec)
    df["Model"] = "GPT2-Zinc-87M+SMC"
    df["Prompt"] = prompt_name
    df["Temperature"] = temperature
    df["TopP"] = top_p
    
    columns = [
        "SMILES", "Valid", "QED", "MW", "logP", "RotB", "TPSA", "HBD", "HBA",
        "Adherence", "Weight", "Model", "Prompt", "Temperature", "TopP",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def generate_with_multi_prefix_smc(
    prefix_prompts: List[PromptSpec],
    constraint_spec: PromptSpec,
    n: int = 1_000,
    particles: int = 50,
    ess_threshold: float = 0.3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 60,
    top_k: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate molecules using SMC with multiple prefixes.
    Samples prefixes randomly and uses SMC to guide generation.
    """
    _seed_everything(seed)
    
    sampler = GenLMSMCSampler(
        model_name=DEFAULT_MODEL_NAME,
        particles=particles,
        ess_threshold=ess_threshold,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    
    all_samples: Dict[str, float] = {}
    samples_per_prefix = max(1, n // len(prefix_prompts))
    
    for prompt in prefix_prompts:
        prefix_samples = sampler.sample(
            prompt=prompt,
            constraint_spec=constraint_spec,
            n=samples_per_prefix,
        )
        # Combine samples, averaging weights if duplicate SMILES
        for smiles, weight in prefix_samples.items():
            if smiles in all_samples:
                all_samples[smiles] = (all_samples[smiles] + weight) / 2.0
            else:
                all_samples[smiles] = weight
        
        if len(all_samples) >= n:
            break
    
    # Take top n by weight
    sorted_samples = sorted(all_samples.items(), key=lambda x: x[1], reverse=True)[:n]
    final_samples = dict(sorted_samples)
    
    # Convert to DataFrame
    df = _samples_to_dataframe(
        final_samples,
        constraint_spec,
        prompt_name="multi_prefix_smc",
        temperature=temperature,
        top_p=top_p,
    )
    
    # Add prefix information (simplified - we used multiple prefixes)
    df["Prefix"] = "multi_prefix"
    
    return df


def run_constraint_experiment(
    constraint_level: str = "loosen",
    use_gradual_constraints: bool = True,
    property_ranges_path: str = "data/train_property_ranges.json",
    dataset: str = "Combined",
    n: int = 1_000,
    particles: int = 50,
    ess_threshold: float = 0.3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 60,
    top_k: int = 30,
    seed: int = 42,
    out_csv: str = "results/smc_results.csv",
    summary_csv: str = "results/smc_summary.csv",
) -> pd.DataFrame:
    """
    Run a single constraint-level experiment using SMC-guided generation.
    Uses gradual constraints (loosen/tight/ultra_tight) by default.
    """
    start_time = time.time()
    
    if not GENLM_AVAILABLE:
        raise ImportError("genlm-control is required. Install with: pip install genlm-control>=0.2.11")
    
    # Use gradual constraints (SmileyLlama-compatible) or legacy percentile-based constraints
    if use_gradual_constraints:
        # Map legacy names to gradual names for backward compatibility
        level_map = {"loose": "loosen", "tight": "tight", "ultra_tight": "ultra_tight"}
        gradual_level = level_map.get(constraint_level, constraint_level)
        constraint_spec = create_gradual_constraint_prompt(gradual_level)
        constraint_spec = PromptSpec(
            name=f"smc_gradual_{gradual_level}",
            text="",  # Prefix doesn't matter for constraint checking
            constraints=constraint_spec.constraints,
        )
    else:
        # Legacy percentile-based constraints
        constraint_ranges = load_property_ranges(property_ranges_path, dataset, constraint_level)
        base_prompt = GPT_ZINC_PROMPTS[0]
        constraint_spec = create_constraint_variant(base_prompt, constraint_ranges, tightness=constraint_level)
        constraint_spec = PromptSpec(
            name=f"smc_{constraint_level}",
            text="",  # Prefix doesn't matter for constraint checking
            constraints=constraint_spec.constraints,
        )
    
    # Get all prefix prompts
    all_prefixes = GPT_ZINC_PROMPTS
    
    # Generate with SMC (using multiple prefixes)
    df = generate_with_multi_prefix_smc(
        prefix_prompts=all_prefixes,
        constraint_spec=constraint_spec,
        n=n,
        particles=particles,
        ess_threshold=ess_threshold,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
    )
    
    # Add metadata
    df["ConstraintLevel"] = constraint_level
    
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
    summary["Model"] = "GPT2-Zinc-87M+SMC"
    summary["Prompt"] = "multi_prefix_smc"
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
    parser = argparse.ArgumentParser(description="SMC-guided GPT2-Zinc generation.")
    parser.add_argument("--constraint-level", type=str, default="loosen", choices=["loosen", "tight", "ultra_tight", "loose"])
    parser.add_argument("--use-gradual-constraints", action="store_true", default=True, help="Use gradual constraints (loosen/tight/ultra_tight)")
    parser.add_argument("--no-gradual-constraints", action="store_false", dest="use_gradual_constraints", help="Use legacy percentile-based constraints")
    parser.add_argument("--property-ranges", type=str, default="data/train_property_ranges.json")
    parser.add_argument("--dataset", type=str, default="Combined", choices=["ZINC", "ChEMBL", "Combined"])
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--particles", type=int, default=50, help="Number of SMC particles")
    parser.add_argument("--ess-threshold", type=float, default=0.3, help="ESS threshold for resampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="results/smc_results.csv")
    parser.add_argument("--summary-csv", type=str, default="results/smc_summary.csv")
    args = parser.parse_args()

    run_constraint_experiment(
        constraint_level=args.constraint_level,
        use_gradual_constraints=args.use_gradual_constraints,
        property_ranges_path=args.property_ranges,
        dataset=args.dataset,
        n=args.n,
        particles=args.particles,
        ess_threshold=args.ess_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        seed=args.seed,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
    )


if __name__ == "__main__":
    main()

