"""
Shared utility functions for molecule generation experiments.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit SMILES parse error messages
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    constraints: Dict[str, Tuple[Optional[float], Optional[float]]]


# Instruction-style prompts for SMILEY (instruction-tuned model)
# Format: "Output a SMILES string for..." to match SmileyLlama's expected format
SMILEY_PROMPTS: List[PromptSpec] = [
    PromptSpec(
        name="lipinski",
        text="Output a SMILES string for a drug like molecule with the following properties: <= 5 H-bond donors, <= 10 H-bond acceptors, <= 500 molecular weight, <= 5 logP:",
        constraints={
            "HBD": (None, 5),
            "HBA": (None, 10),
            "MW": (None, 500),
            "logP": (None, 5),
        },
    ),
    PromptSpec(
        name="mw_logp_rotb",
        text="Output a SMILES string for a molecule with the following properties: molecular weight 300-400, logP 2-4, and <= 7 rotatable bonds:",
        constraints={
            "MW": (300, 400),
            "logP": (2, 4),
            "RotB": (None, 7),
        },
    ),
    PromptSpec(
        name="tpsa_fsp3",
        text="Output a SMILES string for a molecule with the following properties: <= 90 TPSA and > 0.5 Fsp3:",
        constraints={
            "TPSA": (None, 90),
            "Fsp3": (0.5, None),
        },
    ),
    PromptSpec(
        name="druglike_qed",
        text="Output a SMILES string for a drug-like molecule:",
        constraints={
            # QED >= 0.6 is a common heuristic for drug-like behaviour.
            "QED": (0.6, None),
        },
    ),
]

# Strategic prefix-style prompts for GPT-Zinc (base model trained on SMILES)
# Each prompt is designed to bias toward specific molecular features and properties
GPT_ZINC_PROMPTS: List[PromptSpec] = [
    PromptSpec(
        name="aromatic_core",
        text="C1=CC=",  # Encourages ring closure & aromatic substituents
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="amino_aliphatic",
        text="CCN",  # Introduces N heteroatom early for secondary/tertiary amines
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="polar_ether",
        text="COC",  # Increases oxygen / hydrogen bonding for ethers and alcohols
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="carbonyl_anchor",
        text="CC(=O)",  # Helps model build amides, esters
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="heterocycle_friendly",
        text="C1CN",  # Ring templates for morpholine, piperazine
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="sulfur_phosphorus",
        text="CCS",  # Rare atom exposure for thioethers, sulfoxides
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="extended_polar",
        text="CCOCCN",  # Pushes toward drug-like logP ~2-4
        constraints={
            "MW": (200, 500),
            "logP": (2, 4),  # Specific logP range for drug-likeness
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="fallback_generic",
        text="CCC",  # Stable control condition for baseline aliphatic generation
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="aliphatic_ring",
        text="C1CCCCC1",  # Cyclohexane; low logP anchor without π system
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="aromatic_ring",
        text="c1ccccc1",  # Benzene; aromatic core (complements aromatic_core)
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="hetero_aromatic",
        text="n1ccccc1",  # Pyridine; hetero-aromatic ring
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="saturated_n_heterocycle",
        text="N1CCCCC1",  # Piperidine; saturated N-heterocycle
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="morpholine",
        text="O1CCNCC1",  # Morpholine; O/N heterocycle (more specific than heterocycle_friendly)
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="amide_starter",
        text="NC(=O)",  # Amide starter from amine side
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="sulfonamide_starter",
        text="NS(=O)(=O)",  # Sulfonamide starter
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="urea",
        text="NC(=O)N",  # Urea/guanidine flavor
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="ether_builder",
        text="COC",  # Simple ether builder (similar to polar_ether, but explicit)
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="nitrile",
        text="C#N",  # Nitrile handle
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="aryl_chloride",
        text="Clc1ccccc1",  # Aryl chloride; halogen vector
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
    PromptSpec(
        name="phosphonate",
        text="P(=O)(O)O",  # Phosphonate; branching; good for polar/charge
        constraints={
            "MW": (200, 500),
            "logP": (0, 5),
            "RotB": (None, 10),
        },
    ),
]

# Create prompt maps for both model types
SMILEY_PROMPT_MAP = {p.name: p for p in SMILEY_PROMPTS}
GPT_ZINC_PROMPT_MAP = {p.name: p for p in GPT_ZINC_PROMPTS}

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

SMILES_PATTERN = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#\\/]+")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_properties(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {name: fn(mol) for name, fn in PROPERTY_FNS.items()}


def compute_properties_df(smiles_list: Iterable[str]) -> pd.DataFrame:
    records = []
    for s in smiles_list:
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


def check_constraints(
    properties: Dict[str, float],
    constraints: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> bool:
    for key, (lower, upper) in constraints.items():
        value = properties.get(key)
        if value is None or np.isnan(value):
            return False
        if lower is not None and value < lower - 1e-6:
            return False
        if upper is not None and value > upper + 1e-6:
            return False
    return True


def annotate_adherence(df: pd.DataFrame, prompt: PromptSpec) -> pd.DataFrame:
    df = df.copy()
    flags = []
    for _, row in df.iterrows():
        if not row["Valid"]:
            flags.append(False)
            continue
        props = {k: row.get(k, np.nan) for k in PROPERTY_FNS.keys()}
        flags.append(check_constraints(props, prompt.constraints))
    df["Adherence"] = flags
    df["Prompt"] = prompt.name
    return df


def summarise_adherence(df: pd.DataFrame) -> pd.Series:
    total = len(df)
    valid = df["Valid"].sum()
    distinct = df.loc[df["Valid"], "SMILES"].nunique()
    return pd.Series(
        {
            "Valid %": 100.0 * valid / total if total else 0.0,
            "Distinct %": 100.0 * distinct / total if total else 0.0,
            "Adherence %": 100.0 * df["Adherence"].mean() if total else 0.0,
            "QED": df["QED"].mean(skipna=True),
        }
    )


def percentile_ranges(
    df: pd.DataFrame,
    cols: Iterable[str],
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    ranges = {}
    valid = df[df["Valid"]]
    for col in cols:
        values = valid[col].dropna()
        if values.empty:
            continue
        ranges[col] = (float(values.quantile(q_low)), float(values.quantile(q_high)))
    return ranges


def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def iter_candidate_smiles(text: str) -> List[str]:
    return SMILES_PATTERN.findall(text)


def first_valid_smiles(text: str) -> Optional[str]:
    for token in iter_candidate_smiles(text):
        canonical = canonicalize_smiles(token)
        if canonical:
            return canonical
    return None


def all_valid_smiles(text: str) -> List[str]:
    hits: List[str] = []
    for token in iter_candidate_smiles(text):
        canonical = canonicalize_smiles(token)
        if canonical and canonical not in hits:
            hits.append(canonical)
    return hits



def load_property_ranges(
    json_path: str, 
    dataset: str = "Combined",
    constraint_level: str = "loose"
) -> Dict[str, Tuple[float, float]]:
    """
    Load property ranges from JSON file.
    
    Args:
        json_path: Path to property ranges JSON file
        dataset: Which dataset to use ("ZINC", "ChEMBL", or "Combined")
        constraint_level: Which constraint level to use ("loose", "tight", or "ultra_tight")
                         Default: "loose" (backward compatible)
        
    Returns:
        Dictionary mapping property names to (min, max) tuples
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if dataset not in data:
        raise ValueError(f"Dataset '{dataset}' not found in {json_path}. Available: {list(data.keys())}")
    
    # Check if data has new structure (with constraint levels) or old structure (flat)
    dataset_data = data[dataset]
    
    if isinstance(dataset_data, dict) and any(k in dataset_data for k in ["loose", "tight", "ultra_tight"]):
        # New structure with constraint levels
        if constraint_level not in dataset_data:
            raise ValueError(
                f"Constraint level '{constraint_level}' not found in {json_path}. "
                f"Available: {list(dataset_data.keys())}"
            )
        level_data = dataset_data[constraint_level]
    else:
        # Old flat structure (backward compatibility)
        if constraint_level != "loose":
            raise ValueError(
                f"File {json_path} uses old flat structure. Only 'loose' constraint level available. "
                f"Please regenerate with analyze_train_data.py to get all constraint levels."
            )
        level_data = dataset_data
    
    ranges = {}
    for prop, value in level_data.items():
        if isinstance(value, list) and len(value) == 2:
            ranges[prop] = (float(value[0]), float(value[1]))
        else:
            raise ValueError(f"Invalid range format for property {prop}: {value}")
    
    return ranges


def compute_percentile_constraints(
    df: pd.DataFrame,
    properties: List[str],
    q_low: float,
    q_high: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute constraint ranges based on percentiles from a dataframe.
    
    Args:
        df: DataFrame with property columns
        properties: List of property names to compute constraints for
        q_low: Lower percentile (e.g., 0.25 for 25th percentile)
        q_high: Upper percentile (e.g., 0.75 for 75th percentile)
        
    Returns:
        Dictionary mapping property names to (lower_bound, upper_bound) tuples
    """
    valid_df = df[df["Valid"]].copy() if "Valid" in df.columns else df
    constraints = {}
    
    for prop in properties:
        if prop not in valid_df.columns:
            continue
        values = valid_df[prop].dropna()
        if len(values) == 0:
            continue
        constraints[prop] = (float(values.quantile(q_low)), float(values.quantile(q_high)))
    
    return constraints


def create_constraint_variant(
    base_prompt: PromptSpec,
    constraint_ranges: Dict[str, Tuple[float, float]],
    tightness: str = "tight",
) -> PromptSpec:
    """
    Create a variant of a prompt with modified constraints.
    
    Args:
        base_prompt: Original PromptSpec to modify
        constraint_ranges: Dictionary of property -> (min, max) ranges
        tightness: Suffix to add to prompt name ("tight", "ultra_tight", etc.)
        
    Returns:
        New PromptSpec with modified constraints
    """
    # Create new constraints dict, using ranges for properties that exist in constraint_ranges
    new_constraints = {}
    
    for prop, (lower, upper) in base_prompt.constraints.items():
        if prop in constraint_ranges:
            # Use the new range
            new_constraints[prop] = constraint_ranges[prop]
        else:
            # Keep original constraint
            new_constraints[prop] = (lower, upper)
    
    # Create new prompt name with tightness suffix
    new_name = f"{base_prompt.name}_{tightness}"
    
    # Update prompt text for instruction-style prompts (SMILEY)
    # Use regex to properly replace constraint values in the prompt text
    new_text = base_prompt.text
    if any(prop in constraint_ranges for prop in ["MW", "logP", "RotB", "QED"]):
        # Update molecular weight - match patterns like "molecular weight 300-400" or "MW 300-400"
        if "MW" in constraint_ranges:
            mw_min, mw_max = constraint_ranges["MW"]
            if "molecular weight" in base_prompt.text.lower():
                # Replace "molecular weight 300-400" pattern
                new_text = re.sub(
                    r'molecular weight\s+\d+-\d+',
                    f'molecular weight {mw_min:.0f}-{mw_max:.0f}',
                    new_text,
                    flags=re.IGNORECASE
                )
            elif "MW" in base_prompt.text:
                # Replace "MW 300-400" pattern (if MW appears as abbreviation)
                new_text = re.sub(
                    r'MW\s+\d+-\d+',
                    f'MW {mw_min:.0f}-{mw_max:.0f}',
                    new_text
                )
        
        # Update logP - match patterns like "logP 2-4" or "logP 2.0-4.0"
        if "logP" in constraint_ranges:
            logp_min, logp_max = constraint_ranges["logP"]
            if "logp" in base_prompt.text.lower():
                # Replace "logP 2-4" or "logP 2.0-4.0" patterns
                new_text = re.sub(
                    r'logP\s+[\d.]+-[\d.]+',
                    f'logP {logp_min:.1f}-{logp_max:.1f}',
                    new_text,
                    flags=re.IGNORECASE
                )
        
        # Update rotatable bonds - match patterns like "<= 7 rotatable bonds"
        if "RotB" in constraint_ranges:
            rotb_min, rotb_max = constraint_ranges["RotB"]
            if "rotatable" in base_prompt.text.lower():
                # Check if text has both min and max, or just one bound
                has_min_pattern = bool(re.search(r'>=\s*\d+\s+rotatable bonds', new_text, re.IGNORECASE))
                has_max_pattern = bool(re.search(r'<=\s*\d+\s+rotatable bonds', new_text, re.IGNORECASE))
                
                if rotb_min is not None and rotb_min > 0 and rotb_max is not None:
                    # We have both min and max constraints
                    if has_min_pattern and has_max_pattern:
                        # Replace both patterns
                        new_text = re.sub(
                            r'>=\s*\d+\s+rotatable bonds',
                            f'>= {rotb_min:.0f} rotatable bonds',
                            new_text,
                            flags=re.IGNORECASE
                        )
                        new_text = re.sub(
                            r'<=\s*\d+\s+rotatable bonds',
                            f'<= {rotb_max:.0f} rotatable bonds',
                            new_text,
                            flags=re.IGNORECASE
                        )
                    elif has_max_pattern and not has_min_pattern:
                        # Only upper bound in text, but we need both - convert to range format
                        new_text = re.sub(
                            r'<=\s*\d+\s+rotatable bonds',
                            f'{rotb_min:.0f}-{rotb_max:.0f} rotatable bonds',
                            new_text,
                            flags=re.IGNORECASE
                        )
                    elif has_min_pattern and not has_max_pattern:
                        # Only lower bound in text, but we need both - add upper bound
                        new_text = re.sub(
                            r'>=\s*\d+\s+rotatable bonds',
                            f'{rotb_min:.0f}-{rotb_max:.0f} rotatable bonds',
                            new_text,
                            flags=re.IGNORECASE
                        )
                elif rotb_max is not None:
                    # Only upper bound constraint
                    new_text = re.sub(
                        r'<=\s*\d+\s+rotatable bonds',
                        f'<= {rotb_max:.0f} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
                elif rotb_min is not None and rotb_min > 0:
                    # Only lower bound constraint
                    new_text = re.sub(
                        r'>=\s*\d+\s+rotatable bonds',
                        f'>= {rotb_min:.0f} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
    
    return PromptSpec(
        name=new_name,
        text=new_text,
        constraints=new_constraints,
    )

