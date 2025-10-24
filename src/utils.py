"""
Shared utility functions for molecule generation experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
import re


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    constraints: Dict[str, Tuple[Optional[float], Optional[float]]]


# Instruction-style prompts for SMILEY (instruction-tuned model)
SMILEY_PROMPTS: List[PromptSpec] = [
    PromptSpec(
        name="lipinski",
        text="Generate molecules satisfying Lipinski's rule of 5 (HBD <= 5, HBA <= 10, MW <= 500, logP <= 5).",
        constraints={
            "HBD": (None, 5),
            "HBA": (None, 10),
            "MW": (None, 500),
            "logP": (None, 5),
        },
    ),
    PromptSpec(
        name="mw_logp_rotb",
        text="Generate molecules with MW 300-400, logP 2-4, and Rotatable Bonds <= 7.",
        constraints={
            "MW": (300, 400),
            "logP": (2, 4),
            "RotB": (None, 7),
        },
    ),
    PromptSpec(
        name="tpsa_fsp3",
        text="Generate molecules with TPSA <= 90 and Fsp3 > 0.5.",
        constraints={
            "TPSA": (None, 90),
            "Fsp3": (0.5, None),
        },
    ),
    PromptSpec(
        name="druglike_qed",
        text="Generate a drug-like molecule.",
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
]

# Default to SMILEY prompts for backward compatibility
PROMPTS: List[PromptSpec] = SMILEY_PROMPTS

# Create prompt maps for both model types
SMILEY_PROMPT_MAP = {p.name: p for p in SMILEY_PROMPTS}
GPT_ZINC_PROMPT_MAP = {p.name: p for p in GPT_ZINC_PROMPTS}

# Default prompt map (for backward compatibility)
PROMPT_MAP = SMILEY_PROMPT_MAP

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


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


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
