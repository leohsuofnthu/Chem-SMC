"""
Shared utility functions for molecule generation experiments.
"""
from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Conditional rdkit import for test compatibility
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    from rdkit import RDLogger
    # Suppress RDKit SMILES parse error messages
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    # Create dummy objects for type checking
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    QED = None
    rdMolDescriptors = None
    RDLogger = None


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    constraints: Dict[str, Tuple[Optional[float], Optional[float]]]


# Instruction-style prompts for SMILEY (instruction-tuned model)
# Format: "Output a SMILES string for..." to match SmileyLlama's expected format
# Only mw_logp_rotb is used in constraint-based experiments as the base prompt
SMILEY_PROMPTS: List[PromptSpec] = [
    PromptSpec(
        name="mw_logp_rotb",
        text="Output a SMILES string for a molecule with the following properties: molecular weight 300-400, logP 2-4, and <= 7 rotatable bonds:",
        constraints={
            "MW": (300, 400),
            "logP": (2, 4),
            "RotB": (None, 7),
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

# Property computation mapping (conditional on rdkit availability)
if RDKIT_AVAILABLE:
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
else:
    # Dummy mapping when rdkit is not available
    PROPERTY_FNS = {}

PROPERTY_COLUMNS = ["MW", "logP", "RotB", "HBD", "HBA", "TPSA", "QED"]

SMILES_PATTERN = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#\\/]+")

# Constraint level mapping for backward compatibility
CONSTRAINT_LEVEL_MAP = {"loose": "loose", "tight": "tight", "ultra_tight": "ultra_tight"}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES string. Returns None if rdkit is not available or SMILES is invalid."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_properties(smiles: str) -> Optional[Dict[str, float]]:
    """Compute molecular properties. Returns None if rdkit is not available or SMILES is invalid."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {name: fn(mol) for name, fn in PROPERTY_FNS.items()}


def compute_properties_df(smiles_list: Iterable[str], show_progress: bool = False, desc: str = "Computing properties") -> pd.DataFrame:
    """
    Compute molecular properties for a list of SMILES strings.
    
    Args:
        smiles_list: Iterable of SMILES strings
        show_progress: If True, show progress bar (requires tqdm)
        desc: Description for progress bar
    
    Returns:
        DataFrame with SMILES, Valid flag, and computed properties
    """
    records = []
    iterator = smiles_list
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc=desc, unit=" mols", ncols=80)
        except ImportError:
            iterator = smiles_list
    
    for s in iterator:
        if not RDKIT_AVAILABLE:
            records.append(
                {
                    "SMILES": s,
                    "Valid": False,
                    **{k: np.nan for k in PROPERTY_COLUMNS},
                    "Fsp3": np.nan,
                }
            )
            continue
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


def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def seed_all(seed: int) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not available
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Shared file loading utilities (consolidated from evaluate.py and plots.py)
# ============================================================

def load_results_file(path: str, temperature: Optional[float] = None) -> pd.DataFrame:
    """
    Load a single results CSV file with optional temperature filtering.
    
    Args:
        path: Path to CSV file
        temperature: Optional temperature value to filter by
    
    Returns:
        DataFrame with results
    """
    df = pd.read_csv(path)
    if temperature is not None and "Temperature" in df.columns:
        df = df[np.isclose(df["Temperature"], temperature)]
    if "Valid" not in df.columns:
        if RDKIT_AVAILABLE:
            df["Valid"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        else:
            df["Valid"] = False
    return df


def load_results_by_pattern(pattern: str, temperature: Optional[float] = None) -> pd.DataFrame:
    """
    Load all results files matching a glob pattern and combine them.
    Adds ConstraintType column based on filename to distinguish gradual vs range-based.
    
    Args:
        pattern: Glob pattern to match result files (e.g., 'results/baseline_*_results.csv')
        temperature: Optional temperature filter
    
    Returns:
        Combined DataFrame, or empty DataFrame if no files found.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        df = load_results_file(f, temperature)
        # Add ConstraintType based on filename to distinguish gradual vs range-based
        f_str = str(f).lower()
        if "gradual" in f_str:
            df["ConstraintType"] = "gradual"
        elif "range" in f_str:
            df["ConstraintType"] = "range-based"
        elif "baseline" in f_str:
            # Baseline uses range-based constraints
            df["ConstraintType"] = "range-based"
        else:
            # Default: try to infer from ConstraintLevel
            # Note: Both gradual and range-based use "loose" now, so we rely on filename or other indicators
            if "ConstraintLevel" in df.columns:
                # If we can't infer from filename, leave as None - will be handled later
                df["ConstraintType"] = None
            else:
                df["ConstraintType"] = None
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def filter_valid_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include valid molecules.
    
    Args:
        df: DataFrame with 'Valid' column
    
    Returns:
        Filtered DataFrame with only valid molecules
    """
    return df[df["Valid"]].copy() if "Valid" in df.columns else df


def load_experiment_results(
    results_dir: str = "results",
    baseline: Optional[str] = None,
    smc: Optional[str] = None,
    smiley: Optional[str] = None,
    temperature: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load experiment results for baseline, SMC, and SmileyLlama models.
    Handles auto-detection of combined vs individual files, and gradual vs range-based results.
    
    Args:
        results_dir: Directory containing result files
        baseline: Optional baseline results file or pattern (None = auto-detect)
        smc: Optional SMC results file or pattern (None = auto-detect)
        smiley: Optional SmileyLlama results file or pattern (None = auto-detect)
        temperature: Optional temperature filter
    
    Returns:
        Tuple of (baseline_df, smc_gradual_df, smc_range_df, smiley_gradual_df, smiley_range_df, smc_combined_df, smiley_combined_df)
        Note: smc_combined_df and smiley_combined_df are for backward compatibility
    """
    # Load baseline
    if baseline is None:
        baseline_combined = Path(results_dir) / "baseline_results.csv"
        if baseline_combined.exists():
            baseline_df = load_results_file(str(baseline_combined), temperature)
            baseline_df["ConstraintType"] = "range-based"  # Baseline uses range-based constraints
        else:
            baseline_df = load_results_by_pattern(f"{results_dir}/baseline_*_results.csv", temperature)
            # Ensure ConstraintType is set for baseline
            if "ConstraintType" not in baseline_df.columns or baseline_df["ConstraintType"].isna().any():
                baseline_df["ConstraintType"] = "range-based"
    else:
        if "*" in baseline:
            baseline_df = load_results_by_pattern(baseline, temperature)
            if "ConstraintType" not in baseline_df.columns or baseline_df["ConstraintType"].isna().any():
                baseline_df["ConstraintType"] = "range-based"
        else:
            baseline_df = load_results_file(baseline, temperature)
            baseline_df["ConstraintType"] = "range-based"  # Baseline uses range-based constraints
    
    # Load SMC results (distinguish gradual vs range-based)
    smc_gradual_df = pd.DataFrame()
    smc_range_df = pd.DataFrame()
    smc_combined_df = pd.DataFrame()
    
    if smc is None:
        smc_all = Path(results_dir) / "smc_all_results.csv"
        smc_gradual = Path(results_dir) / "smc_gradual_results.csv"
        smc_range = Path(results_dir) / "smc_range_results.csv"
        
        # Always try to load individual files first (they have all constraint levels)
        individual_files = load_results_by_pattern(f"{results_dir}/smc_*_*_results.csv", temperature)
        if not individual_files.empty:
            # Individual files loaded - split by ConstraintType
            if "ConstraintType" in individual_files.columns:
                smc_gradual_df = individual_files[individual_files["ConstraintType"] == "gradual"].copy()
                smc_range_df = individual_files[individual_files["ConstraintType"] == "range-based"].copy()
            smc_combined_df = individual_files.copy()
        elif smc_all.exists():
            smc_combined_df = load_results_file(str(smc_all), temperature)
            # Try to infer ConstraintType from ConstraintLevel
            if "ConstraintType" not in smc_combined_df.columns:
                # ConstraintType should be set from filename in load_results_by_pattern
                # If missing, leave as None - will be handled later
                pass
            # Split into gradual and range-based
            if "ConstraintType" in smc_combined_df.columns:
                smc_gradual_df = smc_combined_df[smc_combined_df["ConstraintType"] == "gradual"].copy()
                smc_range_df = smc_combined_df[smc_combined_df["ConstraintType"] == "range-based"].copy()
        elif smc_gradual.exists() or smc_range.exists():
            if smc_gradual.exists():
                smc_gradual_df = load_results_file(str(smc_gradual), temperature)
                smc_gradual_df["ConstraintType"] = "gradual"
            if smc_range.exists():
                smc_range_df = load_results_file(str(smc_range), temperature)
                smc_range_df["ConstraintType"] = "range-based"
            smc_combined_df = pd.concat([smc_gradual_df, smc_range_df], ignore_index=True) if not (smc_gradual_df.empty and smc_range_df.empty) else pd.DataFrame()
        else:
            # Load all SMC files with pattern matching (fallback)
            smc_combined_df = load_results_by_pattern(f"{results_dir}/smc_*_results.csv", temperature)
            # Split into gradual and range-based based on ConstraintType
            if "ConstraintType" in smc_combined_df.columns:
                smc_gradual_df = smc_combined_df[smc_combined_df["ConstraintType"] == "gradual"].copy()
                smc_range_df = smc_combined_df[smc_combined_df["ConstraintType"] == "range-based"].copy()
    else:
        if "*" in smc:
            smc_combined_df = load_results_by_pattern(smc, temperature)
            # Split into gradual and range-based based on ConstraintType
            if "ConstraintType" in smc_combined_df.columns:
                smc_gradual_df = smc_combined_df[smc_combined_df["ConstraintType"] == "gradual"].copy()
                smc_range_df = smc_combined_df[smc_combined_df["ConstraintType"] == "range-based"].copy()
        else:
            smc_combined_df = load_results_file(smc, temperature)
            if "gradual" in smc.lower():
                smc_gradual_df = smc_combined_df.copy()
                smc_gradual_df["ConstraintType"] = "gradual"
            elif "range" in smc.lower():
                smc_range_df = smc_combined_df.copy()
                smc_range_df["ConstraintType"] = "range-based"
            else:
                # Try to infer from filename or ConstraintLevel
                if "ConstraintType" not in smc_combined_df.columns:
                    if "ConstraintLevel" in smc_combined_df.columns:
                        smc_combined_df["ConstraintType"] = smc_combined_df["ConstraintLevel"].apply(
                            lambda x: "gradual" if x == "loosen" else ("range-based" if x == "loose" else None)
                        )
    
    # Load SmileyLlama results (distinguish gradual vs range-based)
    smiley_gradual_df = pd.DataFrame()
    smiley_range_df = pd.DataFrame()
    smiley_combined_df = pd.DataFrame()
    
    if smiley is None:
        smiley_all = Path(results_dir) / "smiley_all_results.csv"
        smiley_gradual = Path(results_dir) / "smiley_gradual_results.csv"
        smiley_range = Path(results_dir) / "smiley_range_results.csv"
        
        # Always try to load individual files first (they have all constraint levels)
        individual_files = load_results_by_pattern(f"{results_dir}/smiley_*_*_results.csv", temperature)
        if not individual_files.empty:
            # Individual files loaded - split by ConstraintType
            if "ConstraintType" in individual_files.columns:
                smiley_gradual_df = individual_files[individual_files["ConstraintType"] == "gradual"].copy()
                smiley_range_df = individual_files[individual_files["ConstraintType"] == "range-based"].copy()
            smiley_combined_df = individual_files.copy()
        elif smiley_all.exists():
            smiley_combined_df = load_results_file(str(smiley_all), temperature)
            # Try to infer ConstraintType from ConstraintLevel
            if "ConstraintType" not in smiley_combined_df.columns:
                # ConstraintType should be set from filename in load_results_by_pattern
                # If missing, leave as None - will be handled later
                pass
            # Split into gradual and range-based
            if "ConstraintType" in smiley_combined_df.columns:
                smiley_gradual_df = smiley_combined_df[smiley_combined_df["ConstraintType"] == "gradual"].copy()
                smiley_range_df = smiley_combined_df[smiley_combined_df["ConstraintType"] == "range-based"].copy()
        elif smiley_gradual.exists() or smiley_range.exists():
            if smiley_gradual.exists():
                smiley_gradual_df = load_results_file(str(smiley_gradual), temperature)
                smiley_gradual_df["ConstraintType"] = "gradual"
            if smiley_range.exists():
                smiley_range_df = load_results_file(str(smiley_range), temperature)
                smiley_range_df["ConstraintType"] = "range-based"
            smiley_combined_df = pd.concat([smiley_gradual_df, smiley_range_df], ignore_index=True) if not (smiley_gradual_df.empty and smiley_range_df.empty) else pd.DataFrame()
        else:
            # Load all SmileyLlama files with pattern matching (fallback)
            smiley_combined_df = load_results_by_pattern(f"{results_dir}/smiley_*_results.csv", temperature)
            # Split into gradual and range-based based on ConstraintType
            if "ConstraintType" in smiley_combined_df.columns:
                smiley_gradual_df = smiley_combined_df[smiley_combined_df["ConstraintType"] == "gradual"].copy()
                smiley_range_df = smiley_combined_df[smiley_combined_df["ConstraintType"] == "range-based"].copy()
    else:
        if "*" in smiley:
            smiley_combined_df = load_results_by_pattern(smiley, temperature)
            # Split into gradual and range-based based on ConstraintType
            if "ConstraintType" in smiley_combined_df.columns:
                smiley_gradual_df = smiley_combined_df[smiley_combined_df["ConstraintType"] == "gradual"].copy()
                smiley_range_df = smiley_combined_df[smiley_combined_df["ConstraintType"] == "range-based"].copy()
        else:
            smiley_combined_df = load_results_file(smiley, temperature)
            if "gradual" in smiley.lower():
                smiley_gradual_df = smiley_combined_df.copy()
                smiley_gradual_df["ConstraintType"] = "gradual"
            elif "range" in smiley.lower():
                smiley_range_df = smiley_combined_df.copy()
                smiley_range_df["ConstraintType"] = "range-based"
            else:
                # Try to infer from filename or ConstraintLevel
                if "ConstraintType" not in smiley_combined_df.columns:
                    # ConstraintType should be set from filename in load_results_by_pattern
                    # If missing, leave as None - will be handled later
                    pass
    
    return baseline_df, smc_gradual_df, smc_range_df, smiley_gradual_df, smiley_range_df, smc_combined_df, smiley_combined_df


def count_result_files(
    results_dir: str,
    baseline: Optional[str] = None,
    smc: Optional[str] = None,
    smiley: Optional[str] = None,
) -> tuple[int, int, int]:
    """
    Count result files for baseline, SMC, and SmileyLlama.
    
    Args:
        results_dir: Directory containing result files
        baseline: Optional baseline results file or pattern (None = auto-detect)
        smc: Optional SMC results file or pattern (None = auto-detect)
        smiley: Optional SmileyLlama results file or pattern (None = auto-detect)
    
    Returns:
        Tuple of (baseline_count, smc_count, smiley_count)
    """
    baseline_count = 0
    smc_count = 0
    smiley_count = 0
    
    if baseline is None:
        baseline_combined = Path(results_dir) / "baseline_results.csv"
        if baseline_combined.exists():
            baseline_count = 1
        else:
            baseline_files = glob.glob(f"{results_dir}/baseline_*_results.csv")
            baseline_count = len(baseline_files) if baseline_files else 0
    else:
        baseline_count = 1
    
    if smc is None:
        smc_all = Path(results_dir) / "smc_all_results.csv"
        smc_gradual = Path(results_dir) / "smc_gradual_results.csv"
        smc_range = Path(results_dir) / "smc_range_results.csv"
        if smc_all.exists():
            smc_count = 1
        elif smc_gradual.exists() or smc_range.exists():
            smc_count = sum([smc_gradual.exists(), smc_range.exists()])
        else:
            smc_files = glob.glob(f"{results_dir}/smc_*_results.csv")
            smc_count = len(smc_files) if smc_files else 0
    else:
        smc_count = 1
    
    if smiley is None:
        smiley_all = Path(results_dir) / "smiley_all_results.csv"
        smiley_gradual = Path(results_dir) / "smiley_gradual_results.csv"
        smiley_range = Path(results_dir) / "smiley_range_results.csv"
        if smiley_all.exists():
            smiley_count = 1
        elif smiley_gradual.exists() or smiley_range.exists():
            smiley_count = sum([smiley_gradual.exists(), smiley_range.exists()])
        else:
            smiley_files = glob.glob(f"{results_dir}/smiley_*_results.csv")
            smiley_count = len(smiley_files) if smiley_files else 0
    else:
        smiley_count = 1
    
    return baseline_count, smc_count, smiley_count


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
        NOTE: QED is excluded from constraints (only used for analysis)
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
    
    # Properties to exclude from constraints (QED is only for analysis, not constraints)
    excluded_props = {"QED"}
    
    ranges = {}
    for prop, value in level_data.items():
        # Skip QED - it's not used as a constraint
        if prop in excluded_props:
            continue
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


def create_gradual_smiley_constraints(
    constraint_level: str = "loose",
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Create gradual constraints compatible with SmileyLlama's training format.
    
    SmileyLlama understands:
    - MW: <= 300, <= 400, <= 500, <= 600, > 600
    - logP: <= 3, <= 4, <= 5, <= 10, <= 15, > 15
    - RotB: <= 7, <= 10, > 10
    
    Gradual constraint levels (progressive tightening - adding one condition at a time):
    - loose: <= 500 MW (single constraint, covers most drug-like molecules)
    - tight: <= 400 MW + <= 4 logP (adds logP constraint, typical drug-like)
    - ultra_tight: <= 350 MW + <= 3.5 logP + <= 8 RotB (adds RotB constraint, stricter drug-like)
    
    Note: These constraints use upper bounds only (no lower bounds) to match SmileyLlama's training format.
    The reward function will still guide toward reasonable values within these bounds.
    """
    if constraint_level == "loose":
        return {"MW": (None, 500.0)}
    elif constraint_level == "tight":
        return {"MW": (None, 400.0), "logP": (None, 4.0)}
    elif constraint_level == "ultra_tight":
        return {"MW": (None, 350.0), "logP": (None, 3.5), "RotB": (None, 8.0)}
    else:
        raise ValueError(f"Unknown constraint level: {constraint_level}. Use 'loose', 'tight', or 'ultra_tight'")


def create_gradual_smiley_prompt_text(
    constraint_level: str = "loose",
) -> str:
    """
    Create SmileyLlama prompt text for gradual constraints.
    Uses upper-bound format (<= X) that SmileyLlama understands.
    """
    if constraint_level == "loose":
        return "Output a SMILES string for a molecule with the following properties: <= 500 molecular weight:"
    elif constraint_level == "tight":
        return "Output a SMILES string for a molecule with the following properties: <= 400 molecular weight, <= 4 logP:"
    elif constraint_level == "ultra_tight":
        return "Output a SMILES string for a molecule with the following properties: <= 350 molecular weight, <= 3.5 logP, <= 8 rotatable bonds:"
    else:
        raise ValueError(f"Unknown constraint level: {constraint_level}")


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
    # Create new constraints dict, using EXACT ranges for properties that exist in constraint_ranges
    # NOTE: These exact constraints are used for evaluation (both SmileyLlama and SMC)
    # The prompt text below is separate and only affects SmileyLlama's generation, not evaluation
    # IMPORTANT: Use ALL properties from constraint_ranges for fair evaluation across all models
    # This ensures SMC and SmileyLlama use the SAME constraints regardless of base prompt
    new_constraints = {}
    
    # First, add ALL properties from constraint_ranges (this ensures fairness)
    for prop, (lower, upper) in constraint_ranges.items():
        new_constraints[prop] = (float(lower), float(upper))
    
    # Then, add any properties from base_prompt that are NOT in constraint_ranges
    # (for backward compatibility)
    for prop, (lower, upper) in base_prompt.constraints.items():
        if prop not in constraint_ranges:
            # Keep original constraint only if not in constraint_ranges
            new_constraints[prop] = (lower, upper)
    
    # Create new prompt name with tightness suffix
    new_name = f"{base_prompt.name}_{tightness}"
    
    # Update prompt text for instruction-style prompts (SMILEY)
    # IMPORTANT: SmileyLlama was trained on upper-bound format (<= X), not range format (X-Y)
    # Convert ranges to upper-bound format to match training distribution
    # NOTE: This prompt text is ONLY for SmileyLlama generation - evaluation uses new_constraints above
    new_text = base_prompt.text
    if any(prop in constraint_ranges for prop in ["MW", "logP", "RotB"]):
        # Update molecular weight - use both lower and upper bounds when available
        # Training format: <= 300, <= 400, <= 500, <= 600, > 600
        if "MW" in constraint_ranges:
            mw_min, mw_max = constraint_ranges["MW"]
            # Round to nearest training category: 300, 400, 500, 600, or use exact value
            mw_upper = mw_max
            if mw_max <= 300:
                mw_upper = 300
            elif mw_max <= 400:
                mw_upper = 400
            elif mw_max <= 500:
                mw_upper = 500
            elif mw_max <= 600:
                mw_upper = 600
            else:
                mw_upper = int(mw_max)  # Use exact value for > 600
            
            if "molecular weight" in base_prompt.text.lower():
                # If we have both min and max, use ">= X and <= Y" format
                if mw_min is not None and mw_max is not None:
                    mw_lower = int(mw_min)
                    new_text = re.sub(
                        r'molecular weight\s+[\d.]+-[\d.]+',
                        f'>= {mw_lower} and <= {mw_upper:.0f} molecular weight',
                        new_text,
                        flags=re.IGNORECASE
                    )
                    new_text = re.sub(
                        r'<=\s*\d+\s+molecular weight',
                        f'>= {mw_lower} and <= {mw_upper:.0f} molecular weight',
                        new_text,
                        flags=re.IGNORECASE
                    )
                else:
                    # Only upper bound - use upper-bound format
                    new_text = re.sub(
                        r'molecular weight\s+[\d.]+-[\d.]+',
                        f'<= {mw_upper:.0f} molecular weight',
                        new_text,
                        flags=re.IGNORECASE
                    )
                    new_text = re.sub(
                        r'<=\s*\d+\s+molecular weight',
                        f'<= {mw_upper:.0f} molecular weight',
                        new_text,
                        flags=re.IGNORECASE
                    )
            elif "MW" in base_prompt.text:
                # If we have both min and max, use ">= X and <= Y" format
                if mw_min is not None and mw_max is not None:
                    mw_lower = int(mw_min)
                    new_text = re.sub(
                        r'MW\s+[\d.]+-[\d.]+',
                        f'>= {mw_lower} and <= {mw_upper:.0f} MW',
                        new_text
                    )
                else:
                    # Only upper bound
                    new_text = re.sub(
                        r'MW\s+[\d.]+-[\d.]+',
                        f'<= {mw_upper:.0f} MW',
                        new_text
                    )
        
        # Update logP - use both lower and upper bounds when available
        # Training format: <= 3, <= 4, <= 5, <= 10, <= 15, > 15
        if "logP" in constraint_ranges:
            logp_min, logp_max = constraint_ranges["logP"]
            # Round to nearest training category: 3, 4, 5, 10, 15, or use exact value
            logp_upper = logp_max
            if logp_max <= 3:
                logp_upper = 3
            elif logp_max <= 4:
                logp_upper = 4
            elif logp_max <= 5:
                logp_upper = 5
            elif logp_max <= 10:
                logp_upper = 10
            elif logp_max <= 15:
                logp_upper = 15
            else:
                logp_upper = int(logp_max)  # Use exact value for > 15
            
            if "logp" in base_prompt.text.lower():
                # If we have both min and max, use ">= X and <= Y" format
                if logp_min is not None and logp_max is not None:
                    logp_lower = logp_min if logp_min >= 0 else int(logp_min)
                    new_text = re.sub(
                        r'logP\s+[\d.]+-[\d.]+',
                        f'>= {logp_lower:.1f} and <= {logp_upper:.0f} logP',
                        new_text,
                        flags=re.IGNORECASE
                    )
                    new_text = re.sub(
                        r'<=\s*[\d.]+\s+logP',
                        f'>= {logp_lower:.1f} and <= {logp_upper:.0f} logP',
                        new_text,
                        flags=re.IGNORECASE
                    )
                else:
                    # Only upper bound - use upper-bound format
                    new_text = re.sub(
                        r'logP\s+[\d.]+-[\d.]+',
                        f'<= {logp_upper:.0f} logP',
                        new_text,
                        flags=re.IGNORECASE
                    )
                    new_text = re.sub(
                        r'<=\s*[\d.]+\s+logP',
                        f'<= {logp_upper:.0f} logP',
                        new_text,
                        flags=re.IGNORECASE
                    )
        
        # Update rotatable bonds - use both lower and upper bounds when available
        # Training format: <= 7, <= 10, > 10 (upper bounds) or >= X and <= Y (ranges)
        if "RotB" in constraint_ranges:
            rotb_min, rotb_max = constraint_ranges["RotB"]
            if "rotatable" in base_prompt.text.lower():
                # Round to nearest training category: 7, 10, or use exact value
                rotb_upper = rotb_max
                if rotb_max <= 7:
                    rotb_upper = 7
                elif rotb_max <= 10:
                    rotb_upper = 10
                else:
                    rotb_upper = int(rotb_max)  # Use exact value for > 10
                
                # If we have both min and max, use ">= X and <= Y" format
                if rotb_min is not None and rotb_min > 0 and rotb_max is not None:
                    rotb_lower = int(rotb_min)
                    # Replace any format with both bounds
                    new_text = re.sub(
                        r'\d+-\d+\s+rotatable bonds',
                        f'>= {rotb_lower} and <= {rotb_upper:.0f} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
                    new_text = re.sub(
                        r'<=\s*\d+\s+rotatable bonds',
                        f'>= {rotb_lower} and <= {rotb_upper:.0f} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
                    new_text = re.sub(
                        r'>=\s*\d+\s+rotatable bonds',
                        f'>= {rotb_lower} and <= {rotb_upper:.0f} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
                elif rotb_max is not None:
                    # Only upper bound - use upper-bound format
                    new_text = re.sub(
                        r'\d+-\d+\s+rotatable bonds',
                        f'<= {rotb_upper:.0f} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
                    new_text = re.sub(
                        r'<=\s*\d+\s+rotatable bonds',
                        f'<= {rotb_upper:.0f} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
                elif rotb_min is not None and rotb_min > 0:
                    # Only lower bound - use lower-bound format (like Fraction sp3: > 0.4)
                    rotb_lower = int(rotb_min)
                    new_text = re.sub(
                        r'>=\s*\d+\s+rotatable bonds',
                        f'> {rotb_lower} rotatable bonds',
                        new_text,
                        flags=re.IGNORECASE
                    )
    
    # Return PromptSpec with:
    # - text: Upper-bound or range format (>= X and <= Y) for SmileyLlama prompt (generation only)
    # - constraints: Exact ranges for evaluation (both SmileyLlama and SMC)
    return PromptSpec(
        name=new_name,
        text=new_text,  # Used for SmileyLlama generation (>= X and <= Y or <= X format)
        constraints=new_constraints,  # Used for evaluation (exact ranges, fair for all models)
    )


def create_gradual_constraint_prompt(
    constraint_level: str = "loose",
) -> PromptSpec:
    """
    Create a PromptSpec for gradual constraints (loose/tight/ultra_tight).
    Uses SmileyLlama-compatible format with upper bounds only.
    """
    constraints = create_gradual_smiley_constraints(constraint_level)
    text = create_gradual_smiley_prompt_text(constraint_level)
    name = f"gradual_{constraint_level}"
    
    return PromptSpec(
        name=name,
        text=text,
        constraints=constraints,
    )

