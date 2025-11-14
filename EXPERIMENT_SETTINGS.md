# Experiment Settings Reference

Complete reference of all expected settings for each experiment type.

## Quick Reference Table

| Setting | Baseline | SMC (Gradual) | SMC (Range) | SmileyLlama (Gradual) | SmileyLlama (Range) |
|---------|----------|---------------|-------------|----------------------|---------------------|
| **n** | 1000 | 1000 | 1000 | 1000 | 1000 |
| **temperature** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **top_p** | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| **max_new_tokens** | 128 | 60 | 60 | 128 | 128 |
| **batch_size** | 256 | N/A | N/A | 50 | 50 |
| **particles** | N/A | 20 | 20 | N/A | N/A |
| **ess_threshold** | N/A | 0.3 | 0.3 | N/A | N/A |
| **top_k** | N/A | 30 | 30 | N/A | N/A |
| **seed** | 42 | 42 | 42 | 42 | 42 |
| **quantize** | N/A | N/A | N/A | True | True |
| **constraint_type** | Range | Gradual | Range | Gradual | Range |
| **constraint_levels** | loose/tight/ultra_tight | loosen/tight/ultra_tight | loose/tight/ultra_tight | loosen/tight/ultra_tight | loose/tight/ultra_tight |

---

## Detailed Settings by Experiment

### 1. GPT2-Zinc Baseline (Range-based Constraints)

**Purpose**: Baseline comparison using multi-prefix sampling with range-based constraints.

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"` (range-based)
- `property_ranges`: `"data/train_property_ranges.json"`
- `dataset`: `"Combined"`
- `n`: `1000` molecules
- `temperature`: `1.0`
- `top_p`: `0.9`
- `max_new_tokens`: `128`
- `batch_size`: `256`
- `seed`: `42`
- `filter_by_constraints`: `False` (for fair evaluation)

**Output Files:**
- Results: `results/baseline_{level}_results.csv`
- Summary: `results/baseline_{level}_summary.csv`
- Combined: `results/baseline_results.csv`

**Model**: `GPT2-Zinc-87M`

**Constraint Definition:**
- Range-based (percentile-based from training data)
- `loose`: 5th-95th percentile
- `tight`: 25th-75th percentile
- `ultra_tight`: 40th-60th percentile

---

### 2. GPT2-Zinc+SMC (Gradual Constraints)

**Purpose**: SMC-guided generation with gradual (upper-bound) constraints.

**Settings:**
- `constraint_level`: `"loosen"`, `"tight"`, or `"ultra_tight"` (gradual)
- `use_gradual_constraints`: `True`
- `property_ranges`: `"data/train_property_ranges.json"`
- `dataset`: `"Combined"`
- `n`: `1000` molecules
- `particles`: `20` (SMC particles, matches number of GPT_ZINC_PROMPTS)
- `ess_threshold`: `0.3` (effective sample size threshold)
- `temperature`: `1.0` (standardized for fair comparison)
- `top_p`: `0.9`
- `max_new_tokens`: `60`
- `top_k`: `30`
- `seed`: `42`

**Output Files:**
- Results: `results/smc_gradual_{level}_results.csv`
- Summary: `results/smc_gradual_{level}_summary.csv`
- Combined: `results/smc_gradual_results.csv`

**Model**: `GPT2-Zinc-87M+SMC`

**Constraint Definition:**
- Gradual (upper-bound only, SmileyLlama-compatible)
- `loosen`: `MW <= 500`, `logP <= 5`
- `tight`: `MW <= 400`, `logP <= 4`, `RotB <= 10`
- `ultra_tight`: `MW <= 350`, `logP <= 3.5`, `RotB <= 8`

---

### 3. GPT2-Zinc+SMC (Range-based Constraints)

**Purpose**: SMC-guided generation with range-based (percentile) constraints.

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"` (range-based)
- `use_gradual_constraints`: `False`
- `property_ranges`: `"data/train_property_ranges.json"`
- `dataset`: `"Combined"`
- `n`: `1000` molecules
- `particles`: `20` (matches number of GPT_ZINC_PROMPTS)
- `ess_threshold`: `0.3`
- `temperature`: `1.0` (standardized for fair comparison)
- `top_p`: `0.9`
- `max_new_tokens`: `60`
- `top_k`: `30`
- `seed`: `42`

**Output Files:**
- Results: `results/smc_range_{level}_results.csv`
- Summary: `results/smc_range_{level}_summary.csv`
- Combined: `results/smc_range_results.csv`

**Model**: `GPT2-Zinc-87M+SMC`

**Constraint Definition:**
- Range-based (percentile-based from training data)
- Same as Baseline (loose/tight/ultra_tight)

---

### 4. SmileyLlama (Gradual Constraints)

**Purpose**: Instruction-following generation with gradual constraints.

**Settings:**
- `constraint_level`: `"loosen"`, `"tight"`, or `"ultra_tight"` (gradual)
- `use_gradual_constraints`: `True`
- `base_prompt`: `"mw_logp_rotb"`
- `property_ranges`: `"data/train_property_ranges.json"`
- `dataset`: `"Combined"`
- `n`: `1000` molecules
- `temperature`: `1.0`
- `top_p`: `0.9`
- `max_new_tokens`: `128`
- `batch_size`: `50`
- `seed`: `42`
- `quantize`: `True`

**Output Files:**
- Results: `results/smiley_gradual_{level}_results.csv`
- Summary: `results/smiley_gradual_{level}_summary.csv`
- Combined: `results/smiley_gradual_results.csv`

**Model**: `SmileyLlama-8B`

**Constraint Definition:**
- Gradual (upper-bound only, SmileyLlama-compatible)
- Same as SMC gradual constraints

---

### 5. SmileyLlama (Range-based Constraints)

**Purpose**: Instruction-following generation with range-based constraints.

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"` (range-based)
- `use_gradual_constraints`: `False`
- `base_prompt`: `"mw_logp_rotb"`
- `property_ranges`: `"data/train_property_ranges.json"`
- `dataset`: `"Combined"`
- `n`: `1000` molecules
- `temperature`: `1.0`
- `top_p`: `0.9`
- `max_new_tokens`: `128`
- `batch_size`: `50`
- `seed`: `42`
- `quantize`: `True`

**Output Files:**
- Results: `results/smiley_range_{level}_results.csv`
- Summary: `results/smiley_range_{level}_summary.csv`
- Combined: `results/smiley_range_results.csv`

**Model**: `SmileyLlama-8B`

**Constraint Definition:**
- Range-based (percentile-based from training data)
- Same as Baseline (loose/tight/ultra_tight)

---

## Common Settings Across All Experiments

### Data Settings:
- **Property ranges file**: `data/train_property_ranges.json`
- **Dataset**: `Combined` (ZINC + ChEMBL)
- **Number of molecules**: `1000` per experiment

### Generation Parameters (Standardized):
- **Temperature**: `1.0` (all experiments) ✅ Standardized for fair comparison
- **TopP**: `0.9` (all experiments)
- **Seed**: `42` (all experiments)
- **Random state**: Fixed for reproducibility

### SMC-Specific Settings:
- **Particles**: `20` (matches number of GPT_ZINC_PROMPTS) ✅
- **ESS threshold**: `0.3`
- **Top-k**: `30`

### Output Format:
All experiments produce CSV files with consistent columns:
- `SMILES`: Canonical SMILES string
- `Valid`: Boolean validity flag
- `MW`, `logP`, `RotB`, `TPSA`, `HBD`, `HBA`, `QED`: Molecular properties
- `Adherence`: Boolean constraint adherence flag
- `Weight`: Sample weight (1.0 for baseline/SmileyLlama, variable for SMC)
- `Model`: Model name
- `Prompt`: Prompt identifier
- `ConstraintLevel`: Constraint level used
- `Temperature`, `TopP`: Generation parameters

### Summary Files:
All summary files include:
- `Adherence %`: Percentage meeting constraints
- `Valid %`: Percentage of valid SMILES
- `Distinct %`: Percentage of unique molecules
- `QED`: Mean QED score
- `Model`, `ConstraintLevel`, `Prompt`: Metadata
- `Temperature`: Generation temperature
- `Runtime_seconds`, `Runtime_minutes`, `Runtime_formatted`: Timing info

---

## Verification Status

✅ **All settings verified and fixed:**
- [x] Baseline batch_size: 64 → 256 ✅
- [x] SmileyLlama seed: 0 → 42 ✅
- [x] SMC temperature: 0.7 → 1.0 ✅ (standardized for fair comparison)
- [x] SMC particles: 50 → 20 ✅ (matches 20 GPT_ZINC_PROMPTS)
- [x] All other settings match expected values ✅
- [x] Output file naming is consistent ✅
- [x] Summary file format is consistent ✅

**Status**: All experiments are configured correctly with standardized settings for fair comparison. Ready to run.

