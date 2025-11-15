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
| **constraint_levels** | loose/tight/ultra_tight | loose/tight/ultra_tight | loose/tight/ultra_tight | loose/tight/ultra_tight | loose/tight/ultra_tight |

---

## Detailed Settings by Experiment

### 1. GPT2-Zinc Baseline (Range-based Constraints)

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"`
- `n`: `1000`, `temperature`: `1.0`, `top_p`: `0.9`, `max_new_tokens`: `128`, `batch_size`: `256`, `seed`: `42`

**Output Files:**
- `results/baseline_{level}_results.csv`
- `results/baseline_{level}_summary.csv`

**Constraint Definition:**
- `loose`: 5th-95th percentile
- `tight`: 25th-75th percentile
- `ultra_tight`: 40th-60th percentile

---

### 2. GPT2-Zinc+SMC (Gradual Constraints)

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"`
- `use_gradual_constraints`: `True`
- `n`: `1000`, `particles`: `20`, `ess_threshold`: `0.3`, `temperature`: `1.0`, `top_p`: `0.9`, `max_new_tokens`: `60`, `top_k`: `30`, `seed`: `42`

**Output Files:**
- `results/smc_gradual_{level}_results.csv`
- `results/smc_gradual_{level}_summary.csv`

**Constraint Definition:**
- `loosen`: `MW <= 500` (1 condition)
- `tight`: `MW <= 400`, `logP <= 4` (2 conditions)
- `ultra_tight`: `MW <= 350`, `logP <= 3.5`, `RotB <= 8` (3 conditions)

---

### 3. GPT2-Zinc+SMC (Range-based Constraints)

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"`
- `use_gradual_constraints`: `False`
- `n`: `1000`, `particles`: `20`, `ess_threshold`: `0.3`, `temperature`: `1.0`, `top_p`: `0.9`, `max_new_tokens`: `60`, `top_k`: `30`, `seed`: `42`

**Output Files:**
- `results/smc_range_{level}_results.csv`
- `results/smc_range_{level}_summary.csv`

**Constraint Definition:**
- Same as Baseline (loose/tight/ultra_tight)

---

### 4. SmileyLlama (Gradual Constraints)

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"`
- `use_gradual_constraints`: `True`
- `n`: `1000`, `temperature`: `1.0`, `top_p`: `0.9`, `max_new_tokens`: `128`, `batch_size`: `50`, `seed`: `42`, `quantize`: `True`

**Output Files:**
- `results/smiley_gradual_{level}_results.csv`
- `results/smiley_gradual_{level}_summary.csv`

**Constraint Definition:**
- Same as SMC gradual constraints

---

### 5. SmileyLlama (Range-based Constraints)

**Settings:**
- `constraint_level`: `"loose"`, `"tight"`, or `"ultra_tight"`
- `use_gradual_constraints`: `False`
- `n`: `1000`, `temperature`: `1.0`, `top_p`: `0.9`, `max_new_tokens`: `128`, `batch_size`: `50`, `seed`: `42`, `quantize`: `True`

**Output Files:**
- `results/smiley_range_{level}_results.csv`
- `results/smiley_range_{level}_summary.csv`

**Constraint Definition:**
- Same as Baseline (loose/tight/ultra_tight)

---

## Common Settings

- **Property ranges**: `data/train_property_ranges.json`
- **Dataset**: `Combined` (ZINC + ChEMBL)
- **Temperature**: `1.0` (all experiments)
- **TopP**: `0.9` (all experiments)
- **Seed**: `42` (all experiments)

### Output Format:
All experiments produce CSV files with consistent columns:
- `SMILES`: Canonical SMILES string
- `Valid`: Boolean validity flag
- `MW`, `logP`, `RotB`, `TPSA`, `HBD`, `HBA`, `QED`: Molecular properties
- `Adherence`: Boolean constraint adherence flag
- `Weight`: Sample weight (1.0 for baseline/SmileyLlama, variable for SMC)
- `Model`: Model name
- `Prompt`: Prompt identifier
- `ConstraintLevel`: Constraint level used (loose/tight/ultra_tight)
- `Temperature`, `TopP`: Generation parameters
- `Prefix`: Prefix used for generation (for multi-prefix experiments)

### Summary Files:
All summary files include:
- `Adherence %`: Percentage meeting constraints
- `Valid %`: Percentage of valid SMILES
- `Distinct %`: Percentage of unique molecules
- `Diversity`: Tanimoto diversity (1 - mean similarity)
- `QED`: Mean QED score
- `Model`, `ConstraintLevel`, `Prompt`: Metadata
- `Temperature`: Generation temperature
- `Runtime_seconds`, `Runtime_minutes`, `Runtime_formatted`: Timing info


