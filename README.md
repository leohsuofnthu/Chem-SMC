# chem-smc

Two complementary strategies for controlled molecule generation:

1. **GPT2-Zinc-87M Baseline** with multi-prefix sampling and constraint filtering  
2. **SmileyLlama-8B** instruction following with prompt-based constraints

Both approaches are evaluated using constraint-based metrics: Adherence %, Valid %, Distinct %, and Diversity.

## Quickstart
```bash
# 1) Environment
conda create -n chem-smc python=3.10 -y
conda activate chem-smc
pip install -r requirements.txt

# 2) Place training data under data/:
#    - data/zinc250k/train.csv
#    - data/chembl/train.csv

# 3) End-to-end experiment
bash scripts/run_experiments.sh
```

## Quick Start (Dry-Run Test)

Before running the full pipeline, test everything works:

```bash
# Test the pipeline end-to-end (generates 10 molecules, ~1-2 minutes)
python scripts/test_pipeline.py
```

This will:
- Verify all imports work
- Check property ranges file exists
- Test GPT2-Zinc generation (small sample)
- Test evaluation pipeline
- Clean up test files

## Simplified Pipeline

The pipeline now runs **3 experiments per model** (loose/tight/ultra_tight constraints) instead of separate experiments for each prompt.

### Step 1: Analyze Training Data
```bash
python -m src.analyze_train_data --zinc-train data/zinc250k/train.csv --chembl-train data/chembl/train.csv
```
This generates `data/train_property_ranges.json` with constraint levels (loose, tight, ultra_tight).

### Step 2: Run Experiments
```bash
# Run all experiments (2 models × 3 constraint levels = 6 experiments total)
bash scripts/run_experiments.sh

# Or run individually:
# GPT2-Zinc baseline (multi-prefix, constraint-filtered)
python -m src.baseline_generate_constraint --constraint-level loose
python -m src.baseline_generate_constraint --constraint-level tight
python -m src.baseline_generate_constraint --constraint-level ultra_tight

# SmileyLlama (constraint variants)
python -m src.smiley_generate_constraint --constraint-level loose
python -m src.smiley_generate_constraint --constraint-level tight
python -m src.smiley_generate_constraint --constraint-level ultra_tight
```

### Step 3: Evaluate & Visualize
```bash
# Evaluate constraint-based metrics
python -m src.evaluate

# Generate plots
python -m src.plots
```

## Key Changes

**GPT2-Zinc Baseline:**
- **Multi-prefix sampling**: Randomly samples from all 20 available prefixes during generation
- **Constraint filtering**: Filters generated molecules by constraint adherence
- **3 experiments**: One per constraint level (loose/tight/ultra_tight)

**SmileyLlama:**
- **Constraint variants**: Uses pre-computed constraint ranges (loose/tight/ultra_tight)
- **3 experiments**: One per constraint level
- **Instruction-tuned**: Uses proper prompt template format for SmileyLlama

Outputs land in `results/` (CSV tables) and `figures/` (PNGs) with a unified schema:
`SMILES, Valid, QED, MW, logP, RotB, TPSA, HBD, HBA, Adherence, Weight, Model, Prompt, Temperature, TopP`.

## Evaluation Metrics

The evaluation focuses on 4 core metrics for constraint-based generation:
1. **Adherence %** - Percentage of molecules meeting property constraints
2. **Valid %** - Percentage of valid SMILES
3. **Distinct %** - Percentage of unique molecules  
4. **Diversity** - Tanimoto diversity (1 - mean similarity)

No reference dataset needed - evaluation is purely constraint-based.

## Notes
- SmileyLlama is heavy; 4-bit quantisation keeps GPU usage ≈4 GB. Pass `--no_quantize` to force full precision.  
- GPT2-Zinc baseline uses random prefix selection from all 20 available prefixes, then filters by constraints.  
- Adjust prompts or property ranges via `src/utils.py` (`PROMPTS` list).
- Constraint levels are pre-computed from training data: loose (5th-95th), tight (25th-75th), ultra_tight (40th-60th percentiles).
