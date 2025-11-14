# chem-smc

Three strategies for controlled molecule generation:

1. **GPT2-Zinc-87M Baseline** with multi-prefix sampling and constraint evaluation (range-based constraints)
2. **GPT2-Zinc+SMC** with SMC-guided generation using gradual constraints  
3. **SmileyLlama-8B** instruction following with gradual constraints

All approaches are evaluated using constraint-based metrics: Adherence %, Valid %, Distinct %, and Diversity.

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

The pipeline runs **3 constraint levels per model** (loosen/tight/ultra_tight for SMC/SmileyLlama, loose/tight/ultra_tight for Baseline) for a total of **9 experiments**.

### Step 1: Analyze Training Data
```bash
python -m src.analyze_train_data --zinc-train data/zinc250k/train.csv --chembl-train data/chembl/train.csv
```
This generates `data/train_property_ranges.json` with constraint levels (loose, tight, ultra_tight).

### Step 2: Run Experiments
```bash
# Run all experiments (3 models × 3 constraint levels = 9 experiments total)
bash scripts/run_experiments.sh

# Or run individually:
# GPT2-Zinc baseline (range-based constraints: loose/tight/ultra_tight)
python -m src.baseline_generate_constraint --constraint-level loose
python -m src.baseline_generate_constraint --constraint-level tight
python -m src.baseline_generate_constraint --constraint-level ultra_tight

# GPT2-Zinc+SMC (gradual constraints: loosen/tight/ultra_tight)
python -m src.smc_generate_constraint --constraint-level loosen --use-gradual-constraints
python -m src.smc_generate_constraint --constraint-level tight --use-gradual-constraints
python -m src.smc_generate_constraint --constraint-level ultra_tight --use-gradual-constraints

# SmileyLlama (gradual constraints: loosen/tight/ultra_tight)
python -m src.smiley_generate_constraint --constraint-level loosen --use-gradual-constraints
python -m src.smiley_generate_constraint --constraint-level tight --use-gradual-constraints
python -m src.smiley_generate_constraint --constraint-level ultra_tight --use-gradual-constraints
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
- **Range-based constraints**: Uses percentile-based ranges from training data (loose/tight/ultra_tight)
- **Multi-prefix sampling**: Randomly samples from all available prefixes during generation
- **Constraint evaluation**: Evaluates constraint adherence without filtering
- **3 experiments**: One per constraint level

**GPT2-Zinc+SMC:**
- **Gradual constraints**: Uses fixed upper bounds (loosen/tight/ultra_tight)
- **SMC-guided generation**: Sequential Monte Carlo sampling with constraint-based rewards
- **Dynamic reward scaling**: Adjusts rewards based on constraint satisfaction
- **3 experiments**: One per constraint level

**SmileyLlama:**
- **Gradual constraints**: Uses fixed upper bounds compatible with SmileyLlama (loosen/tight/ultra_tight)
- **Instruction-tuned**: Uses proper prompt template format for SmileyLlama
- **3 experiments**: One per constraint level

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
- **Constraint types**: Baseline uses range-based (percentile) constraints, while SMC and SmileyLlama use gradual (upper-bound) constraints
- **Gradual constraints**: loosen (MW ≤ 300), tight (MW ≤ 300, logP ≤ 4), ultra_tight (MW ≤ 300, logP ≤ 4, RotB ≤ 10)
- **Range-based constraints**: Pre-computed from training data - loose (5th-95th percentile), tight (25th-75th), ultra_tight (40th-60th)
- SMC requires `genlm-control` library: `pip install genlm-control>=0.2.11`  
- Adjust prompts or property ranges via `src/utils.py`
