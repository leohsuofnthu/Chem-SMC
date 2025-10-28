#!/usr/bin/env bash

set -euo pipefail

# SMILEY prompts (instruction-style for instruction-tuned model)
SMILEY_PROMPTS=(lipinski mw_logp_rotb tpsa_fsp3 druglike_qed)

# GPT-Zinc prompts (prefix-style for base model)
GPT_ZINC_PROMPTS=(aromatic_core amino_aliphatic polar_ether carbonyl_anchor heterocycle_friendly sulfur_phosphorus extended_polar fallback_generic)

SMILEY_TEMPS=(1.0 0.7)
SMC_TEMPS=(1.2 1.0)

DATA_INPUT=${1:-data/zinc250k.csv}

# echo "[1/4] Preparing ZINC data splits..."
# python -m src.data_prep --input "$DATA_INPUT"

# echo "[2/4] Running baseline GPT2-Zinc generation with strategic prompts..."
# python -m src.baseline_generate \
#   --prompts ${GPT_ZINC_PROMPTS[@]} \
#   --n 1000 \
#   --temperature 1.0 \
#   --top_p 0.9 \
#   --batch_size 256 \
#   --out_csv results/baseline_results.csv \
#   --summary_csv results/baseline_summary.csv

# echo "[3/4] Running SmileyLlama generation with instruction prompts..."
# python -m src.smiley_generate \
#   --prompts ${SMILEY_PROMPTS[@]} \
#   --n 1000 \
#   --temperatures ${SMILEY_TEMPS[@]} \
#   --top_p 0.9 \
#   --batch_size 128 \
#   --out_csv results/smiley_results.csv \
#   --summary_csv results/smiley_summary.csv

echo "[4/5] Running ChemGPT-4.7M generation with strategic prompts..."
python -m src.chemgpt_generate \
  --prompts ${GPT_ZINC_PROMPTS[@]} \
  --n 1000 \
  --temperature 1.0 \
  --top_p 0.9 \
  --batch_size 128 \
  --out_csv results/chemgpt_results.csv \
  --summary_csv results/chemgpt_summary.csv

# echo "[5/5] Running GenLM SMC generation with strategic prompts..."
# python -m src.smc_generate \
#   --prompts ${GPT_ZINC_PROMPTS[@]} \
#   --n 100 \
#   --temperatures ${SMC_TEMPS[@]} \
#   --top_p 0.9 \
#   --particles 10 \
#   --out_csv results/smc_results.csv \
#   --summary_csv results/smc_summary.csv

echo "All experiments completed. Outputs written under results/."
