#!/usr/bin/env bash

set -euo pipefail

PROMPTS=(lipinski mw_logp_rotb tpsa_fsp3 druglike_qed)
PROMPT_ARGS="${PROMPTS[@]}"
SMILEY_TEMPS=(1.0 0.7)
SMC_TEMPS=(1.0 0.7)

DATA_INPUT=${1:-data/zinc250k.csv}

# echo "[1/4] Preparing ZINC data splits..."
# python -m src.data_prep --input "$DATA_INPUT"

# echo "[2/4] Running baseline GPT2-Zinc generation..."
# python -m src.baseline_generate \
#   --prompts ${PROMPTS[@]} \
#   --n 1000 \
#   --temperature 1.0 \
#   --top_p 0.9 \
#   --batch_size 256 \
#   --out_csv results/baseline_results.csv \
#   --summary_csv results/baseline_summary.csv

# echo "[3/4] Running SmileyLlama generation..."
# python -m src.smiley_generate \
#   --prompts ${PROMPTS[@]} \
#   --n 1000 \
#   --temperatures ${SMILEY_TEMPS[@]} \
#   --top_p 0.9 \
#   --batch_size 128 \
#   --out_csv results/smiley_results.csv \
#   --summary_csv results/smiley_summary.csv

echo "[4/4] Running GenLM SMC generation with advanced grammar enforcement..."
python -m src.smc_generate \
  --prompts ${PROMPTS[@]} \
  --n 1000 \
  --temperatures ${SMC_TEMPS[@]} \
  --top_p 0.9 \
  --particles 10 \
  --out_csv results/smc_results.csv \
  --summary_csv results/smc_summary.csv

echo "All experiments completed. Outputs written under results/."
