# PowerShell script to run experiments
param(
    [string]$DataInput = "data/zinc250k.csv"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# SMILEY prompts (instruction-style for instruction-tuned model)
$SmileyPrompts = @("lipinski", "mw_logp_rotb", "tpsa_fsp3", "druglike_qed")

# GPT-Zinc prompts (prefix-style for base model)
$GptZincPrompts = @("aromatic_core", "amino_aliphatic", "polar_ether", "carbonyl_anchor", "heterocycle_friendly", "sulfur_phosphorus", "extended_polar", "fallback_generic")

$SmileyTemps = @("1.0", "0.7")
$SmcTemps = @("1.2", "1.0")

# Convert arrays to space-separated strings for Python arguments
$SmileyPromptArgs = $SmileyPrompts -join " "
$GptZincPromptArgs = $GptZincPrompts -join " "
$SmileyTempArgs = $SmileyTemps -join " "
$SmcTempArgs = $SmcTemps -join " "

# Write-Host "[1/4] Preparing ZINC data splits..." -ForegroundColor Green
# # python -m src.data_prep --input "$DataInput"

Write-Host "[2/4] Running baseline GPT2-Zinc generation with strategic prompts..." -ForegroundColor Green
python -m src.baseline_generate `
  --prompts $GptZincPromptArgs `
  --n 1000 `
  --temperature 1.0 `
  --top_p 0.9 `
  --batch_size 256 `
  --out_csv results/baseline_results.csv `
  --summary_csv results/baseline_summary.csv

Write-Host "[3/4] Running SmileyLlama generation with instruction prompts..." -ForegroundColor Green
python -m src.smiley_generate `
  --prompts $SmileyPromptArgs `
  --n 1000 `
  --temperatures $SmileyTempArgs `
  --top_p 0.9 `
  --batch_size 128 `
  --out_csv results/smiley_results.csv `
  --summary_csv results/smiley_summary.csv

Write-Host "[4/4] Running GenLM SMC generation with strategic prompts..." -ForegroundColor Green
python -m src.smc_generate `
  --prompts $GptZincPromptArgs `
  --n 100 `
  --temperatures $SmcTempArgs `
  --top_p 0.9 `
  --particles 10 `
  --out_csv results/smc_results.csv `
  --summary_csv results/smc_summary.csv

Write-Host "All experiments completed. Outputs written under results/." -ForegroundColor Green
