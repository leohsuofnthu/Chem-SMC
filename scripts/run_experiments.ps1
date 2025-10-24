# PowerShell script to run experiments
param(
    [string]$DataInput = "data/zinc250k.csv"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Define arrays
$Prompts = @("lipinski", "mw_logp_rotb", "tpsa_fsp3", "druglike_qed")
$SmileyTemps = @("1.0", "0.7")
$SmcTemps = @("1.0", "0.7")

# Convert arrays to space-separated strings for Python arguments
$PromptArgs = $Prompts -join " "
$SmileyTempArgs = $SmileyTemps -join " "
$SmcTempArgs = $SmcTemps -join " "

# Write-Host "[1/4] Preparing ZINC data splits..." -ForegroundColor Green
# # python -m src.data_prep --input "$DataInput"

# Write-Host "[2/4] Running baseline GPT2-Zinc generation..." -ForegroundColor Green
# # python -m src.baseline_generate `
# #   --prompts $PromptArgs `
# #   --n 1000 `
# #   --temperature 1.0 `
# #   --top_p 0.9 `
# #   --batch_size 256 `
# #   --out_csv results/baseline_results.csv `
# #   --summary_csv results/baseline_summary.csv

# Write-Host "[3/4] Running SmileyLlama generation..." -ForegroundColor Green
# python -m src.smiley_generate `
#   --prompts $PromptArgs `
#   --n 1000 `
#   --temperatures $SmileyTempArgs `
#   --top_p 0.9 `
#   --batch_size 128 `
#   --out_csv results/smiley_results.csv `
#   --summary_csv results/smiley_summary.csv

Write-Host "[4/4] Running GenLM SMC generation with advanced grammar enforcement..." -ForegroundColor Green
python -m src.smc_generate `
  --prompts $Prompts `
  --n 10 `
  --temperatures $SmcTemps `
  --top_p 0.9 `
  --particles 10 `
  --out_csv results/smc_results.csv `
  --summary_csv results/smc_summary.csv

Write-Host "All experiments completed. Outputs written under results/." -ForegroundColor Green
