# PowerShell script to run simplified constraint-based experiments
# Simplified pipeline: 2 models × 3 constraint levels = 6 experiments

$ErrorActionPreference = "Stop"

$ConstraintLevels = @("loose", "tight", "ultra_tight")
$PropertyRanges = "data/train_property_ranges.json"
$Dataset = "Combined"
$N = 1000

# Record overall start time
$ScriptStartTime = Get-Date

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Simplified Constraint-Based Experiments" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Constraint levels: $($ConstraintLevels -join ', ')" -ForegroundColor White
Write-Host "Property ranges: $PropertyRanges" -ForegroundColor White
Write-Host "Dataset: $Dataset" -ForegroundColor White
Write-Host ""

# Check if property ranges file exists
if (-not (Test-Path $PropertyRanges)) {
    Write-Host "Error: Property ranges file not found: $PropertyRanges" -ForegroundColor Red
    Write-Host "Please run: python -m src.analyze_train_data" -ForegroundColor Yellow
    exit 1
}

# 1. GPT2-Zinc Baseline (multi-prefix, constraint-filtered)
Write-Host "[1/2] Running GPT2-Zinc baseline with multi-prefix constraint filtering..." -ForegroundColor Green
foreach ($level in $ConstraintLevels) {
    Write-Host "  - Constraint level: $level" -ForegroundColor Yellow
    python -m src.baseline_generate_constraint `
        --constraint-level $level `
        --property-ranges $PropertyRanges `
        --dataset $Dataset `
        --n $N `
        --temperature 1.0 `
        --top_p 0.9 `
        --batch_size 256 `
        --out-csv "results/baseline_${level}_results.csv" `
        --summary-csv "results/baseline_${level}_summary.csv"
}

# Combine baseline results
Write-Host "  - Combining baseline results..." -ForegroundColor Yellow
python -c "import pandas as pd; import glob; files = sorted(glob.glob('results/baseline_*_results.csv')); dfs = [pd.read_csv(f) for f in files] if files else []; pd.concat(dfs, ignore_index=True).to_csv('results/baseline_results.csv', index=False) if dfs else None; print(f'  Combined {len(files)} files into baseline_results.csv') if files else print('  No files to combine')"

# 2. GPT2-Zinc+SMC (SMC-guided generation)
Write-Host ""
Write-Host "[2/3] Running GPT2-Zinc+SMC with SMC-guided generation..." -ForegroundColor Green
foreach ($level in $ConstraintLevels) {
    Write-Host "  - Constraint level: $level" -ForegroundColor Yellow
    python -m src.smc_generate_constraint `
        --constraint-level $level `
        --property-ranges $PropertyRanges `
        --dataset $Dataset `
        --n $N `
        --particles 50 `
        --ess-threshold 0.3 `
        --temperature 0.7 `
        --top_p 0.9 `
        --max-new-tokens 60 `
        --top-k 30 `
        --out-csv "results/smc_${level}_results.csv" `
        --summary-csv "results/smc_${level}_summary.csv"
}

# Combine SMC results
Write-Host "  - Combining SMC results..." -ForegroundColor Yellow
python -c "import pandas as pd; import glob; files = sorted(glob.glob('results/smc_*_results.csv')); dfs = [pd.read_csv(f) for f in files] if files else []; pd.concat(dfs, ignore_index=True).to_csv('results/smc_results.csv', index=False) if dfs else None; print(f'  Combined {len(files)} files into smc_results.csv') if files else print('  No files to combine')"

# 3. SmileyLlama (constraint variants)
Write-Host ""
Write-Host "[3/3] Running SmileyLlama with constraint variants..." -ForegroundColor Green
foreach ($level in $ConstraintLevels) {
    Write-Host "  - Constraint level: $level" -ForegroundColor Yellow
    python -m src.smiley_generate_constraint `
        --constraint-level $level `
        --property-ranges $PropertyRanges `
        --dataset $Dataset `
        --base-prompt mw_logp_rotb `
        --n $N `
        --temperature 1.0 `
        --top_p 0.9 `
        --batch_size 128 `
        --quantize `
        --out-csv "results/smiley_${level}_results.csv" `
        --summary-csv "results/smiley_${level}_summary.csv"
}

# Combine SmileyLlama results
Write-Host "  - Combining SmileyLlama results..." -ForegroundColor Yellow
python -c "import pandas as pd; import glob; files = sorted(glob.glob('results/smiley_*_results.csv')); dfs = [pd.read_csv(f) for f in files] if files else []; pd.concat(dfs, ignore_index=True).to_csv('results/smiley_results.csv', index=False) if dfs else None; print(f'  Combined {len(files)} files into smiley_results.csv') if files else print('  No files to combine')"

# Calculate total runtime
$ScriptEndTime = Get-Date
$ScriptRuntime = $ScriptEndTime - $ScriptStartTime
$ScriptRuntimeMin = [math]::Floor($ScriptRuntime.TotalMinutes)
$ScriptRuntimeSec = $ScriptRuntime.Seconds

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Experiments completed!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total runtime: ${ScriptRuntimeMin}m ${ScriptRuntimeSec}s" -ForegroundColor White
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor White
Write-Host "  - results/baseline_results.csv (combined - 3 constraint levels)" -ForegroundColor Gray
Write-Host "  - results/smc_results.csv (combined - 3 constraint levels)" -ForegroundColor Gray
Write-Host "  - results/smiley_results.csv (combined - 3 constraint levels)" -ForegroundColor Gray
Write-Host "  - Individual files: results/baseline_*_results.csv, results/smc_*_results.csv, results/smiley_*_results.csv" -ForegroundColor Gray
Write-Host "  - Summary files: results/*_summary.csv (include runtime)" -ForegroundColor Gray
Write-Host ""
Write-Host "Timing summary:" -ForegroundColor White
python -c "
import pandas as pd
import glob
summary_files = sorted(glob.glob('results/*_summary.csv'))
if summary_files:
    dfs = []
    for f in summary_files:
        df = pd.read_csv(f)
        if 'Runtime_seconds' in df.columns:
            dfs.append(df[['Model', 'ConstraintLevel', 'Runtime_seconds', 'Runtime_formatted']])
    if dfs:
        timing_df = pd.concat(dfs, ignore_index=True)
        print('')
        for _, row in timing_df.iterrows():
            print(f\"  {row['Model']:20s} {row['ConstraintLevel']:12s} {row['Runtime_formatted']:8s} ({row['Runtime_seconds']:.1f}s)\")
        total_time = timing_df['Runtime_seconds'].sum()
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        print(f\"\n  {'Total generation time:':20s} {total_min}m {total_sec}s ({total_time:.1f}s)\")
    else:
        print('  No timing data found in summary files')
else:
    print('  No summary files found')
"
Write-Host ""
Write-Host "To evaluate results, run:" -ForegroundColor Cyan
Write-Host "  python -m src.evaluate" -ForegroundColor Yellow
Write-Host ""
Write-Host "To generate plots, run:" -ForegroundColor Cyan
Write-Host "  python -m src.plots" -ForegroundColor Yellow
