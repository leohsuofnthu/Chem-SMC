#!/usr/bin/env bash

set -euo pipefail

# Simplified experiment pipeline: Run 3 constraint levels per model
# - loose: 5th-95th percentile (baseline)
# - tight: 25th-75th percentile  
# - ultra_tight: 40th-60th percentile

CONSTRAINT_LEVELS=(loose tight ultra_tight)
PROPERTY_RANGES="data/train_property_ranges.json"
DATASET="Combined"
N=1000

# Record overall start time
SCRIPT_START_TIME=$(date +%s)

echo "=========================================="
echo "Simplified Constraint-Based Experiments"
echo "=========================================="
echo "Constraint levels: ${CONSTRAINT_LEVELS[@]}"
echo "Property ranges: $PROPERTY_RANGES"
echo "Dataset: $DATASET"
echo ""

# Check if property ranges file exists
if [ ! -f "$PROPERTY_RANGES" ]; then
    echo "Error: Property ranges file not found: $PROPERTY_RANGES"
    echo "Please run: python -m src.analyze_train_data"
    exit 1
fi

# 1. GPT2-Zinc Baseline (multi-prefix, constraint evaluation)
echo "[1/2] Running GPT2-Zinc baseline with multi-prefix constraint evaluation..."
for level in "${CONSTRAINT_LEVELS[@]}"; do
    echo "  - Constraint level: $level"
    python -m src.baseline_generate_constraint \
        --constraint-level "$level" \
        --property-ranges "$PROPERTY_RANGES" \
        --dataset "$DATASET" \
        --n "$N" \
        --temperature 1.0 \
        --top_p 0.9 \
        --batch_size 256 \
        --out-csv "results/baseline_${level}_results.csv" \
        --summary-csv "results/baseline_${level}_summary.csv"
done

# Combine baseline results
echo "  - Combining baseline results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/baseline_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/baseline_results.csv', index=False)
    print(f'  Combined {len(files)} files into baseline_results.csv')
"

# 2. GPT2-Zinc+SMC (SMC-guided generation)
echo ""
echo "[2/3] Running GPT2-Zinc+SMC with SMC-guided generation..."
for level in "${CONSTRAINT_LEVELS[@]}"; do
    echo "  - Constraint level: $level"
    python -m src.smc_generate_constraint \
        --constraint-level "$level" \
        --property-ranges "$PROPERTY_RANGES" \
        --dataset "$DATASET" \
        --n "$N" \
        --particles 50 \
        --ess-threshold 0.3 \
        --temperature 0.7 \
        --top_p 0.9 \
        --max-new-tokens 60 \
        --top-k 30 \
        --out-csv "results/smc_${level}_results.csv" \
        --summary-csv "results/smc_${level}_summary.csv"
done

# Combine SMC results
echo "  - Combining SMC results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smc_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smc_results.csv', index=False)
    print(f'  Combined {len(files)} files into smc_results.csv')
"

# 3. SmileyLlama (constraint variants)
echo ""
echo "[3/3] Running SmileyLlama with constraint variants..."
for level in "${CONSTRAINT_LEVELS[@]}"; do
    echo "  - Constraint level: $level"
    python -m src.smiley_generate_constraint \
        --constraint-level "$level" \
        --property-ranges "$PROPERTY_RANGES" \
        --dataset "$DATASET" \
        --base-prompt mw_logp_rotb \
        --n "$N" \
        --temperature 1.0 \
        --top_p 0.9 \
        --batch_size 128 \
        --quantize \
        --out-csv "results/smiley_${level}_results.csv" \
        --summary-csv "results/smiley_${level}_summary.csv"
done

# Combine SmileyLlama results
echo "  - Combining SmileyLlama results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smiley_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smiley_results.csv', index=False)
    print(f'  Combined {len(files)} files into smiley_results.csv')
"


# Calculate total runtime
SCRIPT_END_TIME=$(date +%s)
SCRIPT_RUNTIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
SCRIPT_RUNTIME_MIN=$((SCRIPT_RUNTIME / 60))
SCRIPT_RUNTIME_SEC=$((SCRIPT_RUNTIME % 60))

echo ""
echo "=========================================="
echo "Experiments completed!"
echo "=========================================="
echo ""
echo "Total runtime: ${SCRIPT_RUNTIME_MIN}m ${SCRIPT_RUNTIME_SEC}s"
echo ""
echo "Results saved to:"
echo "  - results/baseline_results.csv (combined - 3 constraint levels)"
echo "  - results/smc_results.csv (combined - 3 constraint levels)"
echo "  - results/smiley_results.csv (combined - 3 constraint levels)"
echo "  - Individual files: results/baseline_*_results.csv, results/smc_*_results.csv, results/smiley_*_results.csv"
echo "  - Summary files: results/*_summary.csv (include runtime)"
echo ""
echo "Timing summary:"
python -c "
import pandas as pd
import glob
import sys

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

echo ""
echo "To evaluate results, run:"
echo "  python -m src.evaluate"
echo ""
echo "To generate plots, run:"
echo "  python -m src.plots"

