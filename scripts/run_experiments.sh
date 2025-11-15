#!/usr/bin/env bash

set -euo pipefail

# Simplified experiment pipeline: Run 3 constraint levels per model
# - Gradual constraints (loose/tight/ultra_tight): upper-bound only, progressive (1→2→3 conditions)
#   - loose: <= 500 MW (1 condition, covers most drug-like molecules)
#   - tight: <= 400 MW + <= 4 logP (2 conditions, typical drug-like)
#   - ultra_tight: <= 350 MW + <= 3.5 logP + <= 8 RotB (3 conditions, stricter drug-like)
# - Range-based constraints (loose/tight/ultra_tight): percentile-based ranges
#   - loose: 5th-95th percentile (MW ~233-577, logP ~-0.07-5.73, RotB ~2-10)
#   - tight: 25th-75th percentile (MW ~304-419, logP ~1.88-4.00, RotB ~3-6)
#   - ultra_tight: 40th-60th percentile (MW ~336-372, logP ~2.58-3.36, RotB ~4-5)

GRADUAL_LEVELS=(loose tight ultra_tight)
RANGE_LEVELS=(loose tight ultra_tight)
PROPERTY_RANGES="data/train_property_ranges.json"
DATASET="Combined"
N=1000

# Record overall start time
SCRIPT_START_TIME=$(date +%s)

echo "=========================================="
echo "Complete Constraint-Based Experiments"
echo "=========================================="
echo "Gradual constraint levels: ${GRADUAL_LEVELS[@]}"
echo "Range-based constraint levels: ${RANGE_LEVELS[@]}"
echo "Property ranges: $PROPERTY_RANGES"
echo "Dataset: $DATASET"
echo "Total experiments: 15 (5 model types × 3 constraint levels)"
echo ""

# Check if property ranges file exists
if [ ! -f "$PROPERTY_RANGES" ]; then
    echo "Error: Property ranges file not found: $PROPERTY_RANGES"
    echo "Please run: python -m src.analyze_train_data"
    exit 1
fi

# 1. GPT2-Zinc Baseline (multi-prefix, constraint evaluation)
# Baseline uses legacy percentile-based constraints (loose/tight/ultra_tight)
echo "[1/5] Running GPT2-Zinc baseline with multi-prefix constraint evaluation..."
echo "  Type: Range-based constraints (percentile-based)"
for level in "${RANGE_LEVELS[@]}"; do
    echo "    - Constraint level: $level (range-based)"
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
    print(f'    Combined {len(files)} files into baseline_results.csv')
"

# 2. GPT2-Zinc+SMC with gradual constraints (upper-bound only)
# SMC uses gradual constraints (loose/tight/ultra_tight) with dynamic reward scaling
echo ""
echo "[2/5] Running GPT2-Zinc+SMC with gradual constraints (upper-bound only)..."
echo "  Type: Gradual constraints (upper-bound)"
for level in "${GRADUAL_LEVELS[@]}"; do
    echo "    - Constraint level: $level (gradual constraints)"
    python -m src.smc_generate_constraint \
        --constraint-level "$level" \
        --use-gradual-constraints \
        --property-ranges "$PROPERTY_RANGES" \
        --dataset "$DATASET" \
        --n "$N" \
        --particles 20 \
        --ess-threshold 0.3 \
        --temperature 1.0 \
        --top_p 0.9 \
        --max-new-tokens 60 \
        --top-k 30 \
        --out-csv "results/smc_gradual_${level}_results.csv" \
        --summary-csv "results/smc_gradual_${level}_summary.csv"
done

# Combine SMC gradual results
echo "  - Combining SMC gradual results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smc_gradual_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smc_gradual_results.csv', index=False)
    print(f'    Combined {len(files)} files into smc_gradual_results.csv')
"

# 3. GPT2-Zinc+SMC with range-based constraints (percentile-based)
# SMC uses range-based constraints (loose/tight/ultra_tight) with enhanced reward function
echo ""
echo "[3/5] Running GPT2-Zinc+SMC with range-based constraints (percentile-based)..."
echo "  Type: Range-based constraints (percentile-based)"
for level in "${RANGE_LEVELS[@]}"; do
    echo "    - Constraint level: $level (range-based)"
    python -m src.smc_generate_constraint \
        --constraint-level "$level" \
        --no-gradual-constraints \
        --property-ranges "$PROPERTY_RANGES" \
        --dataset "$DATASET" \
        --n "$N" \
        --particles 20 \
        --ess-threshold 0.3 \
        --temperature 1.0 \
        --top_p 0.9 \
        --max-new-tokens 60 \
        --top-k 30 \
        --out-csv "results/smc_range_${level}_results.csv" \
        --summary-csv "results/smc_range_${level}_summary.csv"
done

# Combine SMC range-based results
echo "  - Combining SMC range-based results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smc_range_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smc_range_results.csv', index=False)
    print(f'    Combined {len(files)} files into smc_range_results.csv')
"

# Combine all SMC results
echo "  - Combining all SMC results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smc_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smc_all_results.csv', index=False)
    print(f'    Combined {len(files)} files into smc_all_results.csv')
"

# 4. SmileyLlama with gradual constraints (upper-bound only)
# SmileyLlama uses gradual constraints (loose/tight/ultra_tight) - SmileyLlama-compatible format
echo ""
echo "[4/5] Running SmileyLlama with gradual constraints (upper-bound only)..."
echo "  Type: Gradual constraints (upper-bound)"
for level in "${GRADUAL_LEVELS[@]}"; do
    echo "    - Constraint level: $level (gradual constraints)"
    python -m src.smiley_generate_constraint \
        --constraint-level "$level" \
        --use-gradual-constraints \
        --property-ranges "$PROPERTY_RANGES" \
        --dataset "$DATASET" \
        --base-prompt mw_logp_rotb \
        --n "$N" \
        --temperature 1.0 \
        --top_p 0.9 \
        --batch_size 50 \
        --seed 42 \
        --quantize \
        --out-csv "results/smiley_gradual_${level}_results.csv" \
        --summary-csv "results/smiley_gradual_${level}_summary.csv"
done

# Combine SmileyLlama gradual results
echo "  - Combining SmileyLlama gradual results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smiley_gradual_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smiley_gradual_results.csv', index=False)
    print(f'    Combined {len(files)} files into smiley_gradual_results.csv')
"

# 5. SmileyLlama with range-based constraints (percentile-based)
# SmileyLlama uses range-based constraints (loose/tight/ultra_tight) with range-to-prompt conversion
echo ""
echo "[5/5] Running SmileyLlama with range-based constraints (percentile-based)..."
echo "  Type: Range-based constraints (percentile-based)"
for level in "${RANGE_LEVELS[@]}"; do
    echo "    - Constraint level: $level (range-based)"
    python -m src.smiley_generate_constraint \
        --constraint-level "$level" \
        --no-gradual-constraints \
        --property-ranges "$PROPERTY_RANGES" \
        --dataset "$DATASET" \
        --base-prompt mw_logp_rotb \
        --n "$N" \
        --temperature 1.0 \
        --top_p 0.9 \
        --batch_size 50 \
        --seed 42 \
        --quantize \
        --out-csv "results/smiley_range_${level}_results.csv" \
        --summary-csv "results/smiley_range_${level}_summary.csv"
done

# Combine SmileyLlama range-based results
echo "  - Combining SmileyLlama range-based results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smiley_range_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smiley_range_results.csv', index=False)
    print(f'    Combined {len(files)} files into smiley_range_results.csv')
"

# Combine all SmileyLlama results
echo "  - Combining all SmileyLlama results..."
python -c "
import pandas as pd
import glob
files = sorted(glob.glob('results/smiley_*_results.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('results/smiley_all_results.csv', index=False)
    print(f'    Combined {len(files)} files into smiley_all_results.csv')
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
echo "  - results/baseline_results.csv (combined - 3 constraint levels, range-based)"
echo "  - results/smc_gradual_results.csv (combined - 3 constraint levels, gradual)"
echo "  - results/smc_range_results.csv (combined - 3 constraint levels, range-based)"
echo "  - results/smc_all_results.csv (all SMC results combined)"
echo "  - results/smiley_gradual_results.csv (combined - 3 constraint levels, gradual)"
echo "  - results/smiley_range_results.csv (combined - 3 constraint levels, range-based)"
echo "  - results/smiley_all_results.csv (all SmileyLlama results combined)"
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

