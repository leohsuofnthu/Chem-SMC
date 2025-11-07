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

# 1. GPT2-Zinc Baseline (multi-prefix, constraint-filtered)
echo "[1/2] Running GPT2-Zinc baseline with multi-prefix constraint filtering..."
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

# 2. SmileyLlama (constraint variants)
echo ""
echo "[2/2] Running SmileyLlama with constraint variants..."
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


echo ""
echo "=========================================="
echo "Experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/baseline_results.csv (combined - 3 constraint levels)"
echo "  - results/smiley_results.csv (combined - 3 constraint levels)"
echo "  - Individual files: results/baseline_*_results.csv, results/smiley_*_results.csv"
echo ""
echo "To evaluate results, run:"
echo "  python -m src.evaluate"
echo ""
echo "To generate plots, run:"
echo "  python -m src.plots"

