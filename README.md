# chem-smc

Two complementary strategies for controlled molecule generation:

1. **GPT2-Zinc-87M + Sequential Monte Carlo (SMC)** via GenLM’s AWRS controller  
2. **SmileyLlama-8B** instruction following with prompt-based constraints

Both approaches share the same prompt set and are evaluated on a ZINC-derived benchmark with identical metrics and visualisations.

## Quickstart
```bash
# 1) Environment
conda create -n chem-smc python=3.10 -y
conda activate chem-smc
pip install -r requirements.txt

# 2) Optional: place ZINC-250k CSV/SMI under data/
#    (or pass --input to data prep)

# 3) End-to-end experiment
jupyter lab  # open notebooks/run_experiment.ipynb
```

The notebook executes the pipeline in order:

1. `src.data_prep` – prepare reference/eval splits + percentile ranges  
2. `src.baseline_generate` – GPT2-Zinc baseline sampling (T = 1.0, 0.7)  
3. `src.smc_generate` – GenLM SMC decoding with 10 particles  
4. `src.smiley_generate` – SmileyLlama generation (bnb int4 by default)  
5. `src.evaluate` – compute validity, distinctness, QED, diversity, KL, adherence  
6. `src.plots` – histograms, QED overlay, bar charts (saved to `figures/`)

## Command-line usage
Every stage is exposed as a module:
```bash
python -m src.data_prep --input data/zinc250k.csv
python -m src.baseline_generate --temperatures 1.0 0.7
python -m src.smc_generate --temperatures 1.0 0.7
python -m src.smiley_generate --temperatures 1.0 --device cuda --quantize
python -m src.evaluate --temperature 1.0
python -m src.plots --temperature 1.0
```

Outputs land in `results/` (CSV tables) and `figures/` (PNGs) with a unified schema:
`SMILES, Valid, QED, MW, logP, RotB, TPSA, HBD, HBA, Adherence, Weight, Model, Prompt, Temperature, TopP`.

## Notes
- SmileyLlama is heavy; 4-bit quantisation keeps GPU usage ≈4 GB. Pass `--no_quantize` to force full precision.  
- GenLM SMC uses a property-aware potential (validity + QED + range adherence) with ESS-based resampling.  
- Adjust prompts or property ranges via `src/utils.py` (`PROMPTS` list).
