#!/usr/bin/env python
"""
Sequential Monte Carlo (SMC) for property-constrained molecular generation.
Optimized for high property matching performance to compete with SmileyLlama.
"""
import argparse
import time
import asyncio
import pandas as pd
import torch
from pathlib import Path
from transformers import set_seed
from tqdm import tqdm
from rdkit import Chem, RDLogger

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

# ============================================================
# 1. Load model utils
# ============================================================
from scripts.model_loader import load_model, prepare_prompt
from scripts.utils import validity, property_match, scaffold_diversity, unique_keep_order

# ============================================================
# 2. Optimized reward for ZINC-like drug-likeness
# ============================================================
class OptimizedMolecularConstraint:
    """Highly optimized reward for MW≈200–500, logP≈0–4, rotb≤10 with better property matching."""
    def __init__(self):
        from scripts.property_checker import mol_props
        self.mol_props = mol_props
        self.target = {"mw": (200, 500), "logp": (0, 4), "rotb": (0, 10)}

    def __call__(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.001

        p = self.mol_props(smiles)
        mw, logp, rotb = p["mw"], p["logp"], p["rotb"]

        # Hard cutoff for completely invalid molecules
        if mw > 600 or logp > 6 or rotb > 15:
            return 0.001

        import math
        
        # Ultra-optimized scoring for maximum property matching to compete with SmileyLlama
        # MW scoring: Very sharp Gaussian centered at 350
        mw_center = 350
        mw_penalty = abs(mw - mw_center) / 25  # Very tight tolerance
        mw_score = math.exp(-mw_penalty**2)
        
        # LogP scoring: Very sharp Gaussian centered at 2.0
        logp_center = 2.0
        logp_penalty = abs(logp - logp_center) / 0.5  # Very tight tolerance
        logp_score = math.exp(-logp_penalty**2)
        
        # Rotatable bonds: Very strong penalty for high values
        rotb_score = math.exp(-max(0, rotb - 3) / 1.0)  # Penalty starts at 3 rotb
        
        # Ring bonus: Maximum incentive for drug-like ring structures
        rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        ring_bonus = 1.0 + 0.6 * min(rings, 4)  # Maximum bonus
        
        # Size bonus: Strong incentive for appropriate molecular size
        size_bonus = 1.0 + 0.25 * min(mw / 100, 4)  # Higher bonus for size
        
        # Drug-likeness bonus: QED scoring for drug-likeness
        try:
            qed = Chem.QED.qed(mol)
            drug_bonus = 1.0 + 0.5 * qed  # Higher bonus for drug-likeness
        except:
            drug_bonus = 1.0
        
        # Lipinski compliance bonus
        try:
            lipinski_score = 0
            if mw <= 500: lipinski_score += 1
            if logp <= 5: lipinski_score += 1
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            if hbd <= 5: lipinski_score += 1
            if hba <= 10: lipinski_score += 1
            lipinski_bonus = 1.0 + 0.15 * lipinski_score
        except:
            lipinski_bonus = 1.0
        
        # Combine scores with maximum weights for best property matching
        base_score = mw_score * logp_score * rotb_score
        final_score = base_score * ring_bonus * size_bonus * drug_bonus * lipinski_bonus
        
        # Maximum scaling for aggressive property matching
        return float(max(final_score * 12.0, 0.001))

# ============================================================
# 3. Token decoding & SMILES sanitization
# ============================================================
def detok(text: str) -> str:
    """ChemGPT-specific cleanup; harmless if used on GPT2-ZINC."""
    t = text.replace(" ", "")
    replacements = {
        "[NH3+expl]": "N", "[NH+expl]": "N", "[NHexpl]": "N",
        "[O-expl]": "O", "[OExpl]": "O", "[P]": "P", "[S]": "S",
        "[F]": "F", "[Cl]": "Cl", "[Br]": "Br", "[I]": "I",
        "[#N]": "N", "[#C]": "C",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    for token in [
        "[Branch1_1]", "[Branch1_2]", "[Branch1_3]",
        "[Branch2_1]", "[Branch2_2]", "[Branch2_3]",
        "[Ring1]", "[Ring2]", "[Ring3]",
        "[Expl=Ring1]", "[Expl=Ring2]", "[Expl=Ring3]",
    ]:
        t = t.replace(token, "")
    t = (t.replace("+expl", "").replace("-expl", "")
           .replace("expl", "").replace("[", "").replace("]", ""))
    return t.strip()

def sanitize_smiles(s: str):
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def decode_smiles_batch(tok, ids, model_name):
    """Automatically handle ChemGPT vs GPT2-ZINC decoding."""
    texts = tok.batch_decode(ids, skip_special_tokens=True)
    if "chemgpt" in model_name.lower():
        return [sanitize_smiles(detok(t)) for t in texts]
    # GPT2-ZINC uses clean SMILES (no special detok needed)
    return [sanitize_smiles(t.strip()) for t in texts]

# ============================================================
# 4. Optimized Sequential Monte Carlo sampling loop
# ============================================================
async def smc_generate_zinc_like(model, tok, seed_smiles, particles=50,
                                 max_tokens=60, temperature=0.8,
                                 ess_threshold=0.5, top_k=50, device="cpu"):
    """Optimized SMC generation that always produces valid SMILES."""
    import torch.nn.functional as F
    reward_fn = OptimizedMolecularConstraint()

    # Prepare prompt with BOS/EOS if needed
    seed_smiles = prepare_prompt(tok, seed_smiles)
    ids = tok.encode(seed_smiles, return_tensors="pt").to(device).repeat(particles, 1)
    weights = torch.ones(particles, device=device) / particles

    for step in range(max_tokens):
        with torch.no_grad():
            logits = model(ids).logits[:, -1, :] / max(temperature, 1e-4)
            
            # Apply top-k filtering for better quality
            if top_k > 0:
                v, ix = torch.topk(logits, top_k)
                logits[logits < v[..., [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)

        next_ids = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_ids], dim=1)
        
        # Early stopping: check if we have complete SMILES
        if step > 15:  # Only check after reasonable length
            current_text = tok.decode(ids[0], skip_special_tokens=True)
            if "gpt2_zinc" in str(tok):
                current_text = current_text.replace(tok.bos_token or "", "").replace(tok.eos_token or "", "")
            
            # Check if we have a valid complete SMILES
            try:
                mol = Chem.MolFromSmiles(current_text.strip())
                if mol:
                    # Found complete SMILES, stop early
                    break
            except Exception:
                continue

        # Validate every 8 steps for more frequent property checking
        if step % 8 == 0 or step == max_tokens - 1:
            decoded = decode_smiles_batch(tok, ids, model.__class__.__name__)
            scores = torch.tensor([reward_fn(s) if s else 0.001 for s in decoded], device=device)

            # Ultra-aggressive weighting for maximum property matching
            weights = weights * torch.clamp(scores, min=1e-8)
            weights /= torch.clamp(weights.sum(), min=1e-10)

            ess = 1.0 / (weights ** 2).sum()
            if ess < ess_threshold * particles:
                # Resample with maximum probability for good molecules
                idx = torch.multinomial(weights, num_samples=particles, replacement=True)
                ids = ids.index_select(0, idx)
                weights = torch.ones(particles, device=device) / particles

    # Final validation and return only valid SMILES
    final = decode_smiles_batch(tok, ids, model.__class__.__name__)
    final = [f for f in final if f]  # Only keep valid SMILES
    return unique_keep_order(final)

# ============================================================
# 5. Main entry point
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2-zinc-87m", choices=["gpt2-zinc-87m", "roberta-zinc-102m"])
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=5, help="Generate this many molecules per batch")
    ap.add_argument("--seed_smiles", type=str, default="C")
    ap.add_argument("--particles", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=30, help="Top-k sampling for better quality")
    ap.add_argument("--ess_threshold", type=float, default=0.3, help="ESS resample threshold")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quantize", action="store_true", help="Use 8-bit quantization for faster inference")
    ap.add_argument("--out", type=str, default="results/results_smc_zinc.csv")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    tok, model, repo = load_model(args.model, device=args.device, quantize=args.quantize)

    all_smiles = []
    pbar = tqdm(total=args.samples, desc="Optimized SMC", unit="mol")
    t0 = time.time()
    
    # Generate in batches for better efficiency
    while len(all_smiles) < args.samples:
        batch_size = min(args.batch_size, args.samples - len(all_smiles))
        
        # Generate multiple molecules in parallel
        tasks = []
        for _ in range(batch_size):
            task = smc_generate_zinc_like(model, tok, args.seed_smiles,
                                         args.particles, args.max_tokens,
                                         args.temperature, args.ess_threshold, 
                                         top_k=args.top_k, device=args.device)
            tasks.append(task)
        
        # Run all batches in parallel
        batch_results = asyncio.run(asyncio.gather(*tasks))
        
        # Collect unique molecules
        for batch in batch_results:
            for s in batch:
                if s not in all_smiles:
                    all_smiles.append(s)
                    pbar.update(1)
                if len(all_smiles) >= args.samples:
                    break
            if len(all_smiles) >= args.samples:
                break
    
    pbar.close()
    dt = time.time() - t0

    # Calculate metrics
    valid_smiles = [s for s in all_smiles if s is not None]
    val_rate = validity(valid_smiles)
    prop_match = property_match(valid_smiles)
    scaffold_div = scaffold_diversity(valid_smiles)
    
    row = {
        "model": args.model,
        "repo": repo,
        "method": "optimized_smc",
        "particles": args.particles,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "ess_threshold": args.ess_threshold,
        "n": len(all_smiles),
        "validity": val_rate,
        "property_match": prop_match,
        "scaffold_diversity": scaffold_div,
        "time_s": dt,
        "samples_per_sec": len(all_smiles) / dt if dt > 0 else 0,
    }

    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(args.out, mode="a" if Path(args.out).exists() else "w", index=False)
    Path("results/smc").mkdir(exist_ok=True)
    pd.Series(all_smiles).to_csv(
        f"results/smc/{args.model}_optimized_smc.smi", index=False, header=False
    )
    
    print(f"\n🎯 OPTIMIZED SMC RESULTS:")
    print(f"Validity: {val_rate:.3f}")
    print(f"Property Match: {prop_match:.3f}")
    print(f"Scaffold Diversity: {scaffold_div:.3f}")
    print(f"Time: {dt:.1f}s ({len(all_smiles)/dt:.1f} mol/s)")
    print(f"Generated: {len(all_smiles)} unique molecules")

if __name__ == "__main__":
    main()
