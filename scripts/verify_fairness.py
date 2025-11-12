#!/usr/bin/env python3
"""
Verification script to ensure SMC and SmileyLlama use identical constraints.
Run this to verify experiments are fair.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_property_ranges,
    create_constraint_variant,
    GPT_ZINC_PROMPTS,
    SMILEY_PROMPT_MAP,
)

def verify_constraints():
    """Verify that SMC and SmileyLlama use identical constraints."""
    print("=" * 70)
    print("Fairness Verification: SMC vs SmileyLlama Constraints")
    print("=" * 70)
    print()
    
    property_ranges_path = "data/train_property_ranges.json"
    dataset = "Combined"
    constraint_levels = ["loose", "tight", "ultra_tight"]
    
    all_match = True
    
    for level in constraint_levels:
        print(f"Checking constraint level: {level}")
        print("-" * 70)
        
        # Load constraint ranges
        constraint_ranges = load_property_ranges(property_ranges_path, dataset, level)
        print(f"Loaded ranges: {constraint_ranges}")
        
        # Create constraint specs using same function
        smc_base = GPT_ZINC_PROMPTS[0]  # SMC uses first GPT-Zinc prompt
        smiley_base = SMILEY_PROMPT_MAP["mw_logp_rotb"]  # SmileyLlama uses mw_logp_rotb
        
        smc_spec = create_constraint_variant(smc_base, constraint_ranges, tightness=level)
        smiley_spec = create_constraint_variant(smiley_base, constraint_ranges, tightness=level)
        
        # Compare constraints (the evaluation part)
        smc_constraints = smc_spec.constraints
        smiley_constraints = smiley_spec.constraints
        
        print(f"SMC constraints:      {smc_constraints}")
        print(f"SmileyLlama constraints: {smiley_constraints}")
        
        # Verify they match
        if smc_constraints == smiley_constraints:
            print("[OK] Constraints MATCH - Fair evaluation!")
        else:
            print("[ERROR] Constraints DO NOT MATCH - UNFAIR!")
            all_match = False
            # Show differences
            for key in set(list(smc_constraints.keys()) + list(smiley_constraints.keys())):
                if smc_constraints.get(key) != smiley_constraints.get(key):
                    print(f"  Difference in {key}:")
                    print(f"    SMC: {smc_constraints.get(key)}")
                    print(f"    SmileyLlama: {smiley_constraints.get(key)}")
        
        # Show prompt text (different, but doesn't affect evaluation)
        print(f"\nPrompt text (for reference, doesn't affect evaluation):")
        print(f"  SMC: '{smc_spec.text[:80]}...' (not used)")
        print(f"  SmileyLlama: '{smiley_spec.text[:80]}...' (used for generation)")
        print()
    
    print("=" * 70)
    if all_match:
        print("[SUCCESS] ALL CONSTRAINT LEVELS MATCH - Experiments are FAIR!")
        return 0
    else:
        print("[FAILURE] CONSTRAINT MISMATCH FOUND - Experiments are UNFAIR!")
        return 1

if __name__ == "__main__":
    exit_code = verify_constraints()
    sys.exit(exit_code)

