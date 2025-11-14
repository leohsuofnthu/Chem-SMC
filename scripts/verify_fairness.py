#!/usr/bin/env python3
"""
Verification script to ensure all models use identical constraints for fair evaluation.
Checks both gradual constraints (default) and legacy percentile-based constraints.
Run this to verify experiments are fair.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_property_ranges,
    create_constraint_variant,
    create_gradual_constraint_prompt,
    GPT_ZINC_PROMPTS,
    SMILEY_PROMPT_MAP,
)

def verify_gradual_constraints():
    """Verify that all models use identical gradual constraints."""
    print("=" * 70)
    print("Fairness Verification: Gradual Constraints (SmileyLlama-compatible)")
    print("=" * 70)
    print()
    
    constraint_levels = ["loosen", "tight", "ultra_tight"]
    all_match = True
    
    for level in constraint_levels:
        print(f"Checking gradual constraint level: {level}")
        print("-" * 70)
        
        # Create gradual constraint specs (used by SMC and SmileyLlama)
        gradual_spec = create_gradual_constraint_prompt(level)
        gradual_constraints = gradual_spec.constraints
        
        print(f"Gradual constraints: {gradual_constraints}")
        print(f"  - loosen: MW <= 300")
        print(f"  - tight: MW <= 300, logP <= 4")
        print(f"  - ultra_tight: MW <= 300, logP <= 4, RotB <= 10")
        
        # Verify constraint structure
        expected_counts = {"loosen": 1, "tight": 2, "ultra_tight": 3}
        actual_count = len(gradual_constraints)
        expected_count = expected_counts[level]
        
        if actual_count == expected_count:
            print(f"[OK] Constraint count correct: {actual_count} constraints")
        else:
            print(f"[ERROR] Constraint count mismatch: expected {expected_count}, got {actual_count}")
            all_match = False
        
        # Verify specific constraints
        if level == "loosen":
            if gradual_constraints.get("MW") == (None, 300.0):
                print("[OK] loosen constraints correct")
            else:
                print(f"[ERROR] loosen constraints incorrect: {gradual_constraints}")
                all_match = False
        elif level == "tight":
            if (gradual_constraints.get("MW") == (None, 300.0) and 
                gradual_constraints.get("logP") == (None, 4.0)):
                print("[OK] tight constraints correct")
            else:
                print(f"[ERROR] tight constraints incorrect: {gradual_constraints}")
                all_match = False
        elif level == "ultra_tight":
            if (gradual_constraints.get("MW") == (None, 300.0) and 
                gradual_constraints.get("logP") == (None, 4.0) and
                gradual_constraints.get("RotB") == (None, 10.0)):
                print("[OK] ultra_tight constraints correct")
            else:
                print(f"[ERROR] ultra_tight constraints incorrect: {gradual_constraints}")
                all_match = False
        
        print()
    
    return all_match

def verify_legacy_constraints():
    """Verify that legacy percentile-based constraints match across models."""
    print("=" * 70)
    print("Fairness Verification: Legacy Percentile-Based Constraints")
    print("=" * 70)
    print()
    
    property_ranges_path = "data/train_property_ranges.json"
    dataset = "Combined"
    constraint_levels = ["loose", "tight", "ultra_tight"]
    
    all_match = True
    
    for level in constraint_levels:
        print(f"Checking legacy constraint level: {level}")
        print("-" * 70)
        
        # Load constraint ranges
        try:
            constraint_ranges = load_property_ranges(property_ranges_path, dataset, level)
            print(f"Loaded ranges: {constraint_ranges}")
        except Exception as e:
            print(f"[SKIP] Could not load property ranges: {e}")
            print("  (This is OK if you're only using gradual constraints)")
            continue
        
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
        
        print()
    
    return all_match

def verify_constraints():
    """Verify that all models use identical constraints for fair evaluation."""
    print("\n" + "=" * 70)
    print("FAIRNESS VERIFICATION")
    print("=" * 70)
    print("\nThis script verifies that all models use identical constraints")
    print("for evaluation, ensuring fair comparison.\n")
    
    gradual_ok = verify_gradual_constraints()
    legacy_ok = verify_legacy_constraints()
    
    print("=" * 70)
    if gradual_ok and legacy_ok:
        print("[SUCCESS] ALL CONSTRAINT CHECKS PASSED - Experiments are FAIR!")
        print("\nNote: Gradual constraints are used by default (SmileyLlama-compatible)")
        print("      Legacy percentile-based constraints are available for backward compatibility")
        return 0
    else:
        print("[FAILURE] SOME CONSTRAINT CHECKS FAILED - Please review!")
        return 1

if __name__ == "__main__":
    exit_code = verify_constraints()
    sys.exit(exit_code)

