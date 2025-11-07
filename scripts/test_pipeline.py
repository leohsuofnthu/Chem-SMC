#!/usr/bin/env python3
"""
Dry-run test script to verify the end-to-end pipeline works.
Runs minimal experiments to test all components.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        from src import analyze_train_data, baseline_generate_constraint, smiley_generate_constraint, evaluate, plots, utils
        print("[OK] All modules imported successfully")
        
        # Test specific imports
        from src.utils import GPT_ZINC_PROMPTS, SMILEY_PROMPTS, load_property_ranges
        print(f"[OK] GPT2-Zinc prompts: {len(GPT_ZINC_PROMPTS)} prefixes")
        print(f"[OK] SmileyLlama prompts: {len(SMILEY_PROMPTS)} prompts")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_property_ranges():
    """Test that property ranges file exists and is valid."""
    print("\n" + "=" * 60)
    print("Testing property ranges file...")
    print("=" * 60)
    
    ranges_path = Path("data/train_property_ranges.json")
    if not ranges_path.exists():
        print(f"[FAIL] Property ranges file not found: {ranges_path}")
        print("  Run: python -m src.analyze_train_data")
        return False
    
    try:
        from src.utils import load_property_ranges
        ranges = load_property_ranges(str(ranges_path), "Combined", "loose")
        print(f"[OK] Property ranges loaded: {len(ranges)} properties")
        print(f"  Properties: {', '.join(ranges.keys())}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load property ranges: {e}")
        return False


def test_baseline_generation(n=10):
    """Test baseline generation with minimal parameters."""
    print("\n" + "=" * 60)
    print("Testing GPT2-Zinc baseline generation (dry-run)...")
    print("=" * 60)
    
    try:
        from src.baseline_generate_constraint import run_constraint_experiment
        
        print(f"  Running with n={n} molecules, constraint_level='loose'...")
        df = run_constraint_experiment(
            constraint_level="loose",
            property_ranges_path="data/train_property_ranges.json",
            dataset="Combined",
            n=n,
            temperature=1.0,
            top_p=0.9,
            batch_size=min(32, n),  # Small batch size for testing
            out_csv="results/test_baseline_results.csv",
            summary_csv="results/test_baseline_summary.csv",
        )
        
        print(f"[OK] Generated {len(df)} molecules")
        print(f"  Valid: {df['Valid'].sum() if 'Valid' in df.columns else 'N/A'}")
        print(f"  Adherence: {df['Adherence'].sum() if 'Adherence' in df.columns else 'N/A'}")
        return True
    except Exception as e:
        print(f"[FAIL] Baseline generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smiley_generation(n=10):
    """Test SmileyLlama generation with minimal parameters (skipped if no GPU)."""
    print("\n" + "=" * 60)
    print("Testing SmileyLlama generation (dry-run, may skip if no GPU)...")
    print("=" * 60)
    
    import torch
    if not torch.cuda.is_available():
        print("[SKIP] GPU not available - skipping SmileyLlama test")
        print("  (This is OK for dry-run, but you'll need GPU for full experiments)")
        return True  # Not a failure, just skipped
    
    try:
        from src.smiley_generate_constraint import run_constraint_experiment
        
        print(f"  Running with n={n} molecules, constraint_level='loose'...")
        df = run_constraint_experiment(
            constraint_level="loose",
            property_ranges_path="data/train_property_ranges.json",
            dataset="Combined",
            base_prompt_name="mw_logp_rotb",
            n=n,
            temperature=1.0,
            top_p=0.9,
            batch_size=min(10, n),  # Very small batch for testing
            quantize=True,
            out_csv="results/test_smiley_results.csv",
            summary_csv="results/test_smiley_summary.csv",
        )
        
        print(f"[OK] Generated {len(df)} molecules")
        print(f"  Valid: {df['Valid'].sum() if 'Valid' in df.columns else 'N/A'}")
        print(f"  Adherence: {df['Adherence'].sum() if 'Adherence' in df.columns else 'N/A'}")
        return True
    except Exception as e:
        print(f"[FAIL] SmileyLlama generation failed: {e}")
        print("  (This might be due to missing model files or GPU issues)")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation with test results."""
    print("\n" + "=" * 60)
    print("Testing evaluation...")
    print("=" * 60)
    
    baseline_path = Path("results/test_baseline_results.csv")
    smiley_path = Path("results/test_smiley_results.csv")
    
    if not baseline_path.exists():
        print("[SKIP] Baseline results not found - skipping evaluation test")
        return True
    
    try:
        import pandas as pd
        from src.evaluate import _load_results, build_tables
        
        baseline = _load_results(str(baseline_path))
        print(f"[OK] Loaded baseline results: {len(baseline)} molecules")
        
        # Try to load smiley if available
        if smiley_path.exists():
            smiley = _load_results(str(smiley_path))
            print(f"[OK] Loaded SmileyLlama results: {len(smiley)} molecules")
            combined = pd.concat([baseline, smiley], ignore_index=True)
        else:
            combined = baseline
        
        summary_table, panel_table = build_tables(combined)
        print(f"[OK] Evaluation tables created")
        print(f"  Summary rows: {len(summary_table)}")
        print(f"  Panel rows: {len(panel_table)}")
        return True
    except Exception as e:
        print(f"[FAIL] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """Remove test files created during dry-run."""
    print("\n" + "=" * 60)
    print("Cleaning up test files...")
    print("=" * 60)
    
    test_files = [
        "results/test_baseline_results.csv",
        "results/test_baseline_summary.csv",
        "results/test_smiley_results.csv",
        "results/test_smiley_summary.csv",
    ]
    
    for file in test_files:
        path = Path(file)
        if path.exists():
            path.unlink()
            print(f"  Removed {file}")
    
    print("[OK] Cleanup complete")


def main():
    """Run all dry-run tests."""
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("DRY-RUN PIPELINE TEST")
    print("=" * 60)
    print("\nThis will test the pipeline with minimal parameters.")
    print("Expected runtime: 1-5 minutes (depending on GPU availability)\n")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    results = {}
    
    # Test 1: Imports
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n[FAIL] Import test failed - cannot continue")
        return 1
    
    # Test 2: Property ranges
    results["property_ranges"] = test_property_ranges()
    if not results["property_ranges"]:
        print("\n[FAIL] Property ranges test failed - run analyze_train_data first")
        return 1
    
    # Test 3: Baseline generation
    results["baseline"] = test_baseline_generation(n=10)
    
    # Test 4: SmileyLlama generation (optional - may skip if no GPU)
    results["smiley"] = test_smiley_generation(n=5)
    
    # Test 5: Evaluation
    results["evaluation"] = test_evaluation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name:20s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Pipeline is ready to run.")
        print("\nTo run full experiments:")
        print("  bash scripts/run_experiments.sh")
        print("  or")
        print("  python -m src.baseline_generate_constraint --constraint-level loose --n 1000")
    else:
        print("\n[WARNING] Some tests failed. Please fix issues before running full experiments.")
    
    # Cleanup
    cleanup_test_files()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

