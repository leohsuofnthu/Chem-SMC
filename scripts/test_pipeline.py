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
        print("[OK] All core modules imported successfully")
        
        # Test SMC module (optional - requires genlm-control)
        try:
            from src import smc_generate_constraint
            print("[OK] SMC module imported successfully")
            smc_available = True
        except ImportError as e:
            print(f"[WARN] SMC module not available: {e}")
            print("  (This is OK if genlm-control is not installed)")
            smc_available = False
        
        # Test specific imports
        from src.utils import GPT_ZINC_PROMPTS, SMILEY_PROMPTS, load_property_ranges
        print(f"[OK] GPT2-Zinc prompts: {len(GPT_ZINC_PROMPTS)} prefixes")
        print(f"[OK] SmileyLlama prompts: {len(SMILEY_PROMPTS)} prompts")
        
        # Test GenLM availability
        try:
            from genlm.control import AWRS, PromptedLLM, Potential
            print("[OK] GenLM Control library available")
            genlm_available = True
        except ImportError:
            print("[WARN] GenLM Control library not available")
            print("  (Required for SMC experiments. Install with: pip install genlm-control>=0.2.11)")
            genlm_available = False
        
        return True, smc_available, genlm_available
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False, False


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


def test_smc_generation(n=10, genlm_available=True):
    """Test SMC generation with minimal parameters (skipped if GenLM not available)."""
    print("\n" + "=" * 60)
    print("Testing GPT2-Zinc+SMC generation (dry-run)...")
    print("=" * 60)
    
    if not genlm_available:
        print("[SKIP] GenLM Control not available - skipping SMC test")
        print("  (Install with: pip install genlm-control>=0.2.11)")
        return True  # Not a failure, just skipped
    
    try:
        from src.smc_generate_constraint import run_constraint_experiment
        
        print(f"  Running with n={n} molecules, constraint_level='loose'...")
        print(f"  Using reduced SMC parameters for faster testing...")
        df = run_constraint_experiment(
            constraint_level="loose",
            property_ranges_path="data/train_property_ranges.json",
            dataset="Combined",
            n=n,
            particles=10,  # Reduced from 50 for testing
            ess_threshold=0.3,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=60,
            top_k=30,
            out_csv="results/test_smc_results.csv",
            summary_csv="results/test_smc_summary.csv",
        )
        
        print(f"[OK] Generated {len(df)} molecules")
        print(f"  Valid: {df['Valid'].sum() if 'Valid' in df.columns else 'N/A'}")
        print(f"  Adherence: {df['Adherence'].sum() if 'Adherence' in df.columns else 'N/A'}")
        if 'Weight' in df.columns:
            print(f"  Average weight: {df['Weight'].mean():.4f}")
        return True
    except Exception as e:
        print(f"[FAIL] SMC generation failed: {e}")
        print("  (This might be due to GenLM configuration issues)")
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
    smc_path = Path("results/test_smc_results.csv")
    smiley_path = Path("results/test_smiley_results.csv")
    
    if not baseline_path.exists():
        print("[SKIP] Baseline results not found - skipping evaluation test")
        return True
    
    try:
        import pandas as pd
        from src.evaluate import _load_results, build_tables
        
        dfs_to_combine = []
        
        baseline = _load_results(str(baseline_path))
        print(f"[OK] Loaded baseline results: {len(baseline)} molecules")
        dfs_to_combine.append(baseline)
        
        # Try to load SMC if available
        if smc_path.exists():
            smc = _load_results(str(smc_path))
            print(f"[OK] Loaded SMC results: {len(smc)} molecules")
            dfs_to_combine.append(smc)
        
        # Try to load smiley if available
        if smiley_path.exists():
            smiley = _load_results(str(smiley_path))
            print(f"[OK] Loaded SmileyLlama results: {len(smiley)} molecules")
            dfs_to_combine.append(smiley)
        
        combined = pd.concat(dfs_to_combine, ignore_index=True)
        print(f"[OK] Combined {len(dfs_to_combine)} datasets: {len(combined)} total molecules")
        
        summary_table, panel_table = build_tables(combined)
        print(f"[OK] Evaluation tables created")
        print(f"  Summary rows: {len(summary_table)}")
        print(f"  Panel rows: {len(panel_table)}")
        
        # Print summary
        if len(summary_table) > 0:
            print("\n  Summary table preview:")
            for _, row in summary_table.iterrows():
                model = row.get('Model', 'Unknown')
                adherence = row.get('Adherence %', 0)
                valid = row.get('Valid %', 0)
                print(f"    {model}: Adherence={adherence:.1f}%, Valid={valid:.1f}%")
        
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
        "results/test_smc_results.csv",
        "results/test_smc_summary.csv",
        "results/test_smiley_results.csv",
        "results/test_smiley_summary.csv",
    ]
    
    removed_count = 0
    for file in test_files:
        path = Path(file)
        if path.exists():
            path.unlink()
            print(f"  Removed {file}")
            removed_count += 1
    
    if removed_count > 0:
        print(f"[OK] Cleanup complete ({removed_count} files removed)")
    else:
        print("[OK] No test files to clean up")


def main():
    """Run all dry-run tests."""
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("DRY-RUN PIPELINE TEST")
    print("=" * 60)
    print("\nThis will test the pipeline with minimal parameters.")
    print("Expected runtime: 2-10 minutes (depending on GPU/GenLM availability)\n")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    results = {}
    genlm_available = False
    
    # Test 1: Imports
    import_result = test_imports()
    if isinstance(import_result, tuple):
        results["imports"], smc_module_available, genlm_available = import_result
    else:
        results["imports"] = import_result
        smc_module_available = False
        genlm_available = False
    
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
    
    # Test 4: SMC generation (optional - requires GenLM)
    if smc_module_available and genlm_available:
        results["smc"] = test_smc_generation(n=5, genlm_available=True)
    else:
        results["smc"] = True  # Skip, not a failure
        print("\n[SKIP] SMC generation test skipped (GenLM not available)")
    
    # Test 5: SmileyLlama generation (optional - may skip if no GPU)
    results["smiley"] = test_smiley_generation(n=5)
    
    # Test 6: Evaluation
    results["evaluation"] = test_evaluation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name:20s} {status}")
    
    # Count actual failures (not skips)
    critical_tests = ["imports", "property_ranges", "baseline", "evaluation"]
    critical_passed = all(results.get(test, False) for test in critical_tests)
    
    if critical_passed:
        print("\n[SUCCESS] All critical tests passed! Pipeline is ready to run.")
        print("\nTo run full experiments:")
        print("  bash scripts/run_experiments.sh")
        print("  or")
        print("  python -m src.baseline_generate_constraint --constraint-level loose --n 1000")
        
        # Warn about optional components
        if not genlm_available:
            print("\n[NOTE] GenLM Control not available - SMC experiments will be skipped")
            print("  Install with: pip install genlm-control>=0.2.11")
        
        import torch
        if not torch.cuda.is_available():
            print("\n[NOTE] GPU not available - SmileyLlama experiments may be slow or fail")
    else:
        print("\n[WARNING] Some critical tests failed. Please fix issues before running full experiments.")
    
    # Cleanup
    cleanup_test_files()
    
    return 0 if critical_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

