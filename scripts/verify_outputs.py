#!/usr/bin/env python3
"""
Verify that all experiment outputs have correct and consistent columns.
Checks result CSV files and summary CSV files for completeness.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Expected columns for result CSV files
EXPECTED_RESULT_COLUMNS = {
    "baseline": [
        "SMILES", "Valid", "QED", "MW", "logP", "RotB", "TPSA", "HBD", "HBA",
        "Adherence", "Weight", "Model", "Prompt", "ConstraintLevel",
        "Temperature", "TopP", "Prefix",
    ],
    "smc": [
        "SMILES", "Valid", "QED", "MW", "logP", "RotB", "TPSA", "HBD", "HBA",
        "Adherence", "Weight", "Model", "Prompt", "ConstraintLevel",
        "Temperature", "TopP", "Prefix",
    ],
    "smiley": [
        "SMILES", "Valid", "QED", "MW", "logP", "RotB", "TPSA", "HBD", "HBA",
        "Adherence", "Weight", "Model", "Prompt", "ConstraintLevel",
        "Temperature", "TopP",
    ],
}

# Expected columns for summary CSV files
EXPECTED_SUMMARY_COLUMNS = [
    "Valid %", "Distinct %", "Adherence %", "QED",
    "ConstraintLevel", "Model", "Prompt", "Temperature",
    "Runtime_seconds", "Runtime_minutes", "Runtime_formatted",
]

def check_result_file(filepath: Path, model_type: str) -> tuple[bool, list[str]]:
    """Check if a result CSV file has correct columns."""
    if not filepath.exists():
        return False, [f"File not found: {filepath}"]
    
    try:
        df = pd.read_csv(filepath)
        expected = EXPECTED_RESULT_COLUMNS.get(model_type, [])
        missing = [col for col in expected if col not in df.columns]
        extra = [col for col in df.columns if col not in expected]
        
        issues = []
        if missing:
            issues.append(f"Missing columns: {missing}")
        if extra:
            issues.append(f"Extra columns: {extra}")
        
        # Check for required data columns
        required_data_cols = ["SMILES", "Valid", "Adherence"]
        missing_data = [col for col in required_data_cols if col not in df.columns]
        if missing_data:
            issues.append(f"Missing required data columns: {missing_data}")
        
        # Check that we have some data
        if len(df) == 0:
            issues.append("File is empty")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Error reading file: {e}"]

def check_summary_file(filepath: Path) -> tuple[bool, list[str]]:
    """Check if a summary CSV file has correct columns."""
    if not filepath.exists():
        return False, [f"File not found: {filepath}"]
    
    try:
        df = pd.read_csv(filepath)
        missing = [col for col in EXPECTED_SUMMARY_COLUMNS if col not in df.columns]
        
        issues = []
        if missing:
            issues.append(f"Missing columns: {missing}")
        
        # Check that we have data
        if len(df) == 0:
            issues.append("File is empty")
        
        # Check that metrics are present
        required_metrics = ["Valid %", "Adherence %", "Distinct %"]
        missing_metrics = [col for col in required_metrics if col not in df.columns]
        if missing_metrics:
            issues.append(f"Missing required metrics: {missing_metrics}")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Error reading file: {e}"]

def main():
    """Check all output files in results directory."""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("Results directory not found. Run experiments first.")
        return 1
    
    print("=" * 60)
    print("Verifying Experiment Output Files")
    print("=" * 60)
    
    all_ok = True
    
    # Check result files
    print("\nChecking result CSV files...")
    result_files = {
        "baseline": list(results_dir.glob("baseline*results.csv")),
        "smc": list(results_dir.glob("smc*results.csv")),
        "smiley": list(results_dir.glob("smiley*results.csv")),
    }
    
    for model_type, files in result_files.items():
        if not files:
            print(f"  [{model_type}]: No result files found (OK if no experiments run yet)")
            continue
        
        for filepath in files:
            if "test_" in filepath.name:
                continue  # Skip test files
            
            ok, issues = check_result_file(filepath, model_type)
            if ok:
                df = pd.read_csv(filepath)
                print(f"  [{model_type}] {filepath.name}: OK ({len(df)} rows, {len(df.columns)} columns)")
            else:
                print(f"  [{model_type}] {filepath.name}: FAIL")
                for issue in issues:
                    print(f"    - {issue}")
                all_ok = False
    
    # Check summary files
    print("\nChecking summary CSV files...")
    summary_files = list(results_dir.glob("*summary.csv"))
    
    for filepath in summary_files:
        if "test_" in filepath.name:
            continue  # Skip test files
        
        ok, issues = check_summary_file(filepath)
        if ok:
            df = pd.read_csv(filepath)
            print(f"  {filepath.name}: OK ({len(df)} rows, {len(df.columns)} columns)")
        else:
            print(f"  {filepath.name}: FAIL")
            for issue in issues:
                print(f"    - {issue}")
            all_ok = False
    
    # Check evaluation tables if they exist
    print("\nChecking evaluation tables...")
    eval_files = {
        "summary_table": results_dir / "summary_table.csv",
        "panel_table": results_dir / "panel_table.csv",
    }
    
    for name, filepath in eval_files.items():
        if not filepath.exists():
            print(f"  {name}: Not found (OK if evaluation not run yet)")
            continue
        
        try:
            df = pd.read_csv(filepath)
            # Check for expected metrics
            expected_metrics = ["Adherence %", "Valid %", "Distinct %", "Diversity"]
            missing = [m for m in expected_metrics if m not in df.columns]
            if missing:
                print(f"  {name}: FAIL - Missing metrics: {missing}")
                all_ok = False
            else:
                print(f"  {name}: OK ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            print(f"  {name}: FAIL - Error: {e}")
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("SUCCESS: All output files are correct and complete!")
        return 0
    else:
        print("WARNING: Some output files have issues. Please review above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

