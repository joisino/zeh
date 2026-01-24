#!/usr/bin/env python3
"""
Carry Error Analysis: Analyze how model size affects carry-related errors.
==========================================================================

Usage:
    uv run python analyze_carry_errors.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

# Script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()


def has_carry_simple(a: int, b: int) -> bool:
    """Check if a × b involves carry (any digit product >= 10)."""
    str_a = str(a)
    str_b = str(b)

    for da in str_a:
        for db in str_b:
            if int(da) * int(db) >= 10:
                return True
    return False


def load_model_data(model_path: Path) -> Dict:
    with open(model_path) as f:
        return json.load(f)


def analyze_logistic_regression(results_dir: Path) -> Optional[Dict]:
    """Analyze carry × model_size interaction using logistic regression."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\nsklearn not available, skipping logistic regression")
        return None

    model_files = [
        ("Qwen2.5-0.5B", "Qwen_Qwen2_5-0_5B-Instruct_grid.json", 0.5),
        ("Qwen2.5-1.5B", "Qwen_Qwen2_5-1_5B-Instruct_grid.json", 1.5),
        ("Qwen2.5-3B", "Qwen_Qwen2_5-3B-Instruct_grid.json", 3.0),
        ("Qwen2.5-7B", "Qwen_Qwen2_5-7B-Instruct_grid.json", 7.0),
        ("Qwen2.5-14B", "Qwen_Qwen2_5-14B-Instruct_grid.json", 14.0),
        ("Qwen2.5-32B", "Qwen_Qwen2_5-32B-Instruct_grid.json", 32.0),
        ("Qwen2.5-72B", "Qwen_Qwen2_5-72B-Instruct_grid.json", 72.0),
    ]

    all_data = []

    for model_name, filename, params in model_files:
        filepath = results_dir / filename
        if not filepath.exists():
            continue

        data = load_model_data(filepath)
        error_set = {(e["a"], e["b"]) for e in data.get("errors", [])}

        for a in range(1, 100):
            for b in range(1, 100):
                carry = 1 if has_carry_simple(a, b) else 0
                correct = 0 if (a, b) in error_set else 1
                log_params = np.log10(params)

                all_data.append(
                    {
                        "carry": carry,
                        "correct": correct,
                        "log_params": log_params,
                    }
                )

    X_base = np.array([[d["carry"], d["log_params"]] for d in all_data])
    X_inter = np.array(
        [[d["carry"], d["log_params"], d["carry"] * d["log_params"]] for d in all_data]
    )
    y = np.array([d["correct"] for d in all_data])

    scaler_base = StandardScaler()
    scaler_inter = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)
    X_inter_scaled = scaler_inter.fit_transform(X_inter)

    # Model without interaction
    model_a = LogisticRegression(max_iter=1000, solver="lbfgs")
    model_a.fit(X_base_scaled, y)

    # Model with interaction
    model_b = LogisticRegression(max_iter=1000, solver="lbfgs")
    model_b.fit(X_inter_scaled, y)

    # Likelihood ratio test
    ll_a = -log_loss(y, model_a.predict_proba(X_base_scaled), normalize=False)
    ll_b = -log_loss(y, model_b.predict_proba(X_inter_scaled), normalize=False)

    chi2_stat = 2 * (ll_b - ll_a)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

    print("\n" + "=" * 70)
    print("Logistic Regression: correct ~ carry + log(params) + carry×log(params)")
    print("=" * 70)

    print("\nModel without interaction:")
    print(f"  carry coef: {model_a.coef_[0][0]:.4f}")
    print(f"  log_params coef: {model_a.coef_[0][1]:.4f}")

    print("\nModel with interaction:")
    print(f"  carry coef: {model_b.coef_[0][0]:.4f}")
    print(f"  log_params coef: {model_b.coef_[0][1]:.4f}")
    print(f"  interaction coef: {model_b.coef_[0][2]:.4f}")

    print("\nLikelihood ratio test:")
    print(f"  χ² = {chi2_stat:.4f}, df = 1, p = {p_value:.6f}")

    if p_value < 0.05:
        interaction_coef = model_b.coef_[0][2]
        if interaction_coef < 0:
            print("\n→ Interaction is significant (p < 0.05)")
            print(
                f"→ Negative interaction ({interaction_coef:.4f}): "
                "larger models have MORE relative difficulty with carry"
            )
        else:
            print("\n→ Interaction is significant (p < 0.05)")
            print(
                f"→ Positive interaction ({interaction_coef:.4f}): "
                "larger models have LESS relative difficulty with carry"
            )
    else:
        print("\n→ Interaction is NOT significant (p >= 0.05)")

    return {
        "model_without_interaction": {
            "carry_coef": float(model_a.coef_[0][0]),
            "log_params_coef": float(model_a.coef_[0][1]),
        },
        "model_with_interaction": {
            "carry_coef": float(model_b.coef_[0][0]),
            "log_params_coef": float(model_b.coef_[0][1]),
            "interaction_coef": float(model_b.coef_[0][2]),
        },
        "likelihood_ratio_test": {
            "chi2": float(chi2_stat),
            "df": 1,
            "p_value": float(p_value),
        },
    }


def print_structured_error_table(results_dir: Path) -> List[Dict]:
    """Print structured error table and return data.

    Structured errors are errors where the difference between predicted and expected
    is a multiple of 10 (within 100), indicating the model made a systematic mistake
    in digit position rather than a random error.
    """
    model_files = [
        ("0.5B", "Qwen_Qwen2_5-0_5B-Instruct_grid.json", 0.5),
        ("1.5B", "Qwen_Qwen2_5-1_5B-Instruct_grid.json", 1.5),
        ("3B", "Qwen_Qwen2_5-3B-Instruct_grid.json", 3.0),
        ("7B", "Qwen_Qwen2_5-7B-Instruct_grid.json", 7.0),
        ("14B", "Qwen_Qwen2_5-14B-Instruct_grid.json", 14.0),
        ("32B", "Qwen_Qwen2_5-32B-Instruct_grid.json", 32.0),
        ("72B", "Qwen_Qwen2_5-72B-Instruct_grid.json", 72.0),
    ]

    print("\n" + "=" * 90)
    print("Structured Error Statistics")
    print("=" * 90)
    print(
        f"\n{'Model':<8} {'Accuracy':>10} {'TotalErr':>10} {'StructuredErr':>14} {'Other':>8} {'StructuredRate':>16}"
    )
    print("-" * 90)

    table_data = []

    for model_name, filename, params in model_files:
        filepath = results_dir / filename
        if not filepath.exists():
            continue

        with open(filepath) as f:
            data = json.load(f)

        errors = data.get("errors", [])
        accuracy = data.get("accuracy", 0)
        total_errors = len(errors)

        # Count structured errors (diff is multiple of 10, within 100)
        # These indicate systematic digit-position mistakes rather than random errors
        structured_errors = 0
        for e in errors:
            expected = e.get("expected")
            predicted = e.get("predicted")
            if expected is not None and predicted is not None:
                diff = abs(predicted - expected)
                if diff > 0 and diff <= 100 and diff % 10 == 0:
                    structured_errors += 1

        other_errors = total_errors - structured_errors
        structured_rate = (
            structured_errors / total_errors * 100 if total_errors > 0 else 0
        )

        print(
            f"{model_name:<8} {accuracy:>10.1%} {total_errors:>10} "
            f"{structured_errors:>14} {other_errors:>8} {structured_rate:>15.0f}%"
        )

        table_data.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "total_errors": total_errors,
                "structured_errors": structured_errors,
                "other_errors": other_errors,
                "structured_rate": structured_rate,
            }
        )

    return table_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths relative to script directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = SCRIPT_DIR / ".." / "data" / "grid_results"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = SCRIPT_DIR / ".." / "data" / "carry_analysis.json"

    results_dir = results_dir.resolve()
    output_path = output_path.resolve()

    # Structured error table
    table_data = print_structured_error_table(results_dir)

    # Logistic regression analysis (uses actual carry detection)
    lr_results = analyze_logistic_regression(results_dir)

    # Prepare output
    results: Dict = {
        "structured_error_stats": table_data,
    }
    if lr_results:
        results["logistic_regression"] = lr_results

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
