#!/usr/bin/env python3
"""
Frequency Correlation Analysis: Analyze correlation between C4 corpus frequency and model accuracy.
===================================================================================================

Analyzes whether models with different sizes show different correlations between
training data frequency and accuracy.

Usage:
    uv run python analyze_frequency_correlation.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import stats

# Script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()


def load_frequency_data(filepath: Path) -> Dict:
    """Load C4 frequency data."""
    with open(filepath) as f:
        return json.load(f)


def load_model_results(results_dir: Path) -> Dict:
    """Load model evaluation results."""
    models = {}
    model_files = [
        ("Qwen2.5-0.5B", "Qwen_Qwen2_5-0_5B-Instruct_grid.json", 0.5),
        ("Qwen2.5-1.5B", "Qwen_Qwen2_5-1_5B-Instruct_grid.json", 1.5),
        ("Qwen2.5-3B", "Qwen_Qwen2_5-3B-Instruct_grid.json", 3.0),
        ("Qwen2.5-7B", "Qwen_Qwen2_5-7B-Instruct_grid.json", 7.0),
        ("Qwen2.5-14B", "Qwen_Qwen2_5-14B-Instruct_grid.json", 14.0),
        ("Qwen2.5-32B", "Qwen_Qwen2_5-32B-Instruct_grid.json", 32.0),
        ("Qwen2.5-72B", "Qwen_Qwen2_5-72B-Instruct_grid.json", 72.0),
    ]

    for model_name, filename, params in model_files:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            models[model_name] = {
                "params": params,
                "accuracy": data["accuracy"],
                "errors": {(e["a"], e["b"]) for e in data.get("errors", [])},
            }

    return models


def analyze_correlation(
    frequency_data: Dict,
    model_data: Dict,
    model_name: str,
) -> Dict:
    """Compute Spearman correlation between frequency and accuracy."""

    pair_counts = frequency_data.get("pair_counts", {})
    error_set = model_data["errors"]

    frequencies = []
    correct_flags = []

    for a in range(1, 100):
        for b in range(1, 100):
            key = f"{a},{b}"
            freq = pair_counts.get(key, 0)
            is_correct = 1 if (a, b) not in error_set else 0

            frequencies.append(freq)
            correct_flags.append(is_correct)

    # Spearman correlation
    spearman_result = stats.spearmanr(frequencies, correct_flags)
    # scipy.stats.spearmanr returns SignificanceResult with .statistic and .pvalue
    rho = float(spearman_result[0])  # type: ignore[index]
    p_value = float(spearman_result[1])  # type: ignore[index]

    return {
        "model": model_name,
        "params": model_data["params"],
        "accuracy": model_data["accuracy"],
        "spearman_rho": rho,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency-file", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths relative to script directory
    if args.frequency_file:
        frequency_path = Path(args.frequency_file)
    else:
        frequency_path = SCRIPT_DIR / ".." / "data" / "c4_frequency_results.json"

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = SCRIPT_DIR / ".." / "data" / "grid_results"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = SCRIPT_DIR / ".." / "data" / "frequency_correlation.json"

    frequency_path = frequency_path.resolve()
    results_dir = results_dir.resolve()
    output_path = output_path.resolve()

    if not frequency_path.exists():
        print(f"Frequency file not found: {frequency_path}")
        print(
            "Please ensure c4_frequency_results.json is present in the data directory."
        )
        return

    frequency_data = load_frequency_data(frequency_path)
    models = load_model_results(results_dir)

    if not models:
        print(f"No model results found in {results_dir}")
        return

    print("=" * 70)
    print("Frequency-Accuracy Correlation Analysis")
    print("=" * 70)

    print("\nC4 corpus statistics:")
    summary = frequency_data.get("summary", {})
    print(f"  Total documents: {summary.get('total_documents', 'N/A'):,}")
    print(f"  Total matches: {summary.get('total_matches', 'N/A'):,}")
    print(f"  Unique pairs: {len(frequency_data.get('pair_counts', {})):,}")

    results = []

    print(
        f"\n{'Model':<15} {'Params':>8} {'Accuracy':>10} {'Spearman ρ':>12} {'p-value':>12}"
    )
    print("-" * 60)

    for model_name in sorted(models.keys(), key=lambda x: models[x]["params"]):
        corr = analyze_correlation(frequency_data, models[model_name], model_name)
        results.append(corr)

        print(
            f"{model_name:<15} {corr['params']:>8.1f}B "
            f"{corr['accuracy']:>10.4f} "
            f"{corr['spearman_rho']:>12.4f} "
            f"{corr['p_value']:>12.2e}"
        )

    # Meta-correlation: model size vs frequency correlation
    meta_rho: Optional[float] = None
    meta_p: Optional[float] = None
    if len(results) >= 3:
        log_params = [np.log10(r["params"]) for r in results]
        rhos = [r["spearman_rho"] for r in results]

        spearman_result = stats.spearmanr(log_params, rhos)
        meta_rho = float(spearman_result[0])  # type: ignore[index]
        meta_p = float(spearman_result[1])  # type: ignore[index]

        print("\n" + "-" * 60)
        print("Meta-correlation (log(params) vs Spearman ρ):")
        print(f"  ρ = {meta_rho:.4f}, p = {meta_p:.6f}")

        if meta_p < 0.05:
            if meta_rho < 0:
                print("\n→ Larger models show WEAKER frequency-accuracy correlation")
                print("→ This suggests larger models rely less on memorization")
            else:
                print("\n→ Larger models show STRONGER frequency-accuracy correlation")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = frequency_data.get("summary", {})
    output_data = {
        "frequency_stats": {
            "total_documents": summary.get("total_documents"),
            "total_matches": summary.get("total_matches"),
            "unique_pairs": len(frequency_data.get("pair_counts", {})),
        },
        "correlations": results,
        "meta_correlation": {
            "spearman_rho": meta_rho,
            "p_value": meta_p,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
