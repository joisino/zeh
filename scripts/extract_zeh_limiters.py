#!/usr/bin/env python3
"""
Extract ZEH-limiting examples: the first error that determines ZEH for each model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()


def find_zeh_and_blocker(
    errors: List[Dict], max_range: int = 99
) -> Tuple[int, Optional[Dict]]:
    """Find ZEH and the first error that limits it."""
    error_set = {(e["a"], e["b"]): e for e in errors}
    for n in range(1, max_range + 1):
        for a in range(1, n + 1):
            if (a, n) in error_set:
                return n - 1, error_set[(a, n)]
            if (n, a) in error_set:
                return n - 1, error_set[(n, a)]
    return max_range, None


def main():
    import argparse

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
        output_path = SCRIPT_DIR / ".." / "data" / "zeh_limiters.json"

    results_dir = results_dir.resolve()
    output_path = output_path.resolve()

    print("=" * 70)
    print("ZEH-Limiting Examples")
    print("=" * 70)
    print()
    print("The first error that determines ZEH for each model:")
    print()

    all_results = []

    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)

        zeh, blocker = find_zeh_and_blocker(data.get("errors", []))

        # Extract model name
        model_name = (
            f.name.replace("Qwen_Qwen2_5-", "Qwen2.5-")
            .replace("_grid.json", "")
            .replace("-Instruct", "")
        )

        result = {
            "model": model_name,
            "zeh": zeh,
            "accuracy": data.get("accuracy"),
        }

        if blocker:
            expected = blocker["a"] * blocker["b"]
            predicted = blocker.get("predicted")

            result["blocker"] = {
                "a": blocker["a"],
                "b": blocker["b"],
                "expected": expected,
                "predicted": predicted,
            }

            print(f"{model_name:>15}: ZEH={zeh:2}")
            print(
                f"                  Blocked by: {blocker['a']}×{blocker['b']}={expected}"
            )
            print(f"                  Model predicted: {predicted}")
            print()
        else:
            print(f"{model_name:>15}: ZEH={zeh} (no errors in range)")
            print()

        all_results.append(result)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
