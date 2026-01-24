#!/usr/bin/env python3
"""
Visualize ZEH vs Accuracy: Generate a figure comparing 7B and 72B models.
==============================================================================================

Generates heatmaps comparing model pairs at different zoom levels to illustrate:
1. How models can look similar at small scales (n<=20) but differ dramatically at large scales (n<=99)
2. How ZEH captures this difference while accuracy does not

Output: qwen_zeh_acc.pdf

Usage:
    uv run python visualize_zeh_vs_accuracy.py --results-dir ../data/grid_results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Color palette
RED = "#ff4b00"
GRAY = "#c8c8cb"
BLUE = "#005aff"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


def load_results(results_dir: Path) -> Dict:
    """Load all Qwen2.5 model results."""
    models = {}
    for json_file in sorted(results_dir.glob("Qwen_Qwen2*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)
        model_name = data["model"]
        models[model_name] = data
    return models


def get_error_set(model_data: Dict) -> Set[Tuple[int, int]]:
    """Get set of (a, b) pairs that are incorrect."""
    errors = model_data.get("errors", [])
    return {(e["a"], e["b"]) for e in errors}


def create_correctness_grid(
    error_set: Set[Tuple[int, int]], max_n: int = 99
) -> np.ndarray:
    """Create a 2D grid where 1 = correct, 0 = error."""
    grid = np.ones((max_n, max_n), dtype=np.float32)
    for a, b in error_set:
        if a <= max_n and b <= max_n:
            grid[a - 1, b - 1] = 0
    return grid


def compute_zeh(error_set: Set[Tuple[int, int]], max_range: int = 99) -> int:
    """Compute ZEH: largest N such that all a*b for 1 <= a,b <= N are correct."""
    for n in range(1, max_range + 1):
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                if (a, b) in error_set:
                    return n - 1
    return max_range


def compute_accuracy_for_range(error_set: Set[Tuple[int, int]], max_n: int) -> float:
    """Compute accuracy for problems where a, b <= max_n."""
    total = max_n * max_n
    errors_in_range = sum(1 for (a, b) in error_set if a <= max_n and b <= max_n)
    correct = total - errors_in_range
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths relative to script directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = SCRIPT_DIR / ".." / "data" / "grid_results"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / ".." / "figures"

    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_results(results_dir)

    # Find 7B and 72B models
    model_7b = None
    model_72b = None
    for name, data in models.items():
        if "7B-Instruct" in name and "72B" not in name:
            model_7b = (name, data)
        elif "72B-Instruct" in name:
            model_72b = (name, data)

    if not model_7b or not model_72b:
        raise ValueError("Could not find 7B and 72B models in results")

    name_7b, data_7b = model_7b
    name_72b, data_72b = model_72b

    error_7b = get_error_set(data_7b)
    error_72b = get_error_set(data_72b)

    grid_7b = create_correctness_grid(error_7b)
    grid_72b = create_correctness_grid(error_72b)

    zeh_7b = compute_zeh(error_7b)
    zeh_72b = compute_zeh(error_72b)

    print(f"7B:  ZEH={zeh_7b}")
    print(f"72B: ZEH={zeh_72b}")

    # Create the 2x3 grid figure
    cmap = ListedColormap([RED, GRAY])

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    plt.subplots_adjust(wspace=0.22, hspace=0.35)

    ranges = [20, 50, 99]
    col_titles = ["$n \\leq 20$", "$n \\leq 50$", "$n \\leq 99$"]

    for col, (n, title) in enumerate(zip(ranges, col_titles)):
        for row, (grid, zeh, err, name) in enumerate(
            [
                (grid_7b, zeh_7b, error_7b, "Qwen2.5-7B-Instruct"),
                (grid_72b, zeh_72b, error_72b, "Qwen2.5-72B-Instruct"),
            ]
        ):
            ax = axes[row, col]
            ax.imshow(
                grid[:n, :n],
                cmap=cmap,
                vmin=0,
                vmax=1,
                aspect="equal",
                extent=[0.5, n + 0.5, n + 0.5, 0.5],
                interpolation="nearest",
            )

            acc = compute_accuracy_for_range(err, n)

            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

            if col == 0:
                ax.set_ylabel(f"{name}\n\n$a$", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("$a$", fontsize=10)

            if row == 1:
                ax.set_xlabel("$b$", fontsize=10)

            # Set integer ticks for n=20
            if n == 20:
                ax.set_xticks([5, 10, 15, 20])
                ax.set_yticks([5, 10, 15, 20])

            # ZEH boundary for n=99
            if n == 99:
                ax.axhline(
                    y=zeh + 0.5, color=BLUE, linestyle="--", linewidth=1.8, alpha=0.9
                )
                ax.axvline(
                    x=zeh + 0.5, color=BLUE, linestyle="--", linewidth=1.8, alpha=0.9
                )

            # Accuracy at bottom-left
            ax.text(
                0.03,
                0.03,
                f"{acc:.1%}",
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="bottom",
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="square,pad=0.3",
                    facecolor="black",
                    alpha=0.65,
                    edgecolor="none",
                ),
            )

            # ZEH label at center of ZEH zone for n=99
            if n == 99:
                zeh_label_x = zeh / 2 + 0.5
                zeh_label_y = zeh / 2 + 0.5
                ax.text(
                    zeh_label_x,
                    zeh_label_y,
                    f"ZEH={zeh}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    color=BLUE,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="square,pad=0.25",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="none",
                    ),
                )

    plt.tight_layout()

    # Save as PDF (main output)
    output_path = output_dir / "qwen_zeh_acc.pdf"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", edgecolor="none")

    # Also save as PNG for quick preview
    fig.savefig(
        output_dir / "qwen_zeh_acc.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
