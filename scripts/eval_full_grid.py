#!/usr/bin/env python3
"""Generate full 99×99 multiplication grid evaluation data.

This script evaluates a model on all 9801 multiplication problems (1-99 × 1-99)
and saves the results including all errors.

Usage:
    uv run python eval_full_grid.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    RANDOM_SEED,
    build_prompt,
    extract_number,  # noqa: E402
    set_all_seeds,
    setup_model,
)


def evaluate_batch(
    model, tokenizer, tasks: List[dict], device: str, batch_size: int = 64
) -> List[dict]:
    """Evaluate a batch of tasks."""
    if not tasks:
        return []

    groups: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        plen = len(task["prompt_tokens"])
        groups[plen].append((idx, task))

    results: List[Optional[dict]] = [None] * len(tasks)

    for plen in sorted(groups.keys()):
        group = groups[plen]

        for batch_start in range(0, len(group), batch_size):
            batch_group = group[batch_start : batch_start + batch_size]
            indices = [g[0] for g in batch_group]
            prompts = [g[1]["prompt"] for g in batch_group]

            inputs = tokenizer(prompts, return_tensors="pt", padding=False).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.0,  # Pure greedy decoding
                )

            input_len = inputs.input_ids.shape[1]
            for j, idx in enumerate(indices):
                generated = outputs[j][input_len:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                predicted = extract_number(text)
                expected = int(batch_group[j][1]["answer"])

                results[idx] = {
                    "a": batch_group[j][1]["a"],
                    "b": batch_group[j][1]["b"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": (predicted == expected),
                    "raw_output": text[:50],
                }

    # Filter out None values (should not happen in practice)
    return [r for r in results if r is not None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-n", type=int, default=99)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Resolve paths relative to script directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / ".." / "data" / "grid_results"
    output_dir = output_dir.resolve()

    set_all_seeds(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = setup_model(args.model, device)

    # Generate all tasks
    print(f"\nGenerating {args.max_n}×{args.max_n} = {args.max_n**2} tasks...")
    tasks = []
    for a in range(1, args.max_n + 1):
        for b in range(1, args.max_n + 1):
            prompt = build_prompt(tokenizer, a, b)
            answer = str(a * b)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

            tasks.append(
                {
                    "a": a,
                    "b": b,
                    "prompt": prompt,
                    "answer": answer,
                    "prompt_tokens": prompt_tokens,
                }
            )

    # Evaluate
    print(f"Evaluating {len(tasks)} tasks...")
    torch.cuda.synchronize()
    start_time = time.time()

    results = evaluate_batch(model, tokenizer, tasks, device, args.batch_size)

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # Collect errors
    errors = []
    correct_count = 0

    for r in results:
        if r["correct"]:
            correct_count += 1
        else:
            errors.append(
                {
                    "a": r["a"],
                    "b": r["b"],
                    "expected": r["expected"],
                    "predicted": r["predicted"],
                }
            )

    accuracy = correct_count / len(results)

    # Compute ZEH
    error_set = {(e["a"], e["b"]) for e in errors}
    zeh = 0
    for n in range(1, args.max_n + 1):
        all_correct = True
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                if (a, b) in error_set:
                    all_correct = False
                    break
            if not all_correct:
                break
        if all_correct:
            zeh = n
        else:
            break

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(results)})")
    print(f"ZEH: {zeh}")
    print(f"Errors: {len(errors)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"{'='*60}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model.replace("/", "_").replace(".", "_")
    output_data = {
        "model": args.model,
        "grid_range": [1, args.max_n],
        "total": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "zeh": zeh,
        "time_sec": round(elapsed, 2),
        "seed": RANDOM_SEED,
        "dtype": "fp16",
        "batch_size": args.batch_size,
        "errors": errors,
    }

    output_path = output_dir / f"{model_name}_grid.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
