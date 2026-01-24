#!/usr/bin/env python3
"""
Prompt Sensitivity Analysis: Test how different prompts affect ZEH.
====================================================================

Tests multiple prompt templates to show that ZEH is relatively stable
across prompt variations.

Usage:
    uv run python eval_prompt_sensitivity.py --model Qwen/Qwen2.5-0.5B-Instruct
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
    build_prompt_with_template,  # noqa: E402
    extract_number,
    set_all_seeds,
    setup_model,
)

PROMPTS = {
    "baseline": {
        "system": "Answer with only the integer.",
        "user": "{a}*{b}=",
    },
    "compute": {
        "system": "You are a calculator. Output only the result.",
        "user": "Compute {a} × {b}",
    },
    "product": {
        "system": "Give only the numerical answer.",
        "user": "What is the product of {a} and {b}?",
    },
    "eval": {
        "system": "Evaluate and respond with just the number.",
        "user": "Evaluate: {a} * {b}",
    },
    "answer": {
        "system": "Provide only the numerical answer.",
        "user": "{a} × {b} = ?",
    },
}


def generate_border_tasks(n: int, tokenizer, system: str, user_fmt: str) -> List[dict]:
    tasks = []

    for b in range(1, n + 1):
        prompt = build_prompt_with_template(tokenizer, n, b, system, user_fmt)
        answer = str(n * b)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

        tasks.append(
            {
                "a": n,
                "b": b,
                "prompt": prompt,
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                "prompt_len": len(prompt_tokens),
            }
        )

    for a in range(1, n):
        prompt = build_prompt_with_template(tokenizer, a, n, system, user_fmt)
        answer = str(a * n)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

        tasks.append(
            {
                "a": a,
                "b": n,
                "prompt": prompt,
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                "prompt_len": len(prompt_tokens),
            }
        )

    return tasks


def run_naive_generation(
    model, tokenizer, tasks: List[dict], device: str, batch_size: int = 64
) -> List[bool]:
    if not tasks:
        return []

    groups: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        groups[len(task["prompt_tokens"])].append((idx, task))

    results: List[Optional[bool]] = [None] * len(tasks)

    for prompt_len in sorted(groups.keys()):
        group = groups[prompt_len]

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
                results[idx] = predicted == expected

    return [r if r is not None else False for r in results]


def find_zeh_with_prompt(
    model,
    tokenizer,
    max_n: int,
    device: str,
    system: str,
    user_fmt: str,
    window_size: int = 30,
    batch_size: int = 64,
) -> Tuple[int, float, List[dict]]:

    torch.cuda.synchronize()
    start_time = time.time()

    found_zeh = 0
    failed_tasks = []

    n = 1
    while n <= max_n:
        window_end = min(n + window_size - 1, max_n)

        all_window_tasks = []
        task_to_n = []

        for curr_n in range(n, window_end + 1):
            border_tasks = generate_border_tasks(curr_n, tokenizer, system, user_fmt)
            for task in border_tasks:
                task["n"] = curr_n
                all_window_tasks.append(task)
                task_to_n.append(curr_n)

        results = run_naive_generation(
            model, tokenizer, all_window_tasks, device, batch_size
        )

        done = False
        for curr_n in range(n, window_end + 1):
            n_results = [results[i] for i, tn in enumerate(task_to_n) if tn == curr_n]
            n_tasks = [
                all_window_tasks[i] for i, tn in enumerate(task_to_n) if tn == curr_n
            ]

            if all(n_results):
                found_zeh = curr_n
            else:
                failed_tasks = [n_tasks[i] for i, r in enumerate(n_results) if not r]
                done = True
                break

        if done:
            break

        n = window_end + 1

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    return found_zeh, elapsed, failed_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-n", type=int, default=99)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths relative to script directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / ".." / "data"
    output_dir = output_dir.resolve()

    set_all_seeds(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = setup_model(args.model, device)
    model_name = args.model.split("/")[-1]

    print(f"\n{'='*60}")
    print(f"Testing prompt sensitivity for: {model_name}")
    print(f"{'='*60}")

    results = {}

    for prompt_name, prompt_config in PROMPTS.items():
        print(f"\n  Prompt: {prompt_name}")
        print(f"    System: {prompt_config['system'][:50]}...")
        print(f"    User: {prompt_config['user']}")

        zeh, elapsed, failed = find_zeh_with_prompt(
            model,
            tokenizer,
            args.max_n,
            device,
            system=prompt_config["system"],
            user_fmt=prompt_config["user"],
        )

        print(f"    ZEH: {zeh}, Time: {elapsed:.1f}s")
        if failed:
            print(
                f"    First failure: {failed[0]['a']}*{failed[0]['b']}={failed[0]['answer']}"
            )

        results[prompt_name] = {
            "zeh": zeh,
            "time_seconds": round(elapsed, 2),
            "failed_examples": [
                {"a": t["a"], "b": t["b"], "answer": t["answer"]} for t in failed[:3]
            ],
        }

    # Summary statistics
    zehs = [r["zeh"] for r in results.values()]
    import statistics

    print(f"\n{'='*60}")
    print(f"Summary for {model_name}")
    print(f"{'='*60}")
    print(f"  ZEH range: {min(zehs)} - {max(zehs)}")
    print(f"  ZEH mean: {statistics.mean(zehs):.1f}")
    print(
        f"  ZEH std: {statistics.stdev(zehs):.1f}"
        if len(zehs) > 1
        else "  ZEH std: N/A"
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": args.model,
        "max_n": args.max_n,
        "prompts": PROMPTS,
        "results": results,
        "summary": {
            "min_zeh": min(zehs),
            "max_zeh": max(zehs),
            "mean_zeh": round(statistics.mean(zehs), 2),
            "std_zeh": round(statistics.stdev(zehs), 2) if len(zehs) > 1 else 0,
        },
    }

    output_path = output_dir / f"{model_name}_prompt_sensitivity.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
