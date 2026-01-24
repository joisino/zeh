#!/usr/bin/env python3
"""
Verification Runtime Benchmark: TF, TF+Prefill, Trie(SDPA), FlashTree
=====================================================================

Measures verification runtime (single forward pass) across multiple methods
on the 1–99 multiplication suite (9801 tasks).

Methods compared:
1. TF (Teacher Forcing): Standard batched verification with teacher forcing
2. TF + Prefill: Teacher forcing with prompt prefilling (common prefix KV cache sharing)
3. Trie (SDPA): Trie structure with SDPA-based attention (dense 4D mask)
4. FlashTree: Triton-based sparse tree attention (memory-efficient, no explicit mask)

Note: Both Trie (SDPA) and FlashTree use teacher forcing and prompt prefilling as well.

Usage:
    uv run python speedtest.py --model Qwen/Qwen2.5-0.5B-Instruct --range 99
    uv run python speedtest.py --all --range 99
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers.cache_utils import DynamicCache

# Script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add scripts directory to path for local imports
sys.path.insert(0, str(SCRIPT_DIR))
# Add src directory to path for local imports
sys.path.insert(0, str(SCRIPT_DIR / ".." / "src"))

from utils import MODELS, RANDOM_SEED, generate_tasks, set_all_seeds
from utils import setup_model_with_dtype as setup_model  # noqa: E402

from flashtree import patch_qwen2_attention_for_treeflash  # noqa: E402  # type: ignore
from flashtree import set_triton_config  # noqa: E402  # type: ignore
from flashtree import (
    run_flashtree_verification as _run_flashtree_verification,
)  # noqa: E402  # type: ignore
from sdpatrie import find_common_prefix_len  # noqa: E402  # type: ignore
from sdpatrie import run_sdpatrie_sdpa_verification  # noqa: E402  # type: ignore

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Method 1: Naive Batch Verification
# ============================================================
def run_naive_verification(
    model, tokenizer, tasks: List[dict], device: str, dtype, batch_size: int = 64
) -> Tuple[List[bool], float, float]:
    """
    Run naive batch verification.

    Returns:
        Tuple of (results, total_time, peak_memory_mb)
    """
    results: List[Optional[bool]] = [None] * len(tasks)
    total_time = 0.0

    torch.cuda.reset_peak_memory_stats()

    # Group by sequence length for efficient batching
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        seq_len = len(task["full_tokens"])
        groups[seq_len].append(idx)

    for seq_len in sorted(groups.keys()):
        group_indices = groups[seq_len]
        for batch_start in range(0, len(group_indices), batch_size):
            batch_indices = group_indices[batch_start : batch_start + batch_size]
            input_ids_list = [tasks[idx]["full_tokens"] for idx in batch_indices]
            input_ids = torch.tensor(input_ids_list, device=device)

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits
            torch.cuda.synchronize()
            total_time += time.time() - start

            for j, task_idx in enumerate(batch_indices):
                task = tasks[task_idx]
                prompt_len = task["prompt_len"]
                answer_tokens = task["answer_tokens"]
                all_match = True
                for k, expected_token in enumerate(answer_tokens):
                    pred_pos = prompt_len - 1 + k
                    pred_token = logits[j, pred_pos].argmax().item()
                    if pred_token != expected_token:
                        all_match = False
                        break
                results[task_idx] = all_match

    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return [r if r is not None else False for r in results], total_time, peak_memory_mb


# ============================================================
# Method 2: Prefill Verification (2-level)
# ============================================================
def run_prefill_verification(
    model, tokenizer, tasks: List[dict], device: str, dtype, batch_size: int = 64
) -> Tuple[List[bool], float, float]:
    """
    Run weaker trie (2-level) verification.

    Returns:
        Tuple of (results, total_time, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()

    all_token_ids = [task["full_tokens"] for task in tasks]
    common_prefix_len = find_common_prefix_len(all_token_ids)
    common_prefix = all_token_ids[0][:common_prefix_len]

    results = {idx: {} for idx in range(len(tasks))}
    total_time = 0.0

    # Prefix prefill
    prefix_input = torch.tensor([common_prefix], device=device)
    prefix_cache_position = torch.arange(common_prefix_len, device=device)
    prefix_cache = DynamicCache()

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        prefix_output = model(
            input_ids=prefix_input,
            cache_position=prefix_cache_position,
            past_key_values=prefix_cache,
            use_cache=True,
        )
    torch.cuda.synchronize()
    total_time += time.time() - start

    prefix_len = common_prefix_len
    prefix_kv = [
        (prefix_cache[i][0], prefix_cache[i][1]) for i in range(len(prefix_cache))
    ]

    # Check prefix predictions
    for idx, task in enumerate(tasks):
        answer_start = task["prompt_len"]
        answer_tokens = task["answer_tokens"]
        for k, expected_token in enumerate(answer_tokens):
            pred_pos_in_full = answer_start - 1 + k
            if pred_pos_in_full < common_prefix_len:
                pred_token = prefix_output.logits[0, pred_pos_in_full].argmax().item()
                results[idx][k] = pred_token == expected_token

    # Group by suffix length
    groups = defaultdict(list)
    for idx, task in enumerate(tasks):
        suffix = task["full_tokens"][common_prefix_len:]
        suffix_len = len(suffix)
        groups[suffix_len].append((idx, suffix))

    # Process suffixes
    for suffix_len in sorted(groups.keys()):
        group = groups[suffix_len]
        for batch_start in range(0, len(group), batch_size):
            batch_group = group[batch_start : batch_start + batch_size]
            current_batch_size = len(batch_group)
            batch_indices = [g[0] for g in batch_group]
            batch_suffixes = [g[1] for g in batch_group]

            suffix_input = torch.tensor(batch_suffixes, device=device)
            position_ids = torch.arange(
                prefix_len, prefix_len + suffix_len, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(current_batch_size, -1)

            total_len = prefix_len + suffix_len
            causal_mask = torch.triu(
                torch.full(
                    (suffix_len, suffix_len),
                    torch.finfo(dtype).min,
                    device=device,
                    dtype=dtype,
                ),
                diagonal=1,
            )
            mask = torch.zeros(
                (current_batch_size, 1, suffix_len, total_len),
                device=device,
                dtype=dtype,
            )
            mask[:, :, :, prefix_len:] = causal_mask.unsqueeze(0).unsqueeze(0)

            suffix_cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(prefix_kv):
                key_expanded = key.expand(current_batch_size, -1, -1, -1).clone()
                value_expanded = value.expand(current_batch_size, -1, -1, -1).clone()
                suffix_cache.update(key_expanded, value_expanded, layer_idx)

            cache_position = torch.arange(
                prefix_len, prefix_len + suffix_len, device=device
            )

            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                outputs = model(
                    input_ids=suffix_input,
                    position_ids=position_ids,
                    attention_mask=mask,
                    cache_position=cache_position,
                    past_key_values=suffix_cache,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            total_time += time.time() - start

            for j, task_idx in enumerate(batch_indices):
                task = tasks[task_idx]
                answer_start = task["prompt_len"]
                answer_tokens = task["answer_tokens"]
                for k, expected_token in enumerate(answer_tokens):
                    pred_pos_in_full = answer_start - 1 + k
                    pred_pos_in_suffix = pred_pos_in_full - prefix_len
                    if pred_pos_in_suffix >= 0 and pred_pos_in_suffix < suffix_len:
                        pred_token = (
                            outputs.logits[j, pred_pos_in_suffix].argmax().item()
                        )
                        results[task_idx][k] = pred_token == expected_token

            del suffix_cache
        torch.cuda.empty_cache()

    final_results = []
    for idx, task in enumerate(tasks):
        answer_len = len(task["answer_tokens"])
        verifs = results[idx]
        all_correct = all(verifs.get(k, False) for k in range(answer_len))
        final_results.append(all_correct)

    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return final_results, total_time, peak_memory_mb


# ============================================================
# Method 3: SDPATrie with SDPA (4D Mask)
# ============================================================
def run_sdpatrie_verification(
    model,
    tokenizer,
    tasks: List[dict],
    device: str,
    dtype,
    max_nodes_per_batch: int = 4096,
) -> Tuple[List[bool], float, float]:
    """
    SDPATrie verification using SDPA with custom 4D attention mask.

    Returns:
        Tuple of (results, total_time, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()

    # Use the shared implementation from sdpatrie.py
    trie_results, total_time = run_sdpatrie_sdpa_verification(
        model,
        tokenizer,
        tasks,
        device,
        dtype,
        max_nodes_per_batch=max_nodes_per_batch,
        prompt_len_key="prompt_len",
        return_time=True,
    )

    # Convert TrieVerificationResult to bool
    results = [r.correct for r in trie_results]

    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results, total_time, peak_memory_mb


# ============================================================
# Method 4: FlashTree with Triton
# ============================================================
def run_flashtree_verification(
    model,
    tokenizer,
    tasks: List[dict],
    device: str,
    dtype,
    max_nodes_per_batch: int = 4096,
) -> Tuple[List[bool], float, float]:
    """
    FlashTree: Triton FlashAttention kernel.

    Returns:
        Tuple of (results, total_time, peak_memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()

    trie_results, total_time = _run_flashtree_verification(
        model,
        tokenizer,
        tasks,
        device,
        dtype,
        max_nodes_per_batch=max_nodes_per_batch,
        prompt_len_key="prompt_len",
        return_time=True,
    )

    # Convert TrieVerificationResult to bool
    results = [r.correct for r in trie_results]

    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results, total_time, peak_memory_mb


# ============================================================
# Warmup Configuration
# ============================================================
WARMUP_ITERATIONS = 3


# ============================================================
# Main Evaluation
# ============================================================
def evaluate_model(
    model_id: str,
    range_val: int,
    device: str,
    output_dir: str,
    batch_size: int = 64,
    max_nodes_per_batch: int = 4096,
) -> Dict:
    """Evaluate a single model with all methods."""
    print(f"\n{'='*70}")
    print(f"Model: {model_id}")
    print(f"Range: 1-{range_val} ({range_val}^2 = {range_val**2} tasks)")
    print(f"{'='*70}")

    # Load model for SDPA-based methods (naive, prefill, sdpatrie)
    model_sdpa, tokenizer, dtype = setup_model(model_id, device)
    tasks = generate_tasks(1, range_val, tokenizer)
    print(f"Tasks: {len(tasks)}")

    warmup_range = min(30, range_val)
    warmup_tasks = generate_tasks(1, warmup_range, tokenizer)

    # Warmup SDPA model
    print(
        f"Warmup SDPA model ({WARMUP_ITERATIONS} iterations with {len(warmup_tasks)} tasks)..."
    )
    for i in range(WARMUP_ITERATIONS):
        _ = run_naive_verification(
            model_sdpa, tokenizer, warmup_tasks, device, dtype, batch_size=batch_size
        )
        torch.cuda.synchronize()
    torch.cuda.empty_cache()

    results = {}

    # 1. Naive
    print("\n[1/4] Naive batch verification...")
    naive_results, naive_time, naive_mem = run_naive_verification(
        model_sdpa, tokenizer, tasks, device, dtype, batch_size=batch_size
    )
    results["naive"] = {
        "results": naive_results,
        "time": naive_time,
        "memory_mb": naive_mem,
        "correct": sum(naive_results),
    }
    print(
        f"  Time: {naive_time*1000:.1f}ms, Memory: {naive_mem:.1f}MB, "
        f"Correct: {results['naive']['correct']}/{len(tasks)}"
    )
    torch.cuda.empty_cache()

    # 2. Prefill
    print("\n[2/4] Prefill verification...")
    prefill_results, prefill_time, prefill_mem = run_prefill_verification(
        model_sdpa, tokenizer, tasks, device, dtype, batch_size=batch_size
    )
    results["prefill"] = {
        "results": prefill_results,
        "time": prefill_time,
        "memory_mb": prefill_mem,
        "correct": sum(prefill_results),
    }
    print(
        f"  Time: {prefill_time*1000:.1f}ms, Memory: {prefill_mem:.1f}MB, "
        f"Correct: {results['prefill']['correct']}/{len(tasks)}"
    )
    torch.cuda.empty_cache()

    # 3. SDPATrie
    print("\n[3/4] SDPATrie verification...")
    print(
        f"  Warmup SDPATrie ({WARMUP_ITERATIONS} iterations with {len(warmup_tasks)} tasks)..."
    )
    for i in range(WARMUP_ITERATIONS):
        _ = run_sdpatrie_verification(
            model_sdpa,
            tokenizer,
            warmup_tasks,
            device,
            dtype,
            max_nodes_per_batch=max_nodes_per_batch,
        )
        torch.cuda.synchronize()
    torch.cuda.empty_cache()

    sdpatrie_results, sdpatrie_time, sdpatrie_mem = run_sdpatrie_verification(
        model_sdpa,
        tokenizer,
        tasks,
        device,
        dtype,
        max_nodes_per_batch=max_nodes_per_batch,
    )
    results["sdpatrie"] = {
        "results": sdpatrie_results,
        "time": sdpatrie_time,
        "memory_mb": sdpatrie_mem,
        "correct": sum(sdpatrie_results),
    }
    print(
        f"  Time: {sdpatrie_time*1000:.1f}ms, Memory: {sdpatrie_mem:.1f}MB, "
        f"Correct: {results['sdpatrie']['correct']}/{len(tasks)}"
    )

    # Cleanup SDPA model
    del model_sdpa
    gc.collect()
    torch.cuda.empty_cache()

    # Load model for Triton-based methods (flashtree)
    print("\n  Loading model for FlashTree (Triton)...")
    model_triton, _, _ = setup_model(model_id, device)
    patched = patch_qwen2_attention_for_treeflash(model_triton)
    print(f"  Patched {patched} attention modules for Triton")

    # Warmup for Triton model
    print(
        f"  Warmup Triton model ({WARMUP_ITERATIONS} iterations with {len(warmup_tasks)} tasks for JIT)..."
    )
    for i in range(WARMUP_ITERATIONS):
        _ = run_flashtree_verification(
            model_triton,
            tokenizer,
            warmup_tasks,
            device,
            dtype,
            max_nodes_per_batch=max_nodes_per_batch,
        )
        torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 4. FlashTree (Triton)
    print("\n[4/4] FlashTree Triton verification...")
    print(
        f"  Warmup FlashTree ({WARMUP_ITERATIONS} iterations with {len(warmup_tasks)} tasks)..."
    )
    for i in range(WARMUP_ITERATIONS):
        _ = run_flashtree_verification(
            model_triton,
            tokenizer,
            warmup_tasks,
            device,
            dtype,
            max_nodes_per_batch=max_nodes_per_batch,
        )
        torch.cuda.synchronize()
    torch.cuda.empty_cache()

    flashtree_results, flashtree_time, flashtree_mem = run_flashtree_verification(
        model_triton,
        tokenizer,
        tasks,
        device,
        dtype,
        max_nodes_per_batch=max_nodes_per_batch,
    )
    results["flashtree"] = {
        "results": flashtree_results,
        "time": flashtree_time,
        "memory_mb": flashtree_mem,
        "correct": sum(flashtree_results),
    }
    print(
        f"  Time: {flashtree_time*1000:.1f}ms, Memory: {flashtree_mem:.1f}MB, "
        f"Correct: {results['flashtree']['correct']}/{len(tasks)}"
    )

    # Cleanup
    del model_triton
    gc.collect()
    torch.cuda.empty_cache()

    # Verify correctness
    print("\n--- Correctness Check ---")
    all_methods = ["naive", "prefill", "sdpatrie", "flashtree"]

    # Helper to extract bool from result (TrieVerificationResult or bool)
    def get_bool(r):
        if hasattr(r, "correct"):
            return r.correct
        return r

    for method in all_methods[1:]:
        match = sum(
            1
            for a, b in zip(results["naive"]["results"], results[method]["results"])
            if get_bool(a) == get_bool(b)
        )
        print(
            f"  naive vs {method}: {match}/{len(tasks)} ({100*match/len(tasks):.2f}%)"
        )

    # Calculate speedups
    speedups = {}
    for method in all_methods[1:]:
        speedups[method] = (
            naive_time / results[method]["time"]
            if results[method]["time"] > 0
            else float("inf")
        )

    # Summary
    summary = {
        "model_id": model_id,
        "range": range_val,
        "total_tasks": len(tasks),
        "naive_time_ms": naive_time * 1000,
        "prefill_time_ms": prefill_time * 1000,
        "sdpatrie_time_ms": sdpatrie_time * 1000,
        "flashtree_time_ms": flashtree_time * 1000,
        # Memory (MB)
        "naive_memory_mb": results["naive"]["memory_mb"],
        "prefill_memory_mb": results["prefill"]["memory_mb"],
        "sdpatrie_memory_mb": results["sdpatrie"]["memory_mb"],
        "flashtree_memory_mb": results["flashtree"]["memory_mb"],
        # Speedups
        "speedup_prefill": speedups["prefill"],
        "speedup_sdpatrie": speedups["sdpatrie"],
        "speedup_flashtree": speedups["flashtree"],
        # Correct counts
        "naive_correct": results["naive"]["correct"],
        "prefill_correct": results["prefill"]["correct"],
        "sdpatrie_correct": results["sdpatrie"]["correct"],
        "flashtree_correct": results["flashtree"]["correct"],
    }

    # Match rates (use get_bool to handle TrieVerificationResult objects)
    for method in all_methods[1:]:
        match = sum(
            1
            for a, b in zip(results["naive"]["results"], results[method]["results"])
            if get_bool(a) == get_bool(b)
        )
        summary[f"naive_{method}_match"] = match
        summary[f"naive_{method}_match_pct"] = 100 * match / len(tasks)

    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(
        f"{'Method':<20} {'Time(ms)':>10} {'Memory(MB)':>12} "
        f"{'Speedup':>10} {'Correct':>10} {'vs Naive':>10}"
    )
    print("-" * 90)
    print(
        f"{'Naive':<20} {naive_time*1000:>10.1f} "
        f"{results['naive']['memory_mb']:>12.1f} {1.0:>10.2f}x "
        f"{results['naive']['correct']:>10} {100.0:>9.1f}%"
    )
    print(
        f"{'Prefill':<20} {prefill_time*1000:>10.1f} "
        f"{results['prefill']['memory_mb']:>12.1f} {speedups['prefill']:>10.2f}x "
        f"{results['prefill']['correct']:>10} {summary['naive_prefill_match_pct']:>9.1f}%"
    )
    print(
        f"{'SDPATrie':<20} {sdpatrie_time*1000:>10.1f} "
        f"{results['sdpatrie']['memory_mb']:>12.1f} {speedups['sdpatrie']:>10.2f}x "
        f"{results['sdpatrie']['correct']:>10} {summary['naive_sdpatrie_match_pct']:>9.1f}%"
    )
    print(
        f"{'FlashTree (Triton)':<20} {flashtree_time*1000:>10.1f} "
        f"{results['flashtree']['memory_mb']:>12.1f} {speedups['flashtree']:>10.2f}x "
        f"{results['flashtree']['correct']:>10} {summary['naive_flashtree_match_pct']:>9.1f}%"
    )
    print(f"{'='*90}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark")
    parser.add_argument(
        "--model", type=str, default=None, help="Specific model to evaluate"
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument("--range", type=int, default=99, help="Range for tasks (n*n)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    # Triton kernel configuration
    parser.add_argument(
        "--block-m", type=int, default=16, help="BLOCK_M for Triton kernel"
    )
    parser.add_argument(
        "--num-warps", type=int, default=2, help="num_warps for Triton kernel"
    )
    parser.add_argument(
        "--num-stages", type=int, default=2, help="num_stages for Triton kernel"
    )
    # Batch size configuration
    parser.add_argument(
        "--max-nodes-per-batch",
        type=int,
        default=4096,
        help="max_nodes_per_batch for Trie methods",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="batch_size for baseline methods"
    )
    args = parser.parse_args()

    # Resolve paths relative to script directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / ".." / "data"
    output_dir = output_dir.resolve()

    set_all_seeds(RANDOM_SEED)

    # Set Triton kernel configuration
    set_triton_config(
        block_m=args.block_m, num_warps=args.num_warps, num_stages=args.num_stages
    )

    if args.all:
        models = MODELS
    elif args.model:
        models = [args.model]
    else:
        models = [MODELS[0]]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_id in models:
        try:
            result = evaluate_model(
                model_id,
                args.range,
                args.device,
                str(output_dir),
                batch_size=args.batch_size,
                max_nodes_per_batch=args.max_nodes_per_batch,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error with {model_id}: {e}")
            import traceback

            traceback.print_exc()

    if all_results:
        # Save results
        filepath = output_dir / "speedtest_results.json"
        with open(filepath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {filepath}")

        # Final summary
        print("\n" + "=" * 110)
        print("FINAL SUMMARY - ALL MODELS")
        print("=" * 110)
        header = (
            f"{'Model':<25} {'Naive':>8} {'Prefill':>8} {'SDPATrie':>8} "
            f"{'FlashTree':>8} {'FT↑':>6}"
        )
        print(header)
        print("-" * 110)
        for r in all_results:
            name = r["model_id"].split("/")[-1].replace("-Instruct", "")
            print(
                f"{name:<25} {r['naive_time_ms']:>7.0f}ms {r['prefill_time_ms']:>7.0f}ms "
                f"{r['sdpatrie_time_ms']:>7.0f}ms {r['flashtree_time_ms']:>7.0f}ms "
                f"{r['speedup_flashtree']:>5.1f}x"
            )
        print("-" * 110)

        # Memory summary
        print("\nPeak Memory Usage (MB):")
        mem_header = (
            f"{'Model':<25} {'Naive':>10} {'Prefill':>12} "
            f"{'SDPATrie':>10} {'FlashTree':>10}"
        )
        print(mem_header)
        print("-" * 70)
        for r in all_results:
            name = r["model_id"].split("/")[-1].replace("-Instruct", "")
            print(
                f"{name:<25} {r['naive_memory_mb']:>10.1f} {r['prefill_memory_mb']:>12.1f} "
                f"{r['sdpatrie_memory_mb']:>10.1f} {r['flashtree_memory_mb']:>10.1f}"
            )
        print("=" * 70)


if __name__ == "__main__":
    main()
