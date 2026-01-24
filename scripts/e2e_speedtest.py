#!/usr/bin/env python3
"""
End-to-End ZEH Evaluation Benchmark
===================================

This script measures end-to-end runtime for computing Zero-Error Horizon (ZEH),
comparing multiple verification methods for multiplication tasks.

ZEH is the largest N such that the model answers all a*b correctly for 1 <= a,b <= N.

Methods compared:
1. Naive: Auto-regressive decoding (incremental), stop at first failure
2. Naive + LA: Auto-regressive decoding with look ahead (batch multiple N's)
3. TF: Teacher forcing (incremental)
4. TF + LA: Teacher forcing with look ahead
5. Trie (SDPA): Trie with SDPA 4D mask (uses TF, LA, and prefilling)
6. FlashTree: Triton kernel (uses TF, LA, and prefilling)

Note: Speedup is relative to Naive (auto-regressive decoding, incremental).

Usage:
    uv run python e2e_speedtest.py --model Qwen/Qwen2.5-0.5B-Instruct --max-n 99
    uv run python e2e_speedtest.py --all --max-n 99
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add scripts directory to path for local imports
sys.path.insert(0, str(SCRIPT_DIR))
# Add src directory to path for local imports
sys.path.insert(0, str(SCRIPT_DIR / ".." / "src"))

from utils import (
    MODELS,
    RANDOM_SEED,
    extract_number,
    generate_tasks_for_n,
    generate_tasks_for_n_border,
    get_digit_tokens,
    is_digit_token,
    set_all_seeds,
)
from utils import setup_model_with_dtype as setup_model  # noqa: E402

from flashtree import patch_qwen2_attention_for_treeflash  # noqa: E402  # type: ignore
from flashtree import run_flashtree_verification  # noqa: E402  # type: ignore
from flashtree import set_triton_config  # noqa: E402  # type: ignore
from sdpatrie import TrieVerificationResult  # noqa: E402  # type: ignore
from sdpatrie import run_sdpatrie_sdpa_verification  # noqa: E402  # type: ignore

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Naive Generation (Baseline 1)
# ============================================================
def run_naive_generation(
    model, tokenizer, tasks: List[dict], device: str, batch_size: int = 256
) -> List[bool]:
    """Run naive greedy generation for a batch of tasks."""
    if not tasks:
        return []

    # Group by prompt length for efficient batching
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
                    repetition_penalty=1.0,
                )

            input_len = inputs.input_ids.shape[1]
            for j, idx in enumerate(indices):
                generated = outputs[j][input_len:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                predicted = extract_number(text)
                expected = int(batch_group[j][1]["answer"])
                results[idx] = predicted == expected

    # All values should be filled; cast to List[bool]
    return [r if r is not None else False for r in results]


# ============================================================
# Naive Batch Verification (Baseline 2)
# ============================================================
@dataclass
class BatchVerificationResult:
    """Result of batch verification with detailed token information."""

    correct: bool
    first_mismatch_position: Optional[int] = None  # Position in answer tokens
    first_mismatch_predicted: Optional[int] = None  # Predicted token ID
    first_mismatch_expected: Optional[int] = None  # Expected token ID


def run_naive_batch(
    model, tokenizer, tasks: List[dict], device: str
) -> List[BatchVerificationResult]:
    """Batch verification: check if model's predicted tokens match expected answer.

    Returns detailed results including the first mismatched token information.
    """
    if not tasks:
        return []

    # Group by total sequence length for efficient batching
    batch_groups: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        total_len = len(task["full_tokens"])
        batch_groups[total_len].append((idx, task))

    results: List[Optional[BatchVerificationResult]] = [None] * len(tasks)

    for seq_len in sorted(batch_groups.keys()):
        group = batch_groups[seq_len]

        # Process all tasks with same sequence length in one batch
        input_ids_list = [task["full_tokens"] for _, task in group]
        input_ids = torch.tensor(input_ids_list, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits

        for j, (task_idx, task) in enumerate(group):
            prompt_len = task["prompt_len"]
            answer_tokens = task["answer_tokens"]

            result = BatchVerificationResult(correct=True)
            for k, expected_token in enumerate(answer_tokens):
                pred_pos = prompt_len - 1 + k
                pred_token = logits[j, pred_pos].argmax().item()

                if pred_token != expected_token:
                    result = BatchVerificationResult(
                        correct=False,
                        first_mismatch_position=k,
                        first_mismatch_predicted=pred_token,
                        first_mismatch_expected=expected_token,
                    )
                    break

            results[task_idx] = result

    # All values should be filled; provide default for any missing
    return [
        r if r is not None else BatchVerificationResult(correct=False) for r in results
    ]


# ============================================================
# SDPATrie Verification with SDPA
# ============================================================
def run_sdpatrie_verification(
    model,
    tokenizer,
    tasks: List[dict],
    device: str,
    dtype,
    max_nodes_per_batch: int = 2048,
) -> List[TrieVerificationResult]:
    """SDPATrie verification using SDPA with custom 4D attention mask.

    Returns TrieVerificationResult with detailed mismatch information.
    """
    result = run_sdpatrie_sdpa_verification(
        model,
        tokenizer,
        tasks,
        device,
        dtype,
        max_nodes_per_batch=max_nodes_per_batch,
        prompt_len_key="prompt_len",
        return_time=False,
    )
    # Convert tuple result to list
    return list(result)


# ============================================================
# ZEH Finding Functions
# ============================================================
def find_zeh_naive_generation(
    model, tokenizer, max_n: int, device: str, batch_size: int = 256
) -> Tuple[int, float, float, Dict]:
    """Find ZEH using naive generation (incremental)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    found_zeh = 0
    total_tasks_evaluated = 0
    failed_tasks = []

    for n in range(1, max_n + 1):
        border_tasks = generate_tasks_for_n_border(n, tokenizer)
        results = run_naive_generation(
            model, tokenizer, border_tasks, device, batch_size=batch_size
        )

        total_tasks_evaluated += len(border_tasks)

        all_passed = all(results)

        if all_passed:
            found_zeh = n
        else:
            failed_tasks = [border_tasks[i] for i, r in enumerate(results) if not r]
            break
    else:
        found_zeh = max_n

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = {
        "total_tasks_evaluated": total_tasks_evaluated,
        "failed_tasks": [
            {"a": t["a"], "b": t["b"], "answer": t["answer"]} for t in failed_tasks[:5]
        ],
    }

    return found_zeh, total_time, peak_memory_mb, stats


def find_zeh_naive_generation_lookahead(
    model, tokenizer, max_n: int, device: str, batch_size: int = 256
) -> Tuple[int, float, float, Dict]:
    """Find ZEH using naive generation with fixed batch size lookahead.

    Tasks are generated on-demand (not all at once) and processed in fixed-size batches.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    found_zeh = 0
    total_tasks_evaluated = 0
    failed_tasks = []

    # Generate tasks on-demand: fill batch, then verify
    current_batch = []
    current_n = 1
    done = False

    while current_n <= max_n and not done:
        # Fill batch with tasks until batch_size is reached
        while len(current_batch) < batch_size and current_n <= max_n:
            border_tasks = generate_tasks_for_n_border(current_n, tokenizer)
            for task in border_tasks:
                task["n"] = current_n
                current_batch.append(task)
            current_n += 1

        if not current_batch:
            break

        # Take batch_size tasks (or all remaining)
        batch_tasks = current_batch[:batch_size]
        current_batch = current_batch[batch_size:]
        total_tasks_evaluated += len(batch_tasks)

        results = run_naive_generation(
            model, tokenizer, batch_tasks, device, batch_size=batch_size
        )

        # Check results in n order
        batch_ns = sorted(set(t["n"] for t in batch_tasks))
        for curr_n in batch_ns:
            n_indices = [i for i, t in enumerate(batch_tasks) if t["n"] == curr_n]
            n_results = [results[i] for i in n_indices]
            n_tasks = [batch_tasks[i] for i in n_indices]

            if all(n_results):
                found_zeh = curr_n
            else:
                failed_tasks = [n_tasks[i] for i, r in enumerate(n_results) if not r]
                done = True
                break

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = {
        "total_tasks_evaluated": total_tasks_evaluated,
        "batch_size": batch_size,
        "failed_tasks": [
            {"a": t["a"], "b": t["b"], "answer": t["answer"]} for t in failed_tasks[:5]
        ],
    }

    return found_zeh, total_time, peak_memory_mb, stats


def find_zeh_naive_batch(
    model, tokenizer, max_n: int, device: str
) -> Tuple[int, float, float, Dict]:
    """Find ZEH using naive batch with smart fallback for ambiguous cases.

    Fallback strategy:
    - If the first mismatched predicted token is a digit, treat as definite failure (no fallback)
    - If the first mismatched predicted token is NOT a digit, treat as ambiguous (fallback)
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    digit_tokens = get_digit_tokens(tokenizer)

    found_zeh = 0
    total_tasks_evaluated = 0
    batch_verify_time = 0.0
    generation_verify_time = 0.0
    batch_false_negatives = 0
    failed_tasks = []

    for n in range(1, max_n + 1):
        border_tasks = generate_tasks_for_n_border(n, tokenizer)
        total_tasks_evaluated += len(border_tasks)

        batch_start = time.time()
        batch_results = run_naive_batch(model, tokenizer, border_tasks, device)
        batch_verify_time += time.time() - batch_start

        if all(r.correct for r in batch_results):
            found_zeh = n
            continue

        # Classify failures: definite failures vs ambiguous cases
        definite_failures = []
        ambiguous_tasks = []

        for i, r in enumerate(batch_results):
            if not r.correct:
                if is_digit_token(r.first_mismatch_predicted, digit_tokens):
                    # Predicted a wrong digit - definite failure
                    definite_failures.append(border_tasks[i])
                else:
                    # Predicted non-digit - ambiguous, need fallback
                    ambiguous_tasks.append(border_tasks[i])

        if definite_failures:
            # Found at least one definite failure
            failed_tasks = definite_failures
            break

        # Fallback verification for ambiguous cases only
        if ambiguous_tasks:
            gen_start = time.time()
            gen_results = run_naive_generation(
                model,
                tokenizer,
                ambiguous_tasks,
                device,
                batch_size=len(ambiguous_tasks),
            )
            generation_verify_time += time.time() - gen_start

            true_failures = [
                ambiguous_tasks[i] for i, r in enumerate(gen_results) if not r
            ]

            if true_failures:
                failed_tasks = true_failures
                break
            else:
                batch_false_negatives += len(ambiguous_tasks)

        found_zeh = n
    else:
        found_zeh = max_n

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = {
        "total_tasks_evaluated": total_tasks_evaluated,
        "batch_verify_time": batch_verify_time,
        "generation_verify_time": generation_verify_time,
        "batch_false_negatives": batch_false_negatives,
        "failed_tasks": [
            {"a": t["a"], "b": t["b"], "answer": t["answer"]} for t in failed_tasks[:5]
        ],
    }

    return found_zeh, total_time, peak_memory_mb, stats


def find_zeh_naive_batch_lookahead(
    model, tokenizer, max_n: int, device: str, batch_size: int = 256
) -> Tuple[int, float, float, Dict]:
    """Find ZEH using naive batch with fixed batch size lookahead and smart fallback.

    Tasks are generated on-demand (not all at once) and processed in fixed-size batches.

    Fallback strategy:
    - If the first mismatched predicted token is a digit, treat as definite failure (no fallback)
    - If the first mismatched predicted token is NOT a digit, treat as ambiguous (fallback)
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    digit_tokens = get_digit_tokens(tokenizer)

    found_zeh = 0
    total_tasks_evaluated = 0
    batch_verify_time = 0.0
    generation_verify_time = 0.0
    batch_false_negatives = 0
    failed_tasks = []

    # Generate tasks on-demand: fill batch, then verify
    current_batch = []
    current_n = 1
    done = False

    while current_n <= max_n and not done:
        # Fill batch with tasks until batch_size is reached
        while len(current_batch) < batch_size and current_n <= max_n:
            border_tasks = generate_tasks_for_n_border(current_n, tokenizer)
            for task in border_tasks:
                task["n"] = current_n
                current_batch.append(task)
            current_n += 1

        if not current_batch:
            break

        # Take batch_size tasks (or all remaining)
        batch_tasks = current_batch[:batch_size]
        current_batch = current_batch[batch_size:]
        total_tasks_evaluated += len(batch_tasks)

        verify_start = time.time()
        batch_results = run_naive_batch(model, tokenizer, batch_tasks, device)
        batch_verify_time += time.time() - verify_start

        # Check results in n order
        batch_ns = sorted(set(t["n"] for t in batch_tasks))
        for curr_n in batch_ns:
            n_indices = [i for i, t in enumerate(batch_tasks) if t["n"] == curr_n]
            n_results = [batch_results[i] for i in n_indices]
            n_tasks = [batch_tasks[i] for i in n_indices]

            if all(r.correct for r in n_results):
                found_zeh = curr_n
            else:
                # Classify failures: definite failures vs ambiguous cases
                definite_failures = []
                ambiguous_tasks_list = []

                for i, r in enumerate(n_results):
                    if not r.correct:
                        if is_digit_token(r.first_mismatch_predicted, digit_tokens):
                            # Predicted a wrong digit - definite failure
                            definite_failures.append(n_tasks[i])
                        else:
                            # Predicted non-digit - ambiguous, need fallback
                            ambiguous_tasks_list.append(n_tasks[i])

                if definite_failures:
                    # Found at least one definite failure
                    failed_tasks = definite_failures
                    done = True
                    break

                # Fallback verification for ambiguous cases only
                if ambiguous_tasks_list:
                    gen_start = time.time()
                    gen_results = run_naive_generation(
                        model,
                        tokenizer,
                        ambiguous_tasks_list,
                        device,
                        batch_size=len(ambiguous_tasks_list),
                    )
                    generation_verify_time += time.time() - gen_start

                    true_failures = [
                        ambiguous_tasks_list[i]
                        for i, r in enumerate(gen_results)
                        if not r
                    ]

                    if true_failures:
                        failed_tasks = true_failures
                        done = True
                        break
                    else:
                        batch_false_negatives += len(ambiguous_tasks_list)

                found_zeh = curr_n

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = {
        "total_tasks_evaluated": total_tasks_evaluated,
        "batch_size": batch_size,
        "batch_verify_time": batch_verify_time,
        "generation_verify_time": generation_verify_time,
        "batch_false_negatives": batch_false_negatives,
        "failed_tasks": [
            {"a": t["a"], "b": t["b"], "answer": t["answer"]} for t in failed_tasks[:5]
        ],
    }

    return found_zeh, total_time, peak_memory_mb, stats


def find_zeh_sdpatrie_lookahead(
    model,
    tokenizer,
    max_n: int,
    device: str,
    dtype,
    batch_size: int = 256,
    max_nodes_per_batch: int = 4096,
) -> Tuple[int, float, float, Dict]:
    """Find ZEH using SDPATrie with fixed batch size lookahead and smart fallback.

    Tasks are generated on-demand (not all at once) and processed in fixed-size batches.

    Fallback strategy:
    - If the first mismatched predicted token is a digit, treat as definite failure (no fallback)
    - If the first mismatched predicted token is NOT a digit, treat as ambiguous (fallback)
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    digit_tokens = get_digit_tokens(tokenizer)

    found_zeh = 0
    total_tasks_evaluated = 0
    trie_verify_time = 0.0
    generation_verify_time = 0.0
    trie_false_negatives = 0
    failed_tasks = []

    # Generate tasks on-demand: fill batch, then verify
    current_batch = []
    current_n = 1
    done = False

    while current_n <= max_n and not done:
        # Fill batch with tasks until batch_size is reached
        while len(current_batch) < batch_size and current_n <= max_n:
            border_tasks = generate_tasks_for_n_border(current_n, tokenizer)
            for task in border_tasks:
                task["n"] = current_n
                current_batch.append(task)
            current_n += 1

        if not current_batch:
            break

        # Take batch_size tasks (or all remaining)
        batch_tasks = current_batch[:batch_size]
        current_batch = current_batch[batch_size:]
        total_tasks_evaluated += len(batch_tasks)

        trie_start = time.time()
        trie_results = run_sdpatrie_verification(
            model,
            tokenizer,
            batch_tasks,
            device,
            dtype,
            max_nodes_per_batch=max_nodes_per_batch,
        )
        trie_verify_time += time.time() - trie_start

        # Check results in n order
        batch_ns = sorted(set(t["n"] for t in batch_tasks))
        for curr_n in batch_ns:
            n_indices = [i for i, t in enumerate(batch_tasks) if t["n"] == curr_n]
            n_results = [trie_results[i] for i in n_indices]
            n_tasks = [batch_tasks[i] for i in n_indices]

            if all(r.correct for r in n_results):
                found_zeh = curr_n
            else:
                # Classify failures: definite failures vs ambiguous cases
                definite_failures = []
                ambiguous_tasks_list = []

                for i, r in enumerate(n_results):
                    if not r.correct:
                        if r.first_mismatch_predicted is not None and is_digit_token(
                            r.first_mismatch_predicted, digit_tokens
                        ):
                            # Predicted a wrong digit - definite failure
                            definite_failures.append(n_tasks[i])
                        else:
                            # Predicted non-digit or missing - ambiguous, need fallback
                            ambiguous_tasks_list.append(n_tasks[i])

                if definite_failures:
                    # Found at least one definite failure
                    failed_tasks = definite_failures
                    done = True
                    break

                # Fallback verification for ambiguous cases only
                if ambiguous_tasks_list:
                    gen_start = time.time()
                    gen_results = run_naive_generation(
                        model,
                        tokenizer,
                        ambiguous_tasks_list,
                        device,
                        batch_size=len(ambiguous_tasks_list),
                    )
                    generation_verify_time += time.time() - gen_start

                    true_failures = [
                        ambiguous_tasks_list[i]
                        for i, r in enumerate(gen_results)
                        if not r
                    ]

                    if true_failures:
                        failed_tasks = true_failures
                        done = True
                        break
                    else:
                        trie_false_negatives += len(ambiguous_tasks_list)

                found_zeh = curr_n

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = {
        "total_tasks_evaluated": total_tasks_evaluated,
        "batch_size": batch_size,
        "trie_verify_time": trie_verify_time,
        "generation_verify_time": generation_verify_time,
        "trie_false_negatives": trie_false_negatives,
        "failed_tasks": [
            {"a": t["a"], "b": t["b"], "answer": t["answer"]} for t in failed_tasks[:5]
        ],
    }

    return found_zeh, total_time, peak_memory_mb, stats


def find_zeh_flashtree_lookahead(
    model,
    tokenizer,
    max_n: int,
    device: str,
    dtype,
    batch_size: int = 256,
    max_nodes_per_batch: int = 4096,
) -> Tuple[int, float, float, Dict]:
    """Find ZEH using FlashTree with fixed batch size lookahead and smart fallback.

    Tasks are ordered by n and processed in fixed-size batches.
    Tasks are generated on-demand to avoid unnecessary task generation overhead.

    Fallback strategy:
    - If the first mismatched predicted token is a digit, treat as definite failure (no fallback)
    - If the first mismatched predicted token is NOT a digit, treat as ambiguous (fallback)
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    digit_tokens = get_digit_tokens(tokenizer)

    found_zeh = 0
    total_tasks_evaluated = 0
    flashtree_verify_time = 0.0
    generation_verify_time = 0.0
    flashtree_false_negatives = 0
    failed_tasks = []

    # Generate tasks on-demand: fill batch, then verify
    current_batch = []
    current_n = 1
    done = False

    while current_n <= max_n and not done:
        # Fill batch with tasks until batch_size is reached
        while len(current_batch) < batch_size and current_n <= max_n:
            border_tasks = generate_tasks_for_n_border(current_n, tokenizer)
            for task in border_tasks:
                task["n"] = current_n
                current_batch.append(task)
            current_n += 1

        batch_tasks = current_batch[:batch_size]
        current_batch = current_batch[batch_size:]

        if not batch_tasks:
            break

        total_tasks_evaluated += len(batch_tasks)

        ft_start = time.time()
        flashtree_results = run_flashtree_verification(
            model,
            tokenizer,
            batch_tasks,
            device,
            dtype,
            max_nodes_per_batch=max_nodes_per_batch,
        )
        flashtree_verify_time += time.time() - ft_start

        # Check results in n order
        batch_ns = sorted(set(t["n"] for t in batch_tasks))
        for curr_n in batch_ns:
            n_indices = [i for i, t in enumerate(batch_tasks) if t["n"] == curr_n]
            n_results = [flashtree_results[i] for i in n_indices]
            n_tasks = [batch_tasks[i] for i in n_indices]

            if all(r.correct for r in n_results):
                found_zeh = curr_n
            else:
                # Classify failures: definite failures vs ambiguous cases
                definite_failures = []
                ambiguous_tasks_list = []

                for i, r in enumerate(n_results):
                    if not r.correct:
                        if r.first_mismatch_predicted is not None and is_digit_token(
                            r.first_mismatch_predicted, digit_tokens
                        ):
                            # Predicted a wrong digit - definite failure
                            definite_failures.append(n_tasks[i])
                        else:
                            # Predicted non-digit or missing - ambiguous, need fallback
                            ambiguous_tasks_list.append(n_tasks[i])

                if definite_failures:
                    # Found at least one definite failure
                    failed_tasks = definite_failures
                    done = True
                    break

                # Fallback verification for ambiguous cases only
                if ambiguous_tasks_list:
                    gen_start = time.time()
                    gen_results = run_naive_generation(
                        model,
                        tokenizer,
                        ambiguous_tasks_list,
                        device,
                        batch_size=len(ambiguous_tasks_list),
                    )
                    generation_verify_time += time.time() - gen_start

                    true_failures = [
                        ambiguous_tasks_list[i]
                        for i, r in enumerate(gen_results)
                        if not r
                    ]

                    if true_failures:
                        failed_tasks = true_failures
                        done = True
                        break
                    else:
                        flashtree_false_negatives += len(ambiguous_tasks_list)

                found_zeh = curr_n

        if done:
            break

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = {
        "total_tasks_evaluated": total_tasks_evaluated,
        "batch_size": batch_size,
        "flashtree_verify_time": flashtree_verify_time,
        "generation_verify_time": generation_verify_time,
        "flashtree_false_negatives": flashtree_false_negatives,
        "failed_tasks": [
            {"a": t["a"], "b": t["b"], "answer": t["answer"]} for t in failed_tasks[:5]
        ],
    }

    return found_zeh, total_time, peak_memory_mb, stats


# ============================================================
# Main Evaluation
# ============================================================
def evaluate_model(
    model_id: str,
    max_n: int,
    device: str,
    batch_size: int = 256,
    max_nodes_per_batch: int = 4096,
    lookahead_batch_size: int = 256,
) -> Dict:
    """Evaluate model with fair baseline comparison."""
    # Load model WITHOUT patching first for baseline measurements
    model, tokenizer, dtype = setup_model(model_id, device)

    print(f"\n{'='*80}")
    print(f"Evaluating: {model_id}")
    print(f"Max N: {max_n}")
    print(f"{'='*80}")

    # Warmup for baseline (unpatched model)
    print("Warming up baseline (unpatched model)...")
    warmup_tasks = generate_tasks_for_n(min(30, max_n), tokenizer)
    _ = run_naive_generation(
        model, tokenizer, warmup_tasks[:10], device, batch_size=batch_size
    )
    _ = run_naive_batch(model, tokenizer, warmup_tasks, device)
    torch.cuda.empty_cache()

    # ====================
    # Baseline measurements (UNPATCHED MODEL)
    # ====================
    print("\n--- Baseline Measurements (Unpatched Model) ---")

    print("  Running Naive Generation (incremental)...")
    naive_gen_zeh, naive_gen_time, naive_gen_mem, naive_gen_stats = (
        find_zeh_naive_generation(
            model, tokenizer, max_n, device, batch_size=batch_size
        )
    )
    torch.cuda.empty_cache()

    print(
        f"  Running Naive Generation (lookahead batch_size={lookahead_batch_size})..."
    )
    naive_gen_la_zeh, naive_gen_la_time, naive_gen_la_mem, naive_gen_la_stats = (
        find_zeh_naive_generation_lookahead(
            model, tokenizer, max_n, device, batch_size=lookahead_batch_size
        )
    )
    torch.cuda.empty_cache()

    print("  Running Naive Batch (incremental)...")
    naive_batch_zeh, naive_batch_time, naive_batch_mem, naive_batch_stats = (
        find_zeh_naive_batch(model, tokenizer, max_n, device)
    )
    torch.cuda.empty_cache()

    print(f"  Running Naive Batch (lookahead batch_size={lookahead_batch_size})...")
    (
        naive_batch_la_zeh,
        naive_batch_la_time,
        naive_batch_la_mem,
        naive_batch_la_stats,
    ) = find_zeh_naive_batch_lookahead(
        model, tokenizer, max_n, device, batch_size=lookahead_batch_size
    )
    torch.cuda.empty_cache()

    # ====================
    # Now patch model for FlashTree and SDPATrie
    # ====================
    print("\n--- Patching model for FlashTree ---")
    patched = patch_qwen2_attention_for_treeflash(model)
    print(f"Patched {patched} attention layers for FlashTree")

    print("Generating warmup tasks...")
    warmup_full_tasks = []
    for n in range(1, max_n + 1):
        border_tasks = generate_tasks_for_n_border(n, tokenizer)
        for task in border_tasks:
            task["n"] = n
        warmup_full_tasks.extend(border_tasks)

    print(
        f"Warming up patched model (3 iterations with {len(warmup_full_tasks)} tasks)..."
    )
    for i in range(3):
        _ = run_sdpatrie_verification(
            model,
            tokenizer,
            warmup_full_tasks,
            device,
            dtype,
            max_nodes_per_batch=max_nodes_per_batch,
        )
        _ = run_flashtree_verification(
            model,
            tokenizer,
            warmup_full_tasks,
            device,
            dtype,
            max_nodes_per_batch=max_nodes_per_batch,
        )
    torch.cuda.empty_cache()

    # ====================
    # SDPATrie and FlashTree measurements (PATCHED MODEL)
    # ====================
    print("\n--- FlashTree Measurements (Patched Model) ---")

    print(f"  Running SDPATrie (SDPA, lookahead batch_size={lookahead_batch_size})...")
    trie_zeh, trie_time, trie_mem, trie_stats = find_zeh_sdpatrie_lookahead(
        model,
        tokenizer,
        max_n,
        device,
        dtype,
        batch_size=lookahead_batch_size,
        max_nodes_per_batch=max_nodes_per_batch,
    )
    torch.cuda.empty_cache()

    print(
        f"  Running FlashTree (Triton, lookahead batch_size={lookahead_batch_size})..."
    )
    flashtree_zeh, flashtree_time, flashtree_mem, flashtree_stats = (
        find_zeh_flashtree_lookahead(
            model,
            tokenizer,
            max_n,
            device,
            dtype,
            batch_size=lookahead_batch_size,
            max_nodes_per_batch=max_nodes_per_batch,
        )
    )
    torch.cuda.empty_cache()

    # Build result dict
    result = {
        "model_id": model_id,
        "max_n_limit": max_n,
        # Times
        "naive_gen_time": naive_gen_time,
        "naive_gen_la_time": naive_gen_la_time,
        "naive_batch_time": naive_batch_time,
        "naive_batch_la_time": naive_batch_la_time,
        "sdpatrie_time": trie_time,
        "flashtree_time": flashtree_time,
        # Memory (MB)
        "naive_gen_memory_mb": naive_gen_mem,
        "naive_gen_la_memory_mb": naive_gen_la_mem,
        "naive_batch_memory_mb": naive_batch_mem,
        "naive_batch_la_memory_mb": naive_batch_la_mem,
        "sdpatrie_memory_mb": trie_mem,
        "flashtree_memory_mb": flashtree_mem,
        # ZEH values
        "naive_gen_zeh": naive_gen_zeh,
        "naive_gen_la_zeh": naive_gen_la_zeh,
        "naive_batch_zeh": naive_batch_zeh,
        "naive_batch_la_zeh": naive_batch_la_zeh,
        "sdpatrie_zeh": trie_zeh,
        "flashtree_zeh": flashtree_zeh,
        # Speedups (vs Naive Generation)
        "speedup_vs_naive_gen": (
            naive_gen_time / trie_time if trie_time > 0 else float("inf")
        ),
        "speedup_vs_naive_batch": (
            naive_batch_time / trie_time if trie_time > 0 else float("inf")
        ),
        "flashtree_speedup_vs_naive_gen": (
            naive_gen_time / flashtree_time if flashtree_time > 0 else float("inf")
        ),
        "flashtree_speedup_vs_naive_batch": (
            naive_batch_time / flashtree_time if flashtree_time > 0 else float("inf")
        ),
        "speedup_trie_vs_flashtree": (
            trie_time / flashtree_time if flashtree_time > 0 else float("inf")
        ),
        # Stats
        "naive_gen_stats": naive_gen_stats,
        "flashtree_stats": flashtree_stats,
        # Match check
        "zeh_match": (
            naive_gen_zeh
            == naive_gen_la_zeh
            == naive_batch_zeh
            == naive_batch_la_zeh
            == trie_zeh
            == flashtree_zeh
        ),
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_id.split('/')[-1]}")
    print(f"{'='*80}")
    print()
    print(f"{'Method':<45} {'ZEH':>6} {'Time':>10} {'Memory':>12}")
    print("-" * 77)
    print(
        f"{'Naive Generation (incremental)':<45} "
        f"{naive_gen_zeh:>6} {naive_gen_time:>10.3f}s {naive_gen_mem:>10.1f}MB"
    )
    la_method = f"Naive Generation (lookahead bs={lookahead_batch_size})"
    print(
        f"{la_method:<45} "
        f"{naive_gen_la_zeh:>6} {naive_gen_la_time:>10.3f}s {naive_gen_la_mem:>10.1f}MB"
    )
    print(
        f"{'Naive Batch (incremental)':<45} "
        f"{naive_batch_zeh:>6} {naive_batch_time:>10.3f}s {naive_batch_mem:>10.1f}MB"
    )
    la_batch_method = f"Naive Batch (lookahead bs={lookahead_batch_size})"
    print(
        f"{la_batch_method:<45} "
        f"{naive_batch_la_zeh:>6} {naive_batch_la_time:>10.3f}s {naive_batch_la_mem:>10.1f}MB"
    )
    trie_method = f"SDPATrie (SDPA, bs={lookahead_batch_size})"
    print(
        f"{trie_method:<45} " f"{trie_zeh:>6} {trie_time:>10.3f}s {trie_mem:>10.1f}MB"
    )
    ft_method = f"FlashTree (Triton, bs={lookahead_batch_size})"
    print(
        f"{ft_method:<45} "
        f"{flashtree_zeh:>6} {flashtree_time:>10.3f}s {flashtree_mem:>10.1f}MB"
    )
    print()
    print("Speedups vs Naive Generation:")
    print(f"  SDPATrie:   {result['speedup_vs_naive_gen']:.2f}x")
    print(f"  FlashTree:   {result['flashtree_speedup_vs_naive_gen']:.2f}x")
    print()
    print(f"FlashTree vs SDPATrie: {result['speedup_trie_vs_flashtree']:.2f}x")
    print()
    zeh_match_str = "✓" if result["zeh_match"] else "✗"
    print(f"ZEH Match (all methods): {zeh_match_str}")

    # Print first failure example if available
    first_failure = None
    for stats_key in ["naive_gen_stats", "naive_batch_stats", "flashtree_stats"]:
        if stats_key in result and result[stats_key].get("failed_tasks"):
            first_failure = result[stats_key]["failed_tasks"][0]
            break

    if first_failure:
        print()
        a_val = first_failure["a"]
        b_val = first_failure["b"]
        print(f"First Failure Example: {a_val}×{b_val}={a_val * b_val} (expected)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="ZEH Evaluation")
    parser.add_argument("--model", type=str, default=None, help="Model to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument(
        "--max-n", type=int, default=99, help="Maximum N for ZEH search"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
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
    parser.add_argument(
        "--lookahead-batch-size",
        type=int,
        default=256,
        help="batch_size for lookahead methods (tasks per batch)",
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

    results = []
    for model_id in models:
        try:
            result = evaluate_model(
                model_id,
                args.max_n,
                args.device,
                batch_size=args.batch_size,
                max_nodes_per_batch=args.max_nodes_per_batch,
                lookahead_batch_size=args.lookahead_batch_size,
            )
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_id}: {e}")
            import traceback

            traceback.print_exc()

    if results:
        # Print summary
        print("\n" + "=" * 160)
        print("FINAL SUMMARY")
        print("=" * 160)
        header = (
            f"{'Model':<25} {'NaiveGen':>10} {'NaiveGenLA':>12} "
            f"{'NaiveBatch':>12} {'NaiveBatchLA':>14} {'SDPATrie':>10} "
            f"{'FlashTree':>10} {'ZEH':>6} {'Match':>6}"
        )
        print(header)
        print("-" * 160)

        for r in results:
            model_name = r["model_id"].split("/")[-1].replace("-Instruct", "")
            zeh_check = "✓" if r["zeh_match"] else "✗"
            print(
                f"{model_name:<25} {r['naive_gen_time']:>10.2f} {r['naive_gen_la_time']:>12.2f} "
                f"{r['naive_batch_time']:>12.2f} {r['naive_batch_la_time']:>14.2f} "
                f"{r['sdpatrie_time']:>10.2f} {r['flashtree_time']:>10.2f} "
                f"{r['naive_gen_zeh']:>6} {zeh_check:>6}"
            )

        print("-" * 160)

        # Save results
        filepath = output_dir / "zeh_results.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    main()
