#!/usr/bin/env python3
"""
SDPATrie: Trie-based Efficient Verification with SDPA
=====================================================

This module provides the Trie data structure for efficient batch verification.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class TrieVerificationResult:
    """Result of trie-based verification with detailed token information."""

    correct: bool
    first_mismatch_position: Optional[int] = None  # Position in answer tokens
    first_mismatch_predicted: Optional[int] = None  # Predicted token ID
    first_mismatch_expected: Optional[int] = None  # Expected token ID


class TrieNode:
    """Node in the Trie structure."""

    __slots__ = ["token_id", "children", "task_indices", "depth", "verify_positions"]

    def __init__(self, token_id: int = -1, depth: int = 0):
        self.token_id = token_id
        self.children: Dict[int, "TrieNode"] = {}
        self.task_indices = set()
        self.depth = depth
        self.verify_positions: List[Tuple[int, int]] = []


class TokenTrie:
    """Trie structure for tokenized sequences with verification positions."""

    def __init__(self):
        self.root = TrieNode()
        self.node_count = 0

    def insert(
        self, tokens: List[int], task_idx: int, answer_start_depth: int, answer_len: int
    ) -> None:
        """Insert a token sequence into the trie."""
        node = self.root
        for i, token_id in enumerate(tokens):
            depth = i + 1

            if token_id not in node.children:
                new_node = TrieNode(token_id, depth)
                node.children[token_id] = new_node
                self.node_count += 1

            child = node.children[token_id]

            if depth > answer_start_depth and depth <= answer_start_depth + answer_len:
                answer_token_idx = depth - answer_start_depth - 1
                child.verify_positions.append((task_idx, answer_token_idx))

            node = child

        node.task_indices.add(task_idx)


def count_nodes(node: TrieNode) -> int:
    """Count total nodes in subtree."""
    count = 1
    for child in node.children.values():
        count += count_nodes(child)
    return count


def find_common_prefix_len(all_token_ids: List[List[int]]) -> int:
    """Find the length of the common prefix among all token sequences."""
    if not all_token_ids:
        return 0

    min_len = min(len(ids) for ids in all_token_ids)
    common_prefix_len = 0

    for i in range(min_len):
        first_token = all_token_ids[0][i]
        if all(ids[i] == first_token for ids in all_token_ids):
            common_prefix_len = i + 1
        else:
            break

    return common_prefix_len


def flatten_trie_fine_grained(
    root: TrieNode, max_nodes_per_batch: int, return_ancestor_pairs: bool = False
) -> List[Tuple]:
    """
    Fine-grained batching using DFS traversal with strict max_nodes_per_batch limit.
    """
    if not root.children:
        return []

    batches = []
    batch_tokens: List[int] = []
    batch_depths: List[int] = []
    batch_ancestors: List[List[int]] = []
    batch_leaf_info: List[Tuple[int, int]] = []
    batch_verify_info: List[Tuple[int, int, int, int]] = []
    batch_ancestor_pairs: Optional[List[Tuple[int, int]]] = (
        [] if return_ancestor_pairs else None
    )
    node_to_batch_idx: Dict[int, int] = {}
    globally_processed_nodes: set = set()

    def finalize_batch():
        nonlocal batch_tokens, batch_depths, batch_ancestors, batch_leaf_info
        nonlocal batch_verify_info, batch_ancestor_pairs, node_to_batch_idx

        if batch_tokens:
            if return_ancestor_pairs:
                batches.append(
                    (
                        batch_tokens,
                        batch_depths,
                        batch_ancestors,
                        batch_leaf_info,
                        batch_verify_info,
                        batch_ancestor_pairs,
                    )
                )
            else:
                batches.append(
                    (
                        batch_tokens,
                        batch_depths,
                        batch_ancestors,
                        batch_leaf_info,
                        batch_verify_info,
                    )
                )

        batch_tokens = []
        batch_depths = []
        batch_ancestors = []
        batch_leaf_info = []
        batch_verify_info = []
        batch_ancestor_pairs = [] if return_ancestor_pairs else None  # type: ignore
        node_to_batch_idx = {}

    def add_node_to_batch(
        node: TrieNode, path: List[TrieNode], path_indices: List[int], is_new: bool
    ) -> int:
        nonlocal batch_tokens, batch_depths, batch_ancestors, batch_leaf_info
        nonlocal batch_verify_info, batch_ancestor_pairs, node_to_batch_idx

        node_id = id(node)

        if node_id in node_to_batch_idx:
            return node_to_batch_idx[node_id]

        flat_idx = len(batch_tokens)
        node_to_batch_idx[node_id] = flat_idx

        batch_tokens.append(node.token_id)
        batch_depths.append(node.depth)

        my_ancestors = path_indices + [flat_idx]
        batch_ancestors.append(my_ancestors)

        if return_ancestor_pairs and batch_ancestor_pairs is not None:
            for j in my_ancestors:
                batch_ancestor_pairs.append((flat_idx, j))

        if is_new:
            if path_indices:
                parent_flat_idx = path_indices[-1]
                for task_idx, answer_token_idx in node.verify_positions:
                    batch_verify_info.append(
                        (task_idx, answer_token_idx, parent_flat_idx, node.token_id)
                    )

            for task_idx in node.task_indices:
                batch_leaf_info.append((task_idx, flat_idx))

            globally_processed_nodes.add(node_id)

        return flat_idx

    def ensure_path_in_batch(path: List[TrieNode]) -> List[int]:
        batch_indices = []
        for i, node in enumerate(path):
            node_id = id(node)
            is_new = node_id not in globally_processed_nodes
            idx = add_node_to_batch(node, path[:i], batch_indices.copy(), is_new)
            batch_indices.append(idx)
        return batch_indices

    def dfs(node: TrieNode, path: List[TrieNode]):
        nonlocal batch_tokens

        nodes_to_add = sum(1 for n in path if id(n) not in node_to_batch_idx)
        current_count = len(batch_tokens)

        if current_count > 0 and current_count + nodes_to_add > max_nodes_per_batch:
            finalize_batch()

        _ = ensure_path_in_batch(path)

        for child_token in sorted(node.children.keys()):
            child = node.children[child_token]
            dfs(child, path + [child])

    for first_token in sorted(root.children.keys()):
        first_child = root.children[first_token]
        dfs(first_child, [first_child])

    finalize_batch()

    return batches


def build_sdpatrie_attention_mask(
    num_nodes: int,
    prefix_len: int,
    ancestors: List[List[int]],
    dtype: torch.dtype,
    device,
    ancestor_pairs: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Build 4D attention mask for SDPATrie SDPA."""
    total_kv_len = prefix_len + num_nodes
    mask_value = torch.finfo(dtype).min

    mask = torch.full((num_nodes, total_kv_len), mask_value, dtype=dtype, device=device)
    mask[:, :prefix_len] = 0.0

    if ancestor_pairs:
        row_indices = torch.tensor(
            [p[0] for p in ancestor_pairs], dtype=torch.long, device=device
        )
        col_indices = torch.tensor(
            [prefix_len + p[1] for p in ancestor_pairs], dtype=torch.long, device=device
        )
        mask[row_indices, col_indices] = 0.0
    else:
        for i, anc_list in enumerate(ancestors):
            for j in anc_list:
                mask[i, prefix_len + j] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)


def run_sdpatrie_sdpa_verification(
    model,
    tokenizer,
    tasks: List[dict],
    device: str,
    dtype: torch.dtype,
    max_nodes_per_batch: int = 2200,
    prompt_len_key: str = "prompt_len",
    return_time: bool = False,
):
    """
    SDPATrie verification using SDPA with custom 4D attention mask.
    """
    import time

    if not tasks:
        return ([], 0.0) if return_time else []

    total_time = 0.0

    all_token_ids = [task["full_tokens"] for task in tasks]
    prefix_len = find_common_prefix_len(all_token_ids)
    common_prefix = all_token_ids[0][:prefix_len]

    trie = TokenTrie()
    for idx, task in enumerate(tasks):
        suffix = task["full_tokens"][prefix_len:]
        answer_start_in_full = task[prompt_len_key]
        answer_start_depth = answer_start_in_full - prefix_len
        answer_len = len(task["answer_tokens"])
        trie.insert(suffix, idx, answer_start_depth, answer_len)

    task_verifications = {idx: {} for idx in range(len(tasks))}

    prefix_input = torch.tensor([common_prefix], device=device)
    prefix_cache_position = torch.arange(prefix_len, device=device)
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

    # task_verifications stores (predicted_token, expected_token) tuples
    for idx, task in enumerate(tasks):
        answer_start = task[prompt_len_key]
        answer_tokens = task["answer_tokens"]

        for k, expected_token in enumerate(answer_tokens):
            pred_pos_in_full = answer_start - 1 + k
            if pred_pos_in_full < prefix_len:
                pred_token = prefix_output.logits[0, pred_pos_in_full].argmax().item()
                task_verifications[idx][k] = (pred_token, expected_token)

    batches = flatten_trie_fine_grained(
        trie.root, max_nodes_per_batch, return_ancestor_pairs=True
    )
    prefix_kv = [
        (prefix_cache[i][0], prefix_cache[i][1]) for i in range(len(prefix_cache))
    ]

    for batch_data in batches:
        tokens, depths, ancestors, leaf_info, verify_info, ancestor_pairs = batch_data
        num_nodes = len(tokens)

        if num_nodes == 0:
            continue

        input_ids = torch.tensor([tokens], device=device)
        position_ids = torch.tensor(
            [[prefix_len + d - 1 for d in depths]], device=device
        )

        mask = build_sdpatrie_attention_mask(
            num_nodes, prefix_len, ancestors, dtype, device, ancestor_pairs
        )

        subtree_cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(prefix_kv):
            subtree_cache.update(key.clone(), value.clone(), layer_idx)

        cache_position = torch.arange(prefix_len, prefix_len + num_nodes, device=device)

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=mask,
                cache_position=cache_position,
                past_key_values=subtree_cache,
                use_cache=True,
            )

        torch.cuda.synchronize()
        total_time += time.time() - start

        if verify_info:
            pred_indices = torch.tensor(
                [v[2] for v in verify_info], dtype=torch.long, device=device
            )
            pred_tokens = outputs.logits[0, pred_indices].argmax(dim=-1).cpu().tolist()

            for i, (
                task_idx,
                answer_token_idx,
                pred_flat_idx,
                expected_token,
            ) in enumerate(verify_info):
                task_verifications[task_idx][answer_token_idx] = (
                    pred_tokens[i],
                    expected_token,
                )

        del subtree_cache, mask

    # Build results with detailed mismatch info
    results = []
    for idx, task in enumerate(tasks):
        answer_len = len(task["answer_tokens"])
        verifs = task_verifications[idx]

        result = TrieVerificationResult(correct=True)
        for k in range(answer_len):
            if k not in verifs:
                # Missing verification - treat as incorrect
                result = TrieVerificationResult(
                    correct=False,
                    first_mismatch_position=k,
                    first_mismatch_predicted=None,
                    first_mismatch_expected=task["answer_tokens"][k],
                )
                break
            pred_token, expected_token = verifs[k]
            if pred_token != expected_token:
                result = TrieVerificationResult(
                    correct=False,
                    first_mismatch_position=k,
                    first_mismatch_predicted=pred_token,
                    first_mismatch_expected=expected_token,
                )
                break
        results.append(result)

    if return_time:
        return results, total_time
    else:
        return results
