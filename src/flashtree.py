#!/usr/bin/env python3
"""
FlashTree: Triton-based Sparse Tree Attention Kernel (GQA-Native)
==================================================================

This module provides a Triton-based sparse attention kernel for tree-structured
attention patterns. Used to accelerate Trie-based verification.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
from transformers.cache_utils import DynamicCache

__all__ = [
    "treeflash_attn_kernel_gqa",
    "triton_sparse_attention_gqa",
    "set_treeflash_ctx",
    "get_treeflash_ctx",
    "TreeFlashForwardWrapper",
    "patch_qwen2_attention_for_treeflash",
    "build_sparse_idx_tensor",
    "clone_dynamic_cache",
    "set_triton_config",
    "get_triton_config",
    "run_flashtree_verification",
]


# ============================================================
# Triton Kernel Configuration
# ============================================================
_TRITON_CONFIG = {
    "BLOCK_M": 16,
    "num_warps": 2,
    "num_stages": 2,
}


def set_triton_config(
    block_m: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> None:
    """Set Triton kernel parameters."""
    global _TRITON_CONFIG
    if block_m is not None:
        assert block_m > 0 and (block_m & (block_m - 1)) == 0
        _TRITON_CONFIG["BLOCK_M"] = block_m
    if num_warps is not None:
        assert num_warps > 0 and (num_warps & (num_warps - 1)) == 0
        _TRITON_CONFIG["num_warps"] = num_warps
    if num_stages is not None:
        assert num_stages > 0
        _TRITON_CONFIG["num_stages"] = num_stages


def get_triton_config() -> dict:
    """Get current Triton kernel configuration."""
    return _TRITON_CONFIG.copy()


# ============================================================
# GQA-Native Triton Kernel
# ============================================================
@triton.jit
def treeflash_attn_kernel_gqa(
    Q_ptr,
    K_ptr,
    V_ptr,
    OUT_ptr,
    IDX_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_K: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    GQA-native Triton kernel: reads K/V without physical replication.

    Q: [B, H, M, D] - queries (full heads)
    K, V: [B, H_KV, N, D] - keys/values (KV heads, NOT repeated)
    IDX: [M, MAX_K] int32 - sparse pattern, -1 for invalid
    OUT: [B, H, M, D] - output
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H
    kv_h = h * H_KV // H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(
        tl.float32
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    has_valid = tl.zeros((BLOCK_M,), tl.int32)

    for j in range(MAX_K):
        idx_ptrs = IDX_ptr + offs_m * MAX_K + j
        idx = tl.load(idx_ptrs, mask=m_mask, other=-1)

        valid = (idx >= 0) & (idx < N)
        has_valid = has_valid | tl.where(valid, 1, 0)
        idx_safe = tl.where(valid, idx, 0)

        k_ptrs = (
            K_ptr
            + b * stride_kb
            + kv_h * stride_kh
            + idx_safe[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(
            k_ptrs, mask=m_mask[:, None] & d_mask[None, :] & valid[:, None], other=0.0
        ).to(tl.float32)

        v_ptrs = (
            V_ptr
            + b * stride_vb
            + kv_h * stride_vh
            + idx_safe[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(
            v_ptrs, mask=m_mask[:, None] & d_mask[None, :] & valid[:, None], other=0.0
        ).to(tl.float32)

        score = tl.sum(q * k, axis=1) * SM_SCALE
        score = tl.where(valid & m_mask, score, float("-inf"))

        m_new = tl.maximum(m_i, score)
        m_new_valid = m_new > float("-inf")
        alpha = tl.where(m_new_valid, tl.exp(m_i - m_new), 0.0)
        p = tl.where(valid & m_mask, tl.exp(score - m_new), 0.0)

        acc = acc * alpha[:, None] + v * p[:, None]
        l_i = l_i * alpha + p
        m_i = m_new

    safe_l = tl.where(has_valid > 0, l_i, 1.0)
    safe_l = tl.maximum(safe_l, 1e-9)
    out = acc / safe_l[:, None]
    out = tl.where(has_valid[:, None] > 0, out, 0.0)

    out_ptrs = (
        OUT_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :])


def triton_sparse_attention_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx: torch.Tensor,
    kv_len: Optional[int] = None,
) -> torch.Tensor:
    """
    GQA-native sparse attention: K/V are NOT repeated.

    Args:
        q: [B, H, M, D] float16 - queries (full heads)
        k: [B, H_KV, N, D] float16 - keys (KV heads, NOT repeated)
        v: [B, H_KV, N, D] float16 - values (KV heads, NOT repeated)
        idx: [M, MAX_K] int32 - sparse indices, -1 for padding
        kv_len: Optional explicit KV length for bounds checking

    Returns:
        out: [B, H, M, D] float16
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and idx.is_cuda
    assert q.dtype == torch.float16
    assert idx.dtype == torch.int32

    idx = idx.contiguous()

    B, H, M, D = q.shape
    _, H_KV, N, _ = k.shape
    MAX_K = idx.shape[1]

    assert H % H_KV == 0

    # Note: kv_len parameter is available for bounds checking if needed

    config = get_triton_config()
    BLOCK_M = config["BLOCK_M"]
    num_warps = config["num_warps"]
    num_stages = config["num_stages"]

    BLOCK_D = triton.next_power_of_2(D)
    assert D <= 128

    sm_scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    treeflash_attn_kernel_gqa[grid](
        q,
        k,
        v,
        out,
        idx,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        M=M,
        N=N,
        H=H,
        H_KV=H_KV,
        HEAD_DIM=D,
        MAX_K=MAX_K,
        SM_SCALE=sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,  # type: ignore[call-arg]
        num_stages=num_stages,  # type: ignore[call-arg]
    )

    return out


# ============================================================
# Global Context for TreeFlash
# ============================================================
_TREEFLASH_CTX = {
    "active": False,
    "idx": None,
    "kv_len": None,
    "use_gqa_kernel": True,
}


def set_treeflash_ctx(
    active: bool,
    idx: Optional[torch.Tensor],
    kv_len: Optional[int] = None,
    use_gqa_kernel: bool = True,
) -> None:
    """Set the TreeFlash context for attention computation."""
    global _TREEFLASH_CTX
    _TREEFLASH_CTX["active"] = active
    _TREEFLASH_CTX["idx"] = idx
    _TREEFLASH_CTX["kv_len"] = kv_len
    _TREEFLASH_CTX["use_gqa_kernel"] = use_gqa_kernel


def get_treeflash_ctx() -> Dict:
    """Get the current TreeFlash context."""
    return _TREEFLASH_CTX


# ============================================================
# Utility Functions
# ============================================================
def clone_dynamic_cache(prefix_cache: DynamicCache) -> DynamicCache:
    """Clone a DynamicCache to avoid modifying the original."""
    out = DynamicCache()
    for layer_idx in range(len(prefix_cache)):
        k, v = prefix_cache[layer_idx]
        out.update(k.clone(), v.clone(), layer_idx)
    return out


def build_sparse_idx_tensor(
    prefix_len: int, ancestors: List[List[int]], device: str
) -> torch.Tensor:
    """
    Build sparse index tensor for Triton kernel.

    IDX[i, :] contains key positions that query i can attend to:
      - All prefix positions: 0, 1, ..., prefix_len-1
      - Ancestor positions in suffix: prefix_len + ancestors[i]

    Returns:
        [M, MAX_K] int32 tensor, padded with -1
    """
    M = len(ancestors)
    idx_lists = []
    max_k = 0

    for anc in ancestors:
        allowed = list(range(prefix_len)) + [prefix_len + j for j in anc]
        idx_lists.append(allowed)
        max_k = max(max_k, len(allowed))

    if max_k <= 0:
        max_k_pad = 16
    else:
        max_k_pad = 2 ** math.ceil(math.log2(max(max_k, 16)))
        if max_k_pad > 4096:
            max_k_pad = ((max_k + 63) // 64) * 64

    arr = -1 * np.ones((M, max_k_pad), dtype=np.int32)
    for i, allowed in enumerate(idx_lists):
        arr[i, : len(allowed)] = np.asarray(allowed, dtype=np.int32)

    return torch.from_numpy(arr).to(device=device)


# ============================================================
# Monkey Patch for Qwen2 Attention
# ============================================================
class TreeFlashForwardWrapper:
    """Wrapper that replaces Qwen2Attention.forward with TreeFlash."""

    def __init__(
        self, module, orig_fwd, num_kv_groups, hdim, hsize, lidx, apply_rotary_fn
    ):
        self.module = module
        self.orig_fwd = orig_fwd
        self.num_kv_groups = num_kv_groups
        self.hdim = hdim
        self.hsize = hsize
        self.lidx = lidx
        self.apply_rotary = apply_rotary_fn

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ctx = get_treeflash_ctx()

        if not ctx["active"]:
            return self.orig_fwd(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        mod = self.module
        bsz, q_len, _ = hidden_states.size()

        query_states = mod.q_proj(hidden_states)
        key_states = mod.k_proj(hidden_states)
        value_states = mod.v_proj(hidden_states)

        num_heads = query_states.shape[-1] // self.hdim
        num_kv_heads = key_states.shape[-1] // self.hdim

        query_states = query_states.view(bsz, q_len, num_heads, self.hdim).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, q_len, num_kv_heads, self.hdim).transpose(
            1, 2
        )
        value_states = value_states.view(bsz, q_len, num_kv_heads, self.hdim).transpose(
            1, 2
        )

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.lidx, cache_kwargs
            )

        idx = ctx["idx"]
        kv_len = ctx.get("kv_len")

        attn_output = triton_sparse_attention_gqa(
            query_states, key_states, value_states, idx, kv_len=kv_len
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hsize)
        attn_output = mod.o_proj(attn_output)

        return attn_output, None


def patch_qwen2_attention_for_treeflash(model) -> int:
    """Patch Qwen2Attention modules to use TreeFlash."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    patched = 0

    for name, module in model.named_modules():
        if not (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
        ):
            continue
        if not hasattr(module, "head_dim"):
            continue
        if not hasattr(module, "layer_idx"):
            continue
        if getattr(module, "_treeflash_patched", False):
            continue

        orig_forward = module.forward

        config = model.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        num_key_value_groups = num_heads // num_kv_heads
        head_dim = module.head_dim
        hidden_size = config.hidden_size
        layer_idx = module.layer_idx

        wrapper = TreeFlashForwardWrapper(
            module,
            orig_forward,
            num_key_value_groups,
            head_dim,
            hidden_size,
            layer_idx,
            apply_rotary_pos_emb,
        )

        module._orig_forward = orig_forward
        module.forward = wrapper
        module._treeflash_patched = True
        patched += 1

    return patched


# ============================================================
# FlashTree Verification
# ============================================================
def run_flashtree_verification(
    model,
    tokenizer,
    tasks: List[dict],
    device: str,
    dtype: torch.dtype,
    max_nodes_per_batch: int = 4096,
    prompt_len_key: str = "prompt_len",
    return_time: bool = False,
):
    """
    FlashTree verification using Triton kernel.

    Args:
        model: HuggingFace model (must be patched with patch_qwen2_attention_for_treeflash)
        tokenizer: HuggingFace tokenizer
        tasks: List of task dicts with "full_tokens", "answer_tokens", and prompt_len_key
        device: Device to run on
        dtype: Data type for computations
        max_nodes_per_batch: Maximum nodes per batch for trie traversal
        prompt_len_key: Key for prompt length in task dicts
        return_time: If True, return (results, total_time) else just results

    Returns:
        List of TrieVerificationResult, or Tuple of (results, total_time) if return_time=True
    """
    import time

    from sdpatrie import TokenTrie  # type: ignore[import-not-found]
    from sdpatrie import (
        TrieVerificationResult,
        find_common_prefix_len,
        flatten_trie_fine_grained,
    )

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

    # Prefix prefill (TreeFlash disabled)
    set_treeflash_ctx(active=False, idx=None)
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

    # Generate batches using fine-grained DFS traversal
    batches = flatten_trie_fine_grained(
        trie.root, max_nodes_per_batch, return_ancestor_pairs=True
    )

    # Suffix processing with FlashTree (Triton kernel)
    for batch_data in batches:
        tokens, depths, ancestors, leaf_info, verify_info, ancestor_pairs = batch_data
        num_nodes = len(tokens)

        if num_nodes == 0:
            continue

        input_ids = torch.tensor([tokens], device=device)
        position_ids = torch.tensor(
            [[prefix_len + d - 1 for d in depths]], device=device
        )

        idx_tensor = build_sparse_idx_tensor(prefix_len, ancestors, device=device)
        kv_len = prefix_len + num_nodes

        subtree_cache = clone_dynamic_cache(prefix_cache)
        cache_position = torch.arange(prefix_len, prefix_len + num_nodes, device=device)

        set_treeflash_ctx(active=True, idx=idx_tensor, kv_len=kv_len)

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=subtree_cache,
                use_cache=True,
            )

        torch.cuda.synchronize()
        total_time += time.time() - start

        set_treeflash_ctx(active=False, idx=None)

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

        del subtree_cache, idx_tensor

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
