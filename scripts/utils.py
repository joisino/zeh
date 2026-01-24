#!/usr/bin/env python3
"""Common utilities for distribute scripts."""

import os
import random
import re
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

RANDOM_SEED = 42
SYSTEM_CALC = "Answer with only the integer."
USER_FMT = "{a}*{b}="

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_model(model_id: str, device: str = "cuda"):
    """Load model and tokenizer."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def setup_model_with_dtype(model_id: str, device: str = "cuda"):
    """Load model and tokenizer, also returning dtype."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {model_id} dtype={dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer, dtype


def build_prompt(tokenizer, a: int, b: int) -> str:
    """Build prompt with chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_CALC},
        {"role": "user", "content": USER_FMT.format(a=a, b=b)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompt_with_template(
    tokenizer, a: int, b: int, system: str, user_fmt: str
) -> str:
    """Build prompt with custom system/user template."""
    user_content = user_fmt.format(a=a, b=b)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_number(text: str) -> Optional[int]:
    """Extract the first number from text."""
    text = text.strip()
    match = re.match(r"^[\s]*(-?[\d,]+)", text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return int(num_str)
        except ValueError:
            pass
    return None


def generate_tasks(range_start: int, range_end: int, tokenizer) -> List[dict]:
    """Generate multiplication tasks for the given range."""
    tasks = []
    for a in range(range_start, range_end + 1):
        for b in range(range_start, range_end + 1):
            prompt = build_prompt(tokenizer, a, b)
            answer = str(a * b)

            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
            full_tokens = prompt_tokens + answer_tokens

            tasks.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "a": a,
                    "b": b,
                    "prompt_len": len(prompt_tokens),
                    "answer_tokens": answer_tokens,
                    "full_tokens": full_tokens,
                }
            )
    return tasks


def generate_tasks_for_n(n: int, tokenizer) -> List[dict]:
    """Generate all multiplication tasks for a given N (1 to N range)."""
    tasks = []
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            prompt = build_prompt(tokenizer, a, b)
            answer = str(a * b)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
            full_tokens = prompt_tokens + answer_tokens

            tasks.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "a": a,
                    "b": b,
                    "prompt_tokens": prompt_tokens,
                    "answer_tokens": answer_tokens,
                    "full_tokens": full_tokens,
                    "prompt_len": len(prompt_tokens),
                }
            )
    return tasks


def generate_tasks_for_n_border(n: int, tokenizer) -> List[dict]:
    """Generate only border tasks for N (tasks involving N)."""
    tasks = []
    # Row N: (N, 1), (N, 2), ..., (N, N)
    for b in range(1, n + 1):
        prompt = build_prompt(tokenizer, n, b)
        answer = str(n * b)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        full_tokens = prompt_tokens + answer_tokens

        tasks.append(
            {
                "prompt": prompt,
                "answer": answer,
                "a": n,
                "b": b,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                "full_tokens": full_tokens,
                "prompt_len": len(prompt_tokens),
            }
        )
    # Column N: (1, N), (2, N), ..., (N-1, N)
    for a in range(1, n):
        prompt = build_prompt(tokenizer, a, n)
        answer = str(a * n)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        full_tokens = prompt_tokens + answer_tokens

        tasks.append(
            {
                "prompt": prompt,
                "answer": answer,
                "a": a,
                "b": n,
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                "full_tokens": full_tokens,
                "prompt_len": len(prompt_tokens),
            }
        )
    return tasks


def get_digit_tokens(tokenizer) -> set:
    """Get the set of token IDs that represent single digits (0-9)."""
    digit_tokens = set()
    for digit in "0123456789":
        token_ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(token_ids) == 1:
            digit_tokens.add(token_ids[0])
    return digit_tokens


def is_digit_token(token_id: Optional[int], digit_tokens: set) -> bool:
    """Check if a token is a digit token."""
    if token_id is None:
        return False
    return token_id in digit_tokens
