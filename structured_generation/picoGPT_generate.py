"""
Constrained generation with picoGPT + FSM masking
==================================================

Combines:
  - picoGPT's pure-NumPy GPT-2 forward pass (MIT license, by Jay Mody)
    https://github.com/jaymody/picoGPT
  - FSM-based constrained decoding from coalescence.py

Weights are loaded directly as NumPy arrays via safetensors — no PyTorch
or TensorFlow needed.
"""

import re
import time

import numpy as np
from transformers import AutoTokenizer

from coalescence import build_tokenizer_index, JSON_SIMPLE, JSON_VERY_COMPLEX
from utils import load_gpt2_params


# ──────────────────────────────────────────────────────────────────────────────
# picoGPT forward pass (pure NumPy)
# Original: https://github.com/jaymody/picoGPT by Jay Mody (MIT License)
# ──────────────────────────────────────────────────────────────────────────────


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)


def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(
        map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1))
    )
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    return linear(np.hstack(out_heads), **c_proj)


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_unconstrained(
    input_ids, params, n_head, max_tokens=40, temperature=0.0, tokenizer=None
):
    """Autoregressive generation without constraints.

    If *tokenizer* is provided, generation stops as soon as the top-level
    JSON closing brace ``}`` is emitted (brace-depth tracking).

    Parameters
    ----------
    input_ids : list[int]
        Prompt token IDs.  Must NOT be mutated.
    params : dict
        GPT-2 parameters (wte, wpe, blocks, ln_f).
    n_head : int
        Number of attention heads.
    max_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature.  0.0 means greedy (argmax).
    tokenizer : optional
        If provided, enables brace-depth stopping.  Must have a
        ``.decode([token_id])`` method.

    Returns
    -------
    list[int]
        The generated token IDs (excluding the prompt).

    Algorithm
    ---------
    1. Copy ``input_ids`` into a working list ``inputs``.
    2. For each step (up to ``max_tokens``):
       a. Run ``gpt2(inputs, **params, n_head=n_head)`` to get logits.
       b. Take the last row of logits (``logits[-1]``).
       c. If ``temperature == 0.0``, pick ``argmax``; otherwise sample
          from ``softmax(logits / temperature)``.
       d. Append the chosen token to ``inputs``.
       e. If ``tokenizer`` is provided, decode the token and track
          brace depth (``{`` increments, ``}`` decrements).  Set a
          ``json_started`` flag on the first ``{``.  Stop when
          ``json_started`` and ``brace_depth <= 0``.
    3. Return ``inputs[len(input_ids):]``.
    """
    # TODO: implement this function
    raise NotImplementedError("generate_unconstrained not implemented")


def generate_constrained(
    input_ids,
    params,
    n_head,
    pattern,
    tokenizer,
    vocabulary=None,
    max_tokens=100,
    temperature=0.8,
    verbose=False,
):
    """Autoregressive generation constrained to match a regex pattern.

    Builds a token-level FSM from the pattern, then at each step either
    emits the longest coalesced token (skipping the LLM entirely on
    deterministic paths) or masks logits before sampling.

    Parameters
    ----------
    input_ids : list[int]
        Prompt token IDs.
    params : dict
        GPT-2 parameters.
    n_head : int
        Number of attention heads.
    pattern : str
        Regex pattern the output must match.
    tokenizer : object
        Must have ``.vocab_size`` and ``.decode()`` methods.
    vocabulary : list[str] or None
        Pre-decoded vocabulary.  Built from tokenizer if None.
    max_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.  0.0 means greedy.
    verbose : bool
        Print progress info.

    Returns
    -------
    (generated_ids, generated_text, n_total, n_coalesced)
        generated_ids : list[int] — token IDs produced
        generated_text : str — decoded output
        n_total : int — total tokens emitted
        n_coalesced : int — tokens emitted via coalescence (no model call)

    Algorithm
    ---------
    1. Resolve vocabulary (``V = tokenizer.vocab_size``).
    2. Build the token-level FSM using ``build_tokenizer_index``.
    3. Initialize: ``inputs = list(input_ids)``, ``state = tok_fsm.initial``,
       ``generated_ids = []``, ``n_coalesced = 0``.
    4. For each step (up to ``max_tokens``):
       a. **Stop** if ``state`` is a final state with no outgoing transitions.
       b. If ``state in coalesced_tokens``: emit the coalesced token
          directly (``next_id, state = coalesced_tokens[state]``),
          increment ``n_coalesced``.  **Skip the model forward pass.**
       c. Otherwise: run ``gpt2`` to get logits, add the precomputed
          mask (``precomputed_masks[state]``), sample or argmax,
          then advance the FSM state via ``tok_fsm.next_state``.
       d. Append ``next_id`` to both ``inputs`` and ``generated_ids``.
    5. Decode ``generated_ids`` and return the 4-tuple.
    """
    # TODO: implement this function
    raise NotImplementedError("generate_constrained not implemented")


def display_tokens(token_ids, tokenizer):
    """Display the sequence of individual tokens with their IDs."""
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    parts = [f"{repr(tok)}({tid})" for tok, tid in zip(tokens, token_ids)]
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("picoGPT + FSM Constrained Generation")
    print("=" * 70)

    # --- Load model ---
    print("\nLoading GPT-2 weights from HuggingFace (safetensors)...")
    params, n_head = load_gpt2_params()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    V = tokenizer.vocab_size
    vocabulary = [tokenizer.decode([i]) for i in range(V)]
    print(
        f"Model loaded: {len(params['blocks'])} layers, {n_head} heads, "
        f"{V:,} tokens\n"
    )

    # --- Same prompt for all demos ---
    # The prompt gives natural-language user profiles; the model must produce
    # a deeply nested JSON record.
    prompt = (
        "Convert each user profile to a JSON record.\n"
        "Profile: Alice Smith, 29, admin at 42 Main St, Paris 75001, loves python and rust, verified, score 82, email alice@gmail.com\n"
        '{"id":1,"first_name":"Alice","last_name":"Smith","email":"alice@gmail.com","age":29,"address":{"street":"42 Main St","city":"Paris","zip":"75001"},"role":"admin","verified":true,"score":82,"tags":["python","rust"]}\n'
        "Profile: Bob Johnson, 45, viewer at 7 Oak Ave, Tokyo 10001, go developer, not verified, score 15, email bob@yahoo.com\n"
        '{"id":2,"first_name":"Bob","last_name":"Johnson","email":"bob@yahoo.com","age":45,"address":{"street":"7 Oak Ave","city":"Tokyo","zip":"10001"},"role":"viewer","verified":false,"score":15,"tags":["go"]}\n'
        "Profile: Diana Garcia, 33, editor at 156 Elm Blvd, Berlin 10115, java and typescript, verified, score 97, email diana@outlook.com\n"
    )
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt (shared):\n{prompt}\n")

    # --- Demo 1: Unconstrained generation (stops at closing }) ---
    print("--- Unconstrained generation (temp=0.0, stop at closing }) ---")
    t0 = time.perf_counter()
    generated = generate_unconstrained(
        input_ids, params, n_head, max_tokens=80, temperature=0.0, tokenizer=tokenizer
    )
    unconstrained_elapsed = time.perf_counter() - t0
    unconstrained_text = tokenizer.decode(generated)
    print(f"  Output: {unconstrained_text}")
    print(f"  Tokens: {display_tokens(generated, tokenizer)}")
    print(f"  Time: {unconstrained_elapsed:.1f}s ({len(generated)} tokens)")

    # Test: does the unconstrained output match the JSON regex?
    match = re.fullmatch(JSON_VERY_COMPLEX, unconstrained_text.strip())
    if match:
        print("  Regex test: PASS (output matches JSON_VERY_COMPLEX)")
    else:
        print("  Regex test: FAIL (output does NOT match JSON_VERY_COMPLEX)")
    print()

    # --- Demo 2: Constrained generation (JSON_VERY_COMPLEX) ---
    print("--- Constrained generation (JSON_VERY_COMPLEX) ---")
    # print(f"  Pattern: {JSON_VERY_COMPLEX}")
    t0 = time.perf_counter()
    constrained_ids, text, n_total, n_coalesced = generate_constrained(
        input_ids,
        params,
        n_head,
        pattern=JSON_VERY_COMPLEX,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        temperature=0.0,
    )
    constrained_elapsed = time.perf_counter() - t0
    print(f"  Result:  {text}")
    print(f"  Tokens:  {display_tokens(constrained_ids, tokenizer)}")
    print(
        f"  Time: {constrained_elapsed:.1f}s "
        f"({n_total} tokens, {n_coalesced} coalesced, {n_total - n_coalesced} model)"
    )

    # Test: does the constrained output match the JSON regex?
    match = re.fullmatch(JSON_VERY_COMPLEX, text.strip())
    if match:
        print("  Regex test: PASS (output matches JSON_VERY_COMPLEX)")
    else:
        print("  Regex test: FAIL (output does NOT match JSON_VERY_COMPLEX)")
    print()
