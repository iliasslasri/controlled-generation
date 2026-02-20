# Part 1: Structured Generation [8 points]

LLMs generate text token by token, sampling from the full vocabulary at each step. But many applications require the output to follow a strict format: valid JSON, a specific schema, a phone number, etc. **Structured generation** guarantees that every generated token is part of a valid output by constraining the sampling at each step.

This part introduces a progression of techniques, from a naive baseline to a fully optimized constrained decoding pipeline. Each module builds on the previous one.

## Overview

| Module | Description |
|---|---|
| `deterministic_finite_automaton.py` | Naive regex masking vs. DFA-based masking |
| `fsm_token.py` | From character-level DFA to token-level FSM |
| `coalescence.py` | Longest-token coalescence |
| `picoGPT_generate.py` | End-to-end constrained generation with GPT-2 |
| `utils.py` | GPT-2 weight loading and caching (utility, no exercises) |

---

## Module 1 — Naive vs. DFA Masking (`deterministic_finite_automaton.py`)

### Motivation

Given a regex pattern (e.g. `([0-9]+)?\.[0-9]+` for decimal numbers), how do we ensure the LLM only generates matching strings?

The simplest idea: at each generation step, try appending every token in the vocabulary to the current completion and check whether the result could still lead to a valid match. This is the **naive approach**.

### Algorithm: Naive Masking

For each generation step:
1. For each token `t` in the vocabulary of size `V`:
   - Form the tentative string `completion + t`
   - Run a **partial** regex match: does this prefix still have a chance of completing to a full match?
   - If not, mask the token (set its logit to `-inf`)
2. Sample from the masked distribution

This works but is **O(V)** regex evaluations per step, which is expensive for real vocabularies (GPT-2 has 50,257 tokens).

### Algorithm: DFA-Based Masking

A regex can be compiled into a **Deterministic Finite Automaton (DFA)**. A DFA has a finite set of states and deterministic transitions on each input character. Instead of re-running the regex from scratch at every step, we can:

1. **Precomputation (one-time):** Parse the regex into a DFA. For each DFA state and each token in the vocabulary, walk the token's characters through the DFA and record where it lands. This produces a mapping: `(state, token_id) -> next_state`.
2. **Generation (per step):** Look up the current DFA state, retrieve the set of valid tokens, build the mask. This is an **O(1)** lookup.

The precomputation is `O(|states| x V x avg_token_length)` but is done only once. After that, each generation step is essentially free.

### What to implement

- `naive_mask(completion, vocabulary, pattern)`: build a logit mask by trying every token with partial regex matching.
- `build_dfa_index(vocabulary, pattern)`: precompute the DFA transition table indexed by state and token.
- `dfa_mask(state, states_to_vocab, vocab_size)`: build a logit mask from the precomputed index in O(1).

### Running the benchmark

```bash
uv run python structured_generation/deterministic_finite_automaton.py
```

This benchmarks both approaches across vocabulary sizes from 100 to 10,000 and prints a comparison table showing the speedup of DFA masking over naive masking.

---

## Module 2 — Character-level DFA to Token-level FSM (`fsm_token.py`)

### Motivation

Module 1 builds a DFA index, but the DFA itself operates on **characters**, while LLMs generate **tokens** (which can be multi-character strings like `"ing"`, `".2"`, or `"123"`). We need to bridge this gap by compiling the character-level DFA into a **token-level Finite State Machine (FSM)**.

### Algorithm

The pipeline has three stages:

**Stage 1 — Parse the regex into a character-level DFA.** We use the `interegular` library which converts a regex pattern into a DFA with states, transitions, and an alphabet mapping characters to symbol indices.

**Stage 2 — Clean up the DFA (`make_deterministic_fsm`).** The raw DFA from `interegular` may contain:
- An implicit "oblivion" (dead/sink) state
- Non-contiguous state numbering
- The special `anything_else` sentinel for characters not explicitly in the alphabet

This function normalizes the DFA: collects all reachable states, remaps them to contiguous integers (0, 1, 2, ...), and builds a clean FSM object.

**Stage 3 — Build the token-level FSM (`create_fsm_index_tokenizer`).** For every (state, token) pair:
1. Walk the token's characters through the character-level DFA starting from that state (using `_walk_token_through_fsm`)
2. If the walk succeeds (no dead-end), record the transition: `state --[token_id]--> landing_state`

The result is a `TokenFSM` object whose transitions are over token IDs instead of characters.

### The `_walk_token_through_fsm` function

This is the core primitive. Given a DFA state and a multi-character token string:
- For each character, look up the symbol index in the DFA alphabet (handling the `anything_else` sentinel)
- Attempt the transition; if any character fails, return `None` (token rejected)
- If all characters succeed, return the final landing state

### What to implement

- `make_deterministic_fsm(fsm)`: normalize a raw DFA to clean integer states.
- `_walk_token_through_fsm(fsm, state, token)`: walk a multi-character token through the character-level DFA.
- `create_fsm_index_tokenizer(fsm, tokenizer)`: build the full token-level FSM by iterating over all (state, token) pairs.

### Running the demo

```bash
uv run python structured_generation/fsm_token.py
```

This demonstrates the full pipeline on the regex `([0-9]+)?\.[0-9]+` with a small vocabulary `["a", ".", ".2", "1"]`, showing the DFA states, the token-level transitions, and a constrained generation run.

---

## Module 3 — Coalescence: Longest-Token Optimization (`coalescence.py`)

### Motivation

The token-level FSM from Module 2 gives us an `O(1)` mask lookup per step, but we can go further.

Consider generating the string `"name"`. With a typical tokenizer vocabulary, there are many ways to produce it: `["name"]`, `["n", "ame"]`, `["na", "me"]`, `["n", "a", "m", "e"]`, etc. When the character-level DFA has a **deterministic path** forward (only one valid character transition at each step), all these tokenizations follow the same forced sequence. Instead of stepping through short tokens one at a time (each requiring an LLM call), we pick the **longest token** that covers as much of the deterministic path as possible. This skips the LLM forward pass entirely for those states and jumps as far ahead as possible in a single step.

### Algorithm: `build_tokenizer_index`

1. Parse the regex and build the character-level DFA (reusing Module 2)
2. Build the token-level FSM index (same as Module 2)
3. **Precompute logit masks:** for each state, build a NumPy array of size `V` with `-inf` at disallowed positions and `0` at allowed positions.
4. **Longest-token coalescence:** for each state, check whether the character-level DFA has a deterministic path forward (using `_walk_deterministic`). If so, among all valid tokens from that state, pick the longest one that stays on the deterministic path and record it in a `coalesced_tokens` dictionary along with its landing state.

Returns:
- `TokenFSM`: the token-level FSM
- `precomputed_masks`: `dict[state] -> np.array` (one mask per state)
- `coalesced_tokens`: `dict[state] -> (token_id, landing_state)`
- `build_ms`: build time in milliseconds

### What to implement

- `_walk_deterministic(fsm, state, n_chars)`: check that the character-level DFA is deterministic for `n_chars` steps from a given state.
- `build_tokenizer_index(pattern, tokenizer, vocabulary)`: the full pipeline including mask precomputation and longest-token coalescence.
- The generation loop that emits the longest coalesced token on deterministic paths, skipping the LLM entirely.

### Running the benchmark

```bash
uv run python structured_generation/coalescence.py
```

This benchmarks on the real GPT-2 vocabulary (50,257 tokens) with three JSON regex patterns of increasing complexity. It compares the per-step time of FSM masking (without coalescence) vs. longest-token coalescence.

### JSON regex patterns

The module defines three JSON patterns used for benchmarking:

- **`JSON_SIMPLE`**: a JSON object with two fixed fields (`name`, `age`) and a few allowed values
- **`JSON_COMPLEX`**: five fields with richer value ranges (names, ages 18-65, cities, booleans, scores)
- **`JSON_VERY_COMPLEX`**: deeply nested JSON with addresses, email validation, arrays, and many allowed values

---

## Module 4 — End-to-End Constrained Generation (`picoGPT_generate.py`)

### Motivation

Modules 1-3 build the masking infrastructure. This module puts it all together with an actual language model: a pure-NumPy implementation of GPT-2 (based on [picoGPT](https://github.com/jaymody/picoGPT) by Jay Mody).

### picoGPT: GPT-2 in NumPy

The module implements the full GPT-2 forward pass in pure NumPy:
- `gelu`, `softmax`, `layer_norm`, `linear`: basic building blocks
- `ffn`: feed-forward network (two linear layers with GELU)
- `attention`: scaled dot-product attention with causal mask
- `mha`: multi-head attention (split heads, attend, concatenate)
- `transformer_block`: layer norm + multi-head attention + layer norm + FFN (with residual connections)
- `gpt2`: token embeddings + position embeddings + transformer blocks + final layer norm + output projection


### Generation modes

**Unconstrained generation (`generate_unconstrained`):**
- Standard autoregressive sampling with optional temperature
- Optional brace-depth tracking to stop after the closing `}` in JSON output
- No guarantee that the output is valid

**Constrained generation (`generate_constrained`):**
1. Build the token-level FSM from the regex pattern (using `build_tokenizer_index` from Module 3)
2. At each step:
   - If the current FSM state has a coalesced token: emit the longest token directly, **skip the forward pass**, and jump to the landing state
   - Otherwise: run the GPT-2 forward pass, apply the precomputed mask, sample
3. Advance the FSM state
4. Stop when the FSM reaches a final state with no outgoing transitions

The output is **guaranteed** to match the regex pattern.

### What to implement

- `generate_unconstrained(...)`: autoregressive generation loop with optional brace-depth stopping.
- `generate_constrained(...)`: generation loop with FSM masking and longest-token coalescence.

### Running the demo

```bash
uv run python structured_generation/picoGPT_generate.py
```

This loads GPT-2, encodes a prompt asking to convert user profiles to JSON, and runs:
1. Unconstrained generation (likely produces invalid JSON)
2. Constrained generation with `JSON_VERY_COMPLEX` (guaranteed valid output)

Both outputs are tested against the regex to verify correctness.

---

## Summary: The Full Pipeline

```
Regex pattern
    |
    v
Character-level DFA          (interegular)
    |
    v
Clean DFA                    (make_deterministic_fsm)
    |
    v
Token-level FSM              (create_fsm_index_tokenizer)
    |
    v
Precomputed masks            (build_tokenizer_index)
+ longest-token coalescence
    |
    v
Constrained generation       (generate_constrained)
with GPT-2
```

Each module tackles one level of this pipeline. By the end, you will have built a system that guarantees LLM outputs match an arbitrary regex pattern, with significant performance optimizations over the naive approach.
