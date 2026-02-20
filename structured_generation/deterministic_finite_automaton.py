import math
import time
from collections import defaultdict

import numpy as np
import regex as re
import interegular
from scipy.special import softmax

REGEX = r"([0-9]+)?\.[0-9]+"


def build_vocabulary(size: int):
    """Build a synthetic vocabulary: a small useful core + junk padding tokens."""
    core = [
        ".",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ".0",
        ".1",
        ".2",
        ".3",
        ".4",
        ".5",
        ".6",
        ".7",
        ".8",
        ".9",
        "10",
        "12",
        "42",
        "100",
        "256",
    ]
    padding = [f"tok_{i}" for i in range(size - len(core))]
    vocab = core + padding
    return vocab[:size]


def naive_mask(completion, vocabulary, pattern):
    """Approach 1: one regex partial match per token per step — O(V) per token.

    For each token in the vocabulary, form the tentative string
    `completion + token` and check whether it can still lead to a full
    match of `pattern` using `re.fullmatch`.

    Parameters
    ----------
    completion : str
        The string generated so far.
    vocabulary : list[str]
        The list of tokens (strings).
    pattern : str
        The regex pattern.

    Returns
    -------
    np.ndarray of shape (len(vocabulary),)
        0.0 for tokens whose tentative string partially matches,
        -math.inf for tokens that cannot lead to a valid match.
    """
    # TODO: implement this function
    raise NotImplementedError("naive_mask not implemented")


def build_dfa_index(vocabulary, pattern):
    """Approach 2: one-time precomputation — build the DFA and index valid tokens per state.

    1. Parse the regex `pattern` into a DFA using `interegular`.
    2. For every (state, token) pair, walk the token's characters through
       the DFA.  Use `fsm.alphabet` to map each character to a symbol
       index (fall back to `interegular.fsm.anything_else` if the
       character is not explicitly in the alphabet).  Follow `fsm.map`
       for each transition.
    3. If the walk succeeds (every character has a valid transition),
       record:
         - `states_to_vocab[state].add(token_id)`
         - `states_token_states[state][token_id] = landing_state`

    Parameters
    ----------
    vocabulary : list[str]
        The list of tokens.
    pattern : str
        The regex pattern.

    Returns
    -------
    fsm : the interegular FSM object
    states_to_vocab : defaultdict(set)
        Mapping from DFA state to the set of valid token indices.
    states_token_states : defaultdict(dict)
        Mapping from DFA state to {token_id: landing_state}.
    """
    # TODO: implement this function
    raise NotImplementedError("build_dfa_index not implemented")


def dfa_mask(state, states_to_vocab, vocab_size):
    """O(1) mask construction using the precomputed index.

    Build a NumPy array of shape (vocab_size,) filled with -inf,
    then set positions corresponding to valid token indices
    (from `states_to_vocab[state]`) to 0.0.

    Parameters
    ----------
    state : int
        The current DFA state.
    states_to_vocab : dict
        Mapping from state to set of valid token indices
        (as returned by `build_dfa_index`).
    vocab_size : int
        Total vocabulary size.

    Returns
    -------
    np.ndarray of shape (vocab_size,)

    Raises
    ------
    KeyError
        If `state` is not in `states_to_vocab`.
    """
    # TODO: implement this function
    raise NotImplementedError("dfa_mask not implemented")


def benchmark(vocab_sizes, n_steps=7, n_repeats=3):
    results = []

    print(f"Regex: {REGEX}")
    print(f"Generating {n_steps} tokens per run, median of {n_repeats} repeats\n")
    print(
        f"{'Vocab size':>12}  {'Naive (ms/step)':>16}  {'DFA mask (ms/step)':>18}  {'DFA precomp (ms)':>16}  {'Speedup':>8}"
    )
    print("-" * 80)

    for V in vocab_sizes:
        vocabulary = build_vocabulary(V)
        logits = np.ones(V)

        # ---- Naive timing ----
        naive_times = []
        for _ in range(n_repeats):
            np.random.seed(42)
            completion = ""
            t0 = time.perf_counter()
            for _ in range(n_steps):
                mask = naive_mask(completion, vocabulary, REGEX)
                masked_logits = logits + mask
                probs = softmax(masked_logits)
                next_id = np.random.choice(V, p=probs)
                completion += vocabulary[next_id]
            naive_times.append((time.perf_counter() - t0) / n_steps)
        naive_ms = np.median(naive_times) * 1000

        # ---- DFA precomputation ----
        t0 = time.perf_counter()
        fsm_bench, s2v, sts = build_dfa_index(vocabulary, REGEX)
        precomp_ms = (time.perf_counter() - t0) * 1000

        # ---- DFA mask timing ----
        dfa_times = []
        for _ in range(n_repeats):
            np.random.seed(42)
            state = fsm_bench.initial
            completion = ""
            t0 = time.perf_counter()
            for _ in range(n_steps):
                mask = dfa_mask(state, s2v, V)
                masked_logits = logits + mask
                probs = softmax(masked_logits)
                next_id = np.random.choice(V, p=probs)
                if state not in sts or next_id not in sts[state]:
                    break  # no valid transition from this state
                state = sts[state][next_id]
                completion += vocabulary[next_id]
            dfa_times.append((time.perf_counter() - t0) / n_steps)
        dfa_ms = np.median(dfa_times) * 1000

        speedup = naive_ms / dfa_ms if dfa_ms > 0 else float("inf")
        results.append((V, naive_ms, dfa_ms, precomp_ms, speedup))
        print(
            f"{V:>12,}  {naive_ms:>14.2f}ms  {dfa_ms:>16.3f}ms  {precomp_ms:>14.1f}ms  {speedup:>7.0f}x"
        )

    return results


if __name__ == "__main__":
    results = benchmark([100, 500, 1_000, 5_000, 10_000])
