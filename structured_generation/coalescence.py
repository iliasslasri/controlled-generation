import time
from collections import defaultdict

import numpy as np
import interegular
from scipy.special import softmax
from transformers import AutoTokenizer

from fsm_token import make_deterministic_fsm, TokenFSM, _walk_token_through_fsm


JSON_SIMPLE = r'\{"name":("John"|"Paul"),"age":(20|30)\}'

JSON_COMPLEX = (
    r'\{'
    r'"name":("Alice"|"Bob"|"Charlie"|"Diana"|"Eve"),'
    r'"age":(1[89]|[2-5][0-9]|6[0-5]),'
    r'"city":("Paris"|"London"|"New York"|"Tokyo"|"Berlin"),'
    r'"active":(true|false),'
    r'"score":(100|[1-9][0-9]?)'
    r'\}'
)

JSON_VERY_COMPLEX = (
    r'\{'
    r'"id":([1-9][0-9]{0,4}),'
    r'"first_name":("Alice"|"Bob"|"Charlie"|"Diana"|"Eve"|"Frank"|"Grace"|"Hector"),'
    r'"last_name":("Smith"|"Johnson"|"Williams"|"Brown"|"Jones"|"Garcia"|"Miller"),'
    r'"email":"[a-z]{3,8}@(gmail|yahoo|outlook)\.com",'
    r'"age":(1[89]|[2-9][0-9]|100),'
    r'"address":\{'
        r'"street":"[0-9]{1,4} (Main|Oak|Elm|Park|Cedar|Maple) (St|Ave|Blvd|Dr)",'
        r'"city":("Paris"|"London"|"New York"|"Tokyo"|"Berlin"|"Sydney"|"Toronto"|"Seoul"),'
        r'"zip":"[0-9]{5}"'
    r'\},'
    r'"role":("admin"|"editor"|"viewer"|"moderator"),'
    r'"verified":(true|false),'
    r'"score":(0|[1-9][0-9]?|100),'
    r'"tags":\["(python|rust|java|go|typescript)"(,"(python|rust|java|go|typescript)"){0,2}\]'
    r'\}'
)


def _walk_deterministic(fsm, state, n_chars):
    """Check that the character-level DFA is deterministic for n_chars steps.

    Returns True if, starting from `state`, each of the next `n_chars`
    character-level transitions has exactly one valid character.

    Note: interegular groups equivalent characters into the same symbol
    index (e.g. [1-9] maps all 9 digits to one symbol). We must count
    actual characters, not just symbols, to avoid coalescing states
    where the model should choose among multiple valid characters.

    Parameters
    ----------
    fsm : the character-level DFA (cleaned up by make_deterministic_fsm)
    state : int, starting state
    n_chars : int, number of character-level steps to check

    Returns
    -------
    bool
        True if exactly one character is valid at each of the next
        `n_chars` steps.  Returns True trivially when `n_chars == 0`.

    Algorithm
    ---------
    1. Build a ``symbol_char_count`` dict mapping each symbol index to
       the number of distinct characters it represents.  Use
       ``fsm.alphabet`` to count characters per symbol.  The
       ``fsm_module.anything_else`` sentinel represents infinitely many
       characters (set its count to ``float("inf")``).
    2. Starting from ``state``, for each of the ``n_chars`` steps:
       a. Get the transitions from the current state: ``fsm.map.get(cur, {})``.
       b. Sum the character counts for all symbols in the transitions.
       c. If the total is not exactly 1, return ``False``.
       d. Otherwise, advance ``cur`` to the single target state.
    3. Return ``True``.
    """
    # TODO: implement this function
    raise NotImplementedError("_walk_deterministic not implemented")


def build_tokenizer_index(pattern, tokenizer, vocabulary=None, verbose=True):
    """Build a token-level FSM index for a regex pattern.

    Parameters
    ----------
    pattern : str, the regex pattern
    tokenizer : list[str] (vocabulary) or a HF tokenizer with
                .vocab_size and .decode() methods
    vocabulary : list[str] or None
        Pre-decoded vocabulary strings. If provided, avoids redundant
        tokenizer.decode() calls.

    Returns
    -------
    tok_fsm : TokenFSM
    precomputed_masks : dict[state] → np.array
        One mask per state, shape ``(V,)``, with ``-inf`` for disallowed
        tokens and ``0.0`` for allowed ones.  Must be read-only
        (``mask.flags.writeable = False``).
    coalesced_tokens : dict[state] → (token_id, landing_state)
        States on deterministic paths mapped to the longest valid token.
    build_ms : build time in milliseconds

    Steps
    -----
    1. Resolve the vocabulary and its size ``V`` (same logic as
       ``create_fsm_index_tokenizer``).
    2. Parse the regex into a character-level DFA and clean it up
       (``interegular.parse_pattern(pattern).to_fsm()`` then
       ``make_deterministic_fsm``).
    3. Build the token-level FSM index: for every ``(state, token_id)``
       pair, call ``_walk_token_through_fsm`` and record successful
       transitions in ``index[state][token_id] = landing``.  Measure
       the elapsed time as ``build_ms``.
    4. Construct a ``TokenFSM`` from the index.
    5. **Precompute logit masks**: for each state, build a NumPy array
       of shape ``(V,)`` filled with ``-inf``, set allowed positions to
       ``0.0``, then make it read-only (``mask.flags.writeable = False``).
    6. **Longest-token coalescence**: for each state that has
       transitions, iterate over all valid ``(token_id, landing)``
       pairs.  For each token, check whether the character-level DFA is
       deterministic for ``len(token_str)`` steps from that state using
       ``_walk_deterministic``.  Among all tokens that pass this check,
       keep the longest one.  Record it in
       ``coalesced_tokens[state] = (best_token_id, best_landing)``.
    7. Return ``(tok_fsm, precomputed_masks, coalesced_tokens, build_ms)``.
    """
    # TODO: implement this function
    raise NotImplementedError("build_tokenizer_index not implemented")


def _time_generation_with_skip(init_state, precomputed_masks, coalesced_tokens,
                               tok_fsm, logits, vocabulary,
                               n_steps, n_repeats, n_warmup=1):
    """Generation loop that skips softmax+sampling for coalesced states."""
    V = len(vocabulary)

    def _run_once():
        np.random.seed(42)
        state = init_state
        t0 = time.perf_counter()
        for _ in range(n_steps):
            if state in coalesced_tokens:
                next_id, state = coalesced_tokens[state]
            else:
                if not tok_fsm.map.get(state):
                    break  # no valid transitions from this state
                mask = precomputed_masks[state]
                masked_logits = logits + mask
                probs = softmax(masked_logits)
                next_id = np.random.choice(V, p=probs)
                state = tok_fsm.next_state(state, next_id)
        return (time.perf_counter() - t0) / n_steps

    for _ in range(n_warmup):
        _run_once()

    times = [_run_once() for _ in range(n_repeats)]
    times_ms = np.array(times) * 1000
    return times_ms.mean(), times_ms.std()


def benchmark_gpt2_json(patterns, n_steps=15, n_repeats=30):
    """Benchmark FSM masking vs longest-token coalescence on real GPT-2 vocab."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    V = tokenizer.vocab_size
    logits = np.ones(V)
    vocabulary = [tokenizer.decode([i]) for i in range(V)]

    print(f"Tokenizer: gpt2 ({V:,} tokens)")
    print(f"Generating {n_steps} tokens per run, mean +/- std of {n_repeats} repeats "
          f"(1 warmup)\n")

    results = {}
    for label, pattern in patterns.items():
        print(f"--- {label} ---")
        print(f"  Pattern: {pattern}")

        tok_fsm, precomputed_masks, coalesced_tokens, build_ms = \
            build_tokenizer_index(pattern, tokenizer, vocabulary=vocabulary)

        # --- Full step: FSM (precomputed mask lookup, no skip) ---
        coal_full, coal_full_std = _time_generation_with_skip(
            init_state=tok_fsm.initial,
            precomputed_masks=precomputed_masks,
            coalesced_tokens={},
            tok_fsm=tok_fsm,
            logits=logits, vocabulary=vocabulary,
            n_steps=n_steps, n_repeats=n_repeats,
        )

        # --- Full step: Longest-token coalescence ---
        skip_full, skip_full_std = _time_generation_with_skip(
            init_state=tok_fsm.initial,
            precomputed_masks=precomputed_masks,
            coalesced_tokens=coalesced_tokens,
            tok_fsm=tok_fsm,
            logits=logits, vocabulary=vocabulary,
            n_steps=n_steps, n_repeats=n_repeats,
        )

        fsm_vs_skip = coal_full / skip_full if skip_full > 0 else float("inf")

        results[label] = dict(
            build_ms=build_ms,
            fsm_full_ms=coal_full, skip_full_ms=skip_full,
            speedup=fsm_vs_skip,
        )

        print(f"  Build:                     {build_ms:>10.0f} ms")
        print(f"  Full step (ms/step):")
        print(f"    FSM (precomputed mask):       {coal_full:>8.3f} +/- {coal_full_std:.3f}")
        print(f"    Longest-token coalescence:    {skip_full:>8.3f} +/- {skip_full_std:.3f}")
        print(f"    Speedup (FSM vs coalescence): {fsm_vs_skip:>8.1f}x\n")

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("GPT-2 JSON benchmark")
    print("=" * 80 + "\n")
    results_gpt2 = benchmark_gpt2_json({
        "simple": JSON_SIMPLE,
        "complex": JSON_COMPLEX,
        "very_complex": JSON_VERY_COMPLEX,
    })
