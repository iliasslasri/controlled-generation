"""Unit tests for the required functions in structured_generation/.

Tests cover the 10 functions students must implement:
  Module 1 (deterministic_finite_automaton.py): naive_mask, build_dfa_index, dfa_mask
  Module 2 (fsm_token.py): make_deterministic_fsm, _walk_token_through_fsm, create_fsm_index_tokenizer
  Module 3 (coalescence.py): _walk_deterministic (via build_tokenizer_index), build_tokenizer_index
  Module 4 (picoGPT_generate.py): generate_unconstrained, generate_constrained
"""
import sys
import os
import math

import numpy as np
import pytest
import interegular

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structured_generation'))


# ============================================================================
# Module 1 — deterministic_finite_automaton.py
# ============================================================================

import deterministic_finite_automaton as dfa_mod

REGEX = dfa_mod.REGEX  # r"([0-9]+)?\.[0-9]+"


# --- naive_mask ---

class TestNaiveMask:
    def test_valid_tokens_have_zero_mask(self):
        vocab = [".", "0", "a"]
        mask = dfa_mod.naive_mask("", vocab, REGEX)
        # "." is a valid start (partial match for ".X")
        assert mask[0] == 0.0
        # "0" is a valid start (partial match for digit prefix)
        assert mask[1] == 0.0
        # "a" cannot start a match
        assert mask[2] == -math.inf

    def test_after_dot_digits_valid(self):
        vocab = ["0", "1", "a", "."]
        mask = dfa_mod.naive_mask(".", vocab, REGEX)
        # after ".", digits are valid
        assert mask[0] == 0.0  # "0"
        assert mask[1] == 0.0  # "1"
        # "a" is never valid
        assert mask[2] == -math.inf

    def test_returns_numpy_array(self):
        vocab = [".", "0"]
        mask = dfa_mod.naive_mask("", vocab, REGEX)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (2,)

    def test_all_invalid_vocabulary(self):
        """A vocabulary with no valid tokens should produce all -inf."""
        vocab = ["x", "y", "hello"]
        mask = dfa_mod.naive_mask("", vocab, REGEX)
        assert np.all(mask == -math.inf)

    def test_multichar_token_partial_match(self):
        """Multi-character token '.2' is a valid partial match from empty."""
        vocab = [".2", "ab"]
        mask = dfa_mod.naive_mask("", vocab, REGEX)
        assert mask[0] == 0.0   # ".2" is valid prefix
        assert mask[1] == -math.inf  # "ab" is not


# --- build_dfa_index ---

class TestBuildDfaIndex:
    def test_returns_three_elements(self):
        vocab = [".", "0", "1", ".2"]
        result = dfa_mod.build_dfa_index(vocab, REGEX)
        assert len(result) == 3

    def test_fsm_has_initial_state(self):
        vocab = [".", "0", "1"]
        fsm, s2v, sts = dfa_mod.build_dfa_index(vocab, REGEX)
        assert hasattr(fsm, 'initial')

    def test_states_to_vocab_contains_valid_tokens(self):
        vocab = [".", "0", "1", "a"]
        fsm, s2v, sts = dfa_mod.build_dfa_index(vocab, REGEX)
        # The initial state should have some valid tokens
        initial_valid = s2v[fsm.initial]
        assert len(initial_valid) > 0
        # "a" (index 3) should not be valid from any state
        all_valid = set()
        for valid_set in s2v.values():
            all_valid.update(valid_set)
        assert 3 not in all_valid  # "a" is never valid

    def test_state_transitions_are_consistent(self):
        vocab = [".", "0", "1"]
        fsm, s2v, sts = dfa_mod.build_dfa_index(vocab, REGEX)
        # Every token in s2v[state] should also have a transition in sts[state]
        for state in s2v:
            for token_id in s2v[state]:
                assert token_id in sts[state]

    def test_multichar_token_transitions(self):
        """Multi-character token '.2' should be valid from the initial state
        and land in the same state as walking '.' then '2'."""
        vocab = [".", "0", "2", ".2"]
        fsm, s2v, sts = dfa_mod.build_dfa_index(vocab, REGEX)
        # ".2" (index 3) should be valid from initial
        assert 3 in s2v[fsm.initial]
        # Walking "." then "2" should land in the same state as ".2"
        state_after_dot = sts[fsm.initial][0]  # "." is index 0
        state_after_dot2 = sts[state_after_dot][2]  # "2" is index 2
        state_dot2 = sts[fsm.initial][3]  # ".2" is index 3
        assert state_after_dot2 == state_dot2


# --- dfa_mask ---

class TestDfaMask:
    def test_valid_tokens_get_zero(self):
        s2v = {0: {1, 3}}
        mask = dfa_mod.dfa_mask(0, s2v, 5)
        assert mask[1] == 0.0
        assert mask[3] == 0.0

    def test_invalid_tokens_get_neginf(self):
        s2v = {0: {1}}
        mask = dfa_mod.dfa_mask(0, s2v, 4)
        assert mask[0] == -np.inf
        assert mask[2] == -np.inf
        assert mask[3] == -np.inf

    def test_empty_state_all_neginf(self):
        s2v = {0: set()}
        mask = dfa_mod.dfa_mask(0, s2v, 3)
        assert np.all(mask == -np.inf)

    def test_unknown_state_raises_key_error(self):
        s2v = {0: {1}}
        with pytest.raises(KeyError):
            dfa_mod.dfa_mask(99, s2v, 3)

    def test_returns_correct_shape(self):
        s2v = {0: {0}}
        mask = dfa_mod.dfa_mask(0, s2v, 10)
        assert mask.shape == (10,)


# --- Integration: naive vs DFA produce consistent results ---

class TestNaiveVsDfa:
    def test_both_approaches_allow_same_tokens_at_start(self):
        vocab = dfa_mod.build_vocabulary(50)
        naive = dfa_mod.naive_mask("", vocab, REGEX)
        fsm, s2v, sts = dfa_mod.build_dfa_index(vocab, REGEX)
        dfa = dfa_mod.dfa_mask(fsm.initial, s2v, len(vocab))
        # Both masks should agree on which tokens are valid (0.0) vs blocked (-inf)
        naive_valid = set(np.where(naive == 0.0)[0])
        dfa_valid = set(np.where(dfa == 0.0)[0])
        assert naive_valid == dfa_valid


# ============================================================================
# Module 2 — fsm_token.py
# ============================================================================

from fsm_token import (
    make_deterministic_fsm,
    TokenFSM,
    _walk_token_through_fsm,
    create_fsm_index_tokenizer,
)


def _make_raw_fsm(pattern=REGEX):
    return interegular.parse_pattern(pattern).to_fsm()


# --- make_deterministic_fsm ---

class TestMakeDeterministicFsm:
    def test_initial_state_is_zero(self):
        raw = _make_raw_fsm()
        clean, mapping = make_deterministic_fsm(raw)
        assert clean.initial == 0

    def test_states_are_contiguous_integers(self):
        raw = _make_raw_fsm()
        clean, mapping = make_deterministic_fsm(raw)
        assert all(isinstance(s, int) for s in clean.states)
        assert clean.states == set(range(len(clean.states)))

    def test_mapping_covers_all_original_states(self):
        raw = _make_raw_fsm()
        clean, mapping = make_deterministic_fsm(raw)
        for state in raw.states:
            if state is not None:  # oblivion/None may not be in map
                assert state in mapping or state not in raw.map

    def test_finals_are_subset_of_states(self):
        raw = _make_raw_fsm()
        clean, mapping = make_deterministic_fsm(raw)
        assert clean.finals.issubset(clean.states)

    def test_transitions_point_to_valid_states(self):
        raw = _make_raw_fsm()
        clean, mapping = make_deterministic_fsm(raw)
        for state, transitions in clean.map.items():
            assert state in clean.states
            for sym, target in transitions.items():
                assert target in clean.states

    def test_simple_pattern(self):
        """A simple pattern like 'ab' should produce a small clean FSM."""
        raw = interegular.parse_pattern("ab").to_fsm()
        clean, mapping = make_deterministic_fsm(raw)
        assert clean.initial == 0
        assert len(clean.finals) >= 1


# --- _walk_token_through_fsm ---

class TestWalkTokenThroughFsm:
    def _get_clean_fsm(self, pattern=REGEX):
        raw = _make_raw_fsm(pattern)
        clean, _ = make_deterministic_fsm(raw)
        return clean

    def test_single_digit_valid_from_initial(self):
        fsm = self._get_clean_fsm()
        # A digit like "1" should be walkable from the initial state
        result = _walk_token_through_fsm(fsm, fsm.initial, "1")
        assert result is not None

    def test_dot_valid_from_initial(self):
        fsm = self._get_clean_fsm()
        result = _walk_token_through_fsm(fsm, fsm.initial, ".")
        assert result is not None

    def test_letter_rejected_from_initial(self):
        fsm = self._get_clean_fsm()
        result = _walk_token_through_fsm(fsm, fsm.initial, "a")
        assert result is None

    def test_multichar_token(self):
        fsm = self._get_clean_fsm()
        # ".2" should be valid from initial state (matches the pattern start)
        result = _walk_token_through_fsm(fsm, fsm.initial, ".2")
        assert result is not None

    def test_returns_correct_landing_state(self):
        fsm = self._get_clean_fsm()
        # Walk "." then "2" character by character should give same result as ".2"
        state_after_dot = _walk_token_through_fsm(fsm, fsm.initial, ".")
        state_after_dot2 = _walk_token_through_fsm(fsm, state_after_dot, "2")
        state_dot2 = _walk_token_through_fsm(fsm, fsm.initial, ".2")
        assert state_after_dot2 == state_dot2

    def test_empty_token_returns_same_state(self):
        fsm = self._get_clean_fsm()
        result = _walk_token_through_fsm(fsm, fsm.initial, "")
        assert result == fsm.initial


# --- create_fsm_index_tokenizer ---

class TestCreateFsmIndexTokenizer:
    def test_returns_token_fsm_and_index(self):
        raw = _make_raw_fsm()
        clean, _ = make_deterministic_fsm(raw)
        vocab = ["a", ".", ".2", "1"]
        token_fsm, index = create_fsm_index_tokenizer(clean, vocab)
        assert isinstance(token_fsm, TokenFSM)
        assert isinstance(index, dict)

    def test_initial_state_matches(self):
        raw = _make_raw_fsm()
        clean, _ = make_deterministic_fsm(raw)
        vocab = [".", "0", "1"]
        token_fsm, _ = create_fsm_index_tokenizer(clean, vocab)
        assert token_fsm.initial == clean.initial

    def test_finals_match(self):
        raw = _make_raw_fsm()
        clean, _ = make_deterministic_fsm(raw)
        vocab = [".", "0"]
        token_fsm, _ = create_fsm_index_tokenizer(clean, vocab)
        assert token_fsm.finals == set(clean.finals)

    def test_invalid_token_not_in_index(self):
        raw = _make_raw_fsm()
        clean, _ = make_deterministic_fsm(raw)
        vocab = ["a", ".", "0"]
        token_fsm, index = create_fsm_index_tokenizer(clean, vocab)
        # "a" (token_id=0) should not appear in any state's transitions
        for state_transitions in index.values():
            assert 0 not in state_transitions

    def test_constrained_generation_produces_valid_output(self):
        """End-to-end: generate with the token FSM and verify the result."""
        import regex as re
        raw = _make_raw_fsm()
        clean, _ = make_deterministic_fsm(raw)
        vocab = [".", "0", "1", "2", ".2", "10"]
        token_fsm, _ = create_fsm_index_tokenizer(clean, vocab)

        np.random.seed(42)
        state = token_fsm.initial
        completion = ""
        for _ in range(5):
            allowed = token_fsm.allowed_token_ids(state)
            if not allowed:
                break
            next_id = np.random.choice(list(allowed))
            state = token_fsm.next_state(state, next_id)
            completion += vocab[next_id]

        # The completion should at least partially match the regex
        assert re.fullmatch(REGEX, completion, partial=True) is not None


# ============================================================================
# Module 3 — coalescence.py
# ============================================================================

from coalescence import (
    _walk_deterministic,
    build_tokenizer_index,
    JSON_SIMPLE,
)


# --- _walk_deterministic ---

class TestWalkDeterministic:
    def _get_clean_fsm(self, pattern):
        raw = interegular.parse_pattern(pattern).to_fsm()
        clean, _ = make_deterministic_fsm(raw)
        return clean

    def test_literal_string_is_deterministic(self):
        """Pattern 'abc' has exactly one valid char at each step."""
        fsm = self._get_clean_fsm(r"abc")
        assert _walk_deterministic(fsm, fsm.initial, 3) is True

    def test_alternation_is_not_deterministic(self):
        """Pattern '(a|b)c' — first step has two valid chars, not deterministic."""
        fsm = self._get_clean_fsm(r"(a|b)c")
        assert _walk_deterministic(fsm, fsm.initial, 1) is False

    def test_zero_chars_is_always_deterministic(self):
        """Checking 0 characters should always return True."""
        fsm = self._get_clean_fsm(r"(a|b)")
        assert _walk_deterministic(fsm, fsm.initial, 0) is True

    def test_digit_range_is_not_deterministic(self):
        """Pattern '[0-9]' — 10 valid chars at the first step."""
        fsm = self._get_clean_fsm(r"[0-9]")
        assert _walk_deterministic(fsm, fsm.initial, 1) is False

    def test_partial_deterministic_path(self):
        """Pattern 'a[0-9]' — first step is deterministic (only 'a'),
        but two steps is not (second step has 10 valid digits)."""
        fsm = self._get_clean_fsm(r"a[0-9]")
        assert _walk_deterministic(fsm, fsm.initial, 1) is True
        assert _walk_deterministic(fsm, fsm.initial, 2) is False


# --- build_tokenizer_index ---

class TestBuildTokenizerIndex:
    @pytest.fixture
    def simple_vocab(self):
        return ["{", "}", '"', ":", ",", "n", "a", "m", "e",
                "J", "o", "h", "1", "2", "0", "3",
                "P", "u", "l", "g", "t", "r", "f", "s"]

    def test_returns_four_elements(self, simple_vocab):
        result = build_tokenizer_index(
            JSON_SIMPLE, simple_vocab, verbose=False
        )
        assert len(result) == 4

    def test_returns_token_fsm(self, simple_vocab):
        tok_fsm, masks, coalesced, build_ms = build_tokenizer_index(
            JSON_SIMPLE, simple_vocab, verbose=False
        )
        assert isinstance(tok_fsm, TokenFSM)
        assert isinstance(tok_fsm.initial, int)
        assert isinstance(tok_fsm.finals, set)

    def test_precomputed_masks_are_numpy_arrays(self, simple_vocab):
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            JSON_SIMPLE, simple_vocab, verbose=False
        )
        for state, mask in masks.items():
            assert isinstance(mask, np.ndarray)
            assert mask.shape == (len(simple_vocab),)

    def test_precomputed_masks_are_read_only(self, simple_vocab):
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            JSON_SIMPLE, simple_vocab, verbose=False
        )
        for mask in masks.values():
            assert not mask.flags.writeable

    def test_coalesced_tokens_are_valid_transitions(self, simple_vocab):
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            JSON_SIMPLE, simple_vocab, verbose=False
        )
        for state, (token_id, landing) in coalesced.items():
            allowed = tok_fsm.allowed_token_ids(state)
            assert token_id in allowed
            assert tok_fsm.next_state(state, token_id) == landing

    def test_coalesced_tokens_pick_longest(self):
        """When multiple tokens cover a deterministic path, the longest wins."""
        # Pattern forces "ab" — vocab has "a", "b", and "ab"
        vocab = ["a", "b", "ab"]
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            r"ab", vocab, verbose=False
        )
        # The initial state should pick "ab" (index 2) over "a" (index 0)
        assert tok_fsm.initial in coalesced
        token_id, landing = coalesced[tok_fsm.initial]
        assert vocab[token_id] == "ab"

    def test_build_time_is_positive(self, simple_vocab):
        _, _, _, build_ms = build_tokenizer_index(
            JSON_SIMPLE, simple_vocab, verbose=False
        )
        assert build_ms > 0

    def test_no_coalescence_on_nondeterministic_path(self):
        """States with multiple valid characters should NOT be coalesced."""
        # Pattern "(a|b)c" — initial state has two choices (a or b),
        # so the initial state must not be coalesced.
        vocab = ["a", "b", "c"]
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            r"(a|b)c", vocab, verbose=False
        )
        assert tok_fsm.initial not in coalesced

    def test_generation_stays_on_valid_transitions(self):
        """Generate tokens using the FSM and verify all transitions are valid."""
        vocab = ["{", "}", '"', ":", ",", "n", "a", "m", "e",
                 "J", "o", "h", "1", "2", "0", "3"]
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            JSON_SIMPLE, vocab, verbose=False
        )
        np.random.seed(0)
        state = tok_fsm.initial
        for step in range(20):
            allowed = tok_fsm.allowed_token_ids(state)
            if not allowed:
                break
            if state in coalesced:
                next_id, _ = coalesced[state]
            else:
                from scipy.special import softmax as sp_softmax
                logits = np.ones(len(vocab)) + masks[state]
                probs = sp_softmax(logits)
                next_id = np.random.choice(len(vocab), p=probs)
            assert next_id in allowed
            state = tok_fsm.next_state(state, next_id)


# ============================================================================
# Module 4 — picoGPT_generate.py
# ============================================================================

# Purge any cached 'utils' from other test modules (e.g. meta_generation/utils)
# so that picoGPT_generate's `from utils import ...` resolves to structured_generation/utils.
sys.modules.pop("utils", None)
from picoGPT_generate import (
    generate_unconstrained,
    generate_constrained,
)


# --- generate_unconstrained ---

class TestGenerateUnconstrained:
    def _make_tiny_params(self, n_vocab=10, n_seq=6, d_model=8, n_head=2):
        np.random.seed(42)
        params = {
            "wte": np.random.randn(n_vocab, d_model).astype(np.float32) * 0.01,
            "wpe": np.random.randn(n_seq, d_model).astype(np.float32) * 0.01,
            "blocks": [{
                "ln_1": {"g": np.ones(d_model, dtype=np.float32),
                         "b": np.zeros(d_model, dtype=np.float32)},
                "ln_2": {"g": np.ones(d_model, dtype=np.float32),
                         "b": np.zeros(d_model, dtype=np.float32)},
                "attn": {
                    "c_attn": {"w": np.random.randn(d_model, 3 * d_model).astype(np.float32) * 0.01,
                               "b": np.zeros(3 * d_model, dtype=np.float32)},
                    "c_proj": {"w": np.random.randn(d_model, d_model).astype(np.float32) * 0.01,
                               "b": np.zeros(d_model, dtype=np.float32)},
                },
                "mlp": {
                    "c_fc": {"w": np.random.randn(d_model, 4 * d_model).astype(np.float32) * 0.01,
                             "b": np.zeros(4 * d_model, dtype=np.float32)},
                    "c_proj": {"w": np.random.randn(4 * d_model, d_model).astype(np.float32) * 0.01,
                               "b": np.zeros(d_model, dtype=np.float32)},
                },
            }],
            "ln_f": {"g": np.ones(d_model, dtype=np.float32),
                     "b": np.zeros(d_model, dtype=np.float32)},
        }
        return params, n_head

    def test_generates_tokens(self):
        n_vocab = 10
        params, n_head = self._make_tiny_params(n_vocab=n_vocab)
        input_ids = [0, 1]
        generated = generate_unconstrained(input_ids, params, n_head, max_tokens=3)
        assert len(generated) > 0
        assert len(generated) <= 3
        assert all(0 <= t < n_vocab for t in generated)

    def test_greedy_is_deterministic(self):
        params, n_head = self._make_tiny_params()
        input_ids = [0, 1]
        gen1 = generate_unconstrained(input_ids, params, n_head, max_tokens=3, temperature=0.0)
        gen2 = generate_unconstrained(input_ids, params, n_head, max_tokens=3, temperature=0.0)
        assert gen1 == gen2

    def test_brace_depth_stopping(self):
        """When tokenizer is provided, generation should stop after closing }."""

        class _BraceTokenizer:
            """Tokenizer where token 0 = '{', token 1 = '}', rest = letters."""
            def decode(self, ids):
                mapping = {0: "{", 1: "}", 2: "a", 3: "b"}
                if isinstance(ids, list) and len(ids) == 1:
                    return mapping.get(ids[0], "x")
                return "".join(mapping.get(i, "x") for i in ids)

        n_vocab = 4
        params, n_head = self._make_tiny_params(n_vocab=n_vocab)
        tokenizer = _BraceTokenizer()

        # Force the model to emit: { a } by manipulating weights
        # We just verify that when a tokenizer is passed, max_tokens is respected
        # and the output is a list of valid token ids
        generated = generate_unconstrained(
            [0, 1], params, n_head, max_tokens=10,
            temperature=0.0, tokenizer=tokenizer,
        )
        assert isinstance(generated, list)
        assert all(0 <= t < n_vocab for t in generated)
        # With tokenizer, should stop at or before max_tokens
        assert len(generated) <= 10

    def test_does_not_modify_input_ids(self):
        """Input list should not be mutated."""
        params, n_head = self._make_tiny_params()
        input_ids = [0, 1]
        original = list(input_ids)
        generate_unconstrained(input_ids, params, n_head, max_tokens=2, temperature=0.0)
        assert input_ids == original


# --- generate_constrained ---

class _FakeTokenizer:
    """Minimal tokenizer stub backed by a fixed vocabulary list."""
    def __init__(self, vocab):
        self._vocab = list(vocab)
        self.vocab_size = len(self._vocab)

    def decode(self, ids):
        if isinstance(ids, list) and len(ids) == 1:
            return self._vocab[ids[0]]
        return "".join(self._vocab[i] for i in ids)

    def encode(self, text):
        # character-level encoding (sufficient for tests)
        return [self._vocab.index(ch) for ch in text]


def _make_tiny_params(n_vocab, n_seq=8, d_model=8, n_head=2):
    """Build minimal GPT-2 params for testing."""
    np.random.seed(42)
    params = {
        "wte": np.random.randn(n_vocab, d_model).astype(np.float32) * 0.01,
        "wpe": np.random.randn(n_seq, d_model).astype(np.float32) * 0.01,
        "blocks": [{
            "ln_1": {"g": np.ones(d_model, dtype=np.float32),
                     "b": np.zeros(d_model, dtype=np.float32)},
            "ln_2": {"g": np.ones(d_model, dtype=np.float32),
                     "b": np.zeros(d_model, dtype=np.float32)},
            "attn": {
                "c_attn": {"w": np.random.randn(d_model, 3 * d_model).astype(np.float32) * 0.01,
                           "b": np.zeros(3 * d_model, dtype=np.float32)},
                "c_proj": {"w": np.random.randn(d_model, d_model).astype(np.float32) * 0.01,
                           "b": np.zeros(d_model, dtype=np.float32)},
            },
            "mlp": {
                "c_fc": {"w": np.random.randn(d_model, 4 * d_model).astype(np.float32) * 0.01,
                         "b": np.zeros(4 * d_model, dtype=np.float32)},
                "c_proj": {"w": np.random.randn(4 * d_model, d_model).astype(np.float32) * 0.01,
                           "b": np.zeros(d_model, dtype=np.float32)},
            },
        }],
        "ln_f": {"g": np.ones(d_model, dtype=np.float32),
                 "b": np.zeros(d_model, dtype=np.float32)},
    }
    return params, n_head


class TestGenerateConstrained:
    def test_fully_deterministic_pattern(self):
        """A pattern with a single possible string should be fully coalesced."""
        vocab = ["a", "b", "c", "ab", "abc"]
        tokenizer = _FakeTokenizer(vocab)
        params, n_head = _make_tiny_params(n_vocab=len(vocab))
        # Pattern forces exactly "abc"
        ids, text, n_total, n_coalesced = generate_constrained(
            [0], params, n_head,
            pattern=r"abc", tokenizer=tokenizer,
            vocabulary=list(vocab), verbose=False,
        )
        assert text == "abc"
        # All tokens should be coalesced (no model call needed)
        assert n_coalesced == n_total

    def test_longest_token_is_used(self):
        """Should pick 'abc' (1 token) over 'a'+'b'+'c' (3 tokens)."""
        vocab = ["a", "b", "c", "abc"]
        tokenizer = _FakeTokenizer(vocab)
        params, n_head = _make_tiny_params(n_vocab=len(vocab))
        ids, text, n_total, n_coalesced = generate_constrained(
            [0], params, n_head,
            pattern=r"abc", tokenizer=tokenizer,
            vocabulary=list(vocab), verbose=False,
        )
        assert text == "abc"
        # "abc" should be emitted as a single coalesced token
        assert n_total == 1
        assert vocab[ids[0]] == "abc"

    def test_output_matches_pattern(self):
        """Output must match the regex pattern."""
        import regex as re
        vocab = ["a", "b", "1", "2", "3"]
        tokenizer = _FakeTokenizer(vocab)
        params, n_head = _make_tiny_params(n_vocab=len(vocab))
        pattern = r"(a|b)(1|2|3)"
        ids, text, n_total, n_coalesced = generate_constrained(
            [0], params, n_head,
            pattern=pattern, tokenizer=tokenizer,
            vocabulary=list(vocab), temperature=0.0, verbose=False,
        )
        assert re.fullmatch(pattern, text) is not None

    def test_returns_four_values(self):
        vocab = ["x", "y"]
        tokenizer = _FakeTokenizer(vocab)
        params, n_head = _make_tiny_params(n_vocab=len(vocab))
        result = generate_constrained(
            [0], params, n_head,
            pattern=r"(x|y)+", tokenizer=tokenizer,
            vocabulary=list(vocab), max_tokens=3, temperature=0.0, verbose=False,
        )
        assert len(result) == 4
        ids, text, n_total, n_coalesced = result
        assert isinstance(ids, list)
        assert isinstance(text, str)
        assert isinstance(n_total, int)
        assert isinstance(n_coalesced, int)
