"""Unit tests for structured_generation/deterministic_finite_automaton.py."""
import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structured_generation'))
import deterministic_finite_automaton as dfa_mod


REGEX = dfa_mod.REGEX  # r"([0-9]+)?\.[0-9]+"


# --- build_vocabulary ---

class TestBuildVocabulary:
    def test_returns_correct_size(self):
        vocab = dfa_mod.build_vocabulary(50)
        assert len(vocab) == 50

    def test_contains_core_tokens(self):
        vocab = dfa_mod.build_vocabulary(100)
        for tok in [".", "0", "1", "9", ".0", ".9", "42"]:
            assert tok in vocab

    def test_padding_tokens_fill_remainder(self):
        vocab = dfa_mod.build_vocabulary(30)
        # 26 core tokens, so 4 padding tokens
        padding = [t for t in vocab if t.startswith("tok_")]
        assert len(padding) == 4

    def test_small_size_truncates_core(self):
        vocab = dfa_mod.build_vocabulary(5)
        assert len(vocab) == 5
        assert vocab[0] == "."


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
