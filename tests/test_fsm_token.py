"""Unit tests for structured_generation/fsm_token.py."""
import sys
import os

import numpy as np
import pytest
import interegular
from interegular import fsm as fsm_module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structured_generation'))
from fsm_token import (
    make_deterministic_fsm,
    TokenFSM,
    _walk_token_through_fsm,
    create_fsm_index_tokenizer,
)


REGEX = r"([0-9]+)?\.[0-9]+"


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


# --- TokenFSM ---

class TestTokenFSM:
    def test_allowed_token_ids_returns_set(self):
        fsm = TokenFSM(initial=0, finals={2}, map={0: {1: 1, 3: 2}})
        allowed = fsm.allowed_token_ids(0)
        assert allowed == {1, 3}

    def test_allowed_token_ids_empty_for_unknown_state(self):
        fsm = TokenFSM(initial=0, finals={1}, map={0: {0: 1}})
        assert fsm.allowed_token_ids(99) == set()

    def test_next_state_transitions(self):
        fsm = TokenFSM(initial=0, finals={2}, map={0: {5: 1}, 1: {3: 2}})
        assert fsm.next_state(0, 5) == 1
        assert fsm.next_state(1, 3) == 2

    def test_next_state_raises_on_invalid(self):
        fsm = TokenFSM(initial=0, finals={1}, map={0: {0: 1}})
        with pytest.raises(KeyError):
            fsm.next_state(0, 999)


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
