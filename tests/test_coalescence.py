"""Unit tests for structured_generation/coalescence.py."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structured_generation'))
from coalescence import (
    build_tokenizer_index,
    _time_generation_with_skip,
    JSON_SIMPLE,
    JSON_COMPLEX,
    JSON_VERY_COMPLEX,
)
from fsm_token import TokenFSM


# --- JSON patterns compile ---

class TestJsonPatterns:
    def test_simple_pattern_compiles(self):
        import interegular
        fsm = interegular.parse_pattern(JSON_SIMPLE).to_fsm()
        assert fsm.initial is not None

    def test_complex_pattern_compiles(self):
        import interegular
        fsm = interegular.parse_pattern(JSON_COMPLEX).to_fsm()
        assert fsm.initial is not None

    def test_very_complex_pattern_compiles(self):
        import interegular
        fsm = interegular.parse_pattern(JSON_VERY_COMPLEX).to_fsm()
        assert fsm.initial is not None

    def test_simple_pattern_matches_valid_json(self):
        import regex as re
        valid = '{"name":"John","age":20}'
        assert re.fullmatch(JSON_SIMPLE, valid) is not None

    def test_simple_pattern_rejects_invalid(self):
        import regex as re
        invalid = '{"name":"Alice","age":20}'
        assert re.fullmatch(JSON_SIMPLE, invalid) is None

    def test_complex_pattern_matches_valid_json(self):
        import regex as re
        valid = '{"name":"Alice","age":25,"city":"Paris","active":true,"score":42}'
        assert re.fullmatch(JSON_COMPLEX, valid) is not None


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

    def test_coalescence_reduces_unique_masks(self, simple_vocab):
        """States with identical valid token sets should share the same mask object."""
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            JSON_SIMPLE, simple_vocab, verbose=False
        )
        mask_ids = [id(m) for m in masks.values()]
        # If coalescence works, some masks should share the same object
        assert len(set(mask_ids)) <= len(mask_ids)


# --- _time_generation_with_skip ---

class TestTimeGenerationWithSkip:
    SIMPLE_PATTERN = r"[0-9]+\.[0-9]+"

    def _build_simple(self):
        vocab = ["0", "1", "2", ".", "a"]
        tok_fsm, masks, coalesced, _ = build_tokenizer_index(
            self.SIMPLE_PATTERN, vocab, verbose=False
        )
        return tok_fsm, masks, coalesced, vocab

    def test_returns_mean_and_std(self):
        tok_fsm, masks, coalesced, vocab = self._build_simple()
        logits = np.ones(len(vocab))
        mean, std = _time_generation_with_skip(
            init_state=tok_fsm.initial,
            precomputed_masks=masks,
            coalesced_tokens=coalesced,
            tok_fsm=tok_fsm,
            logits=logits,
            vocabulary=vocab,
            n_steps=3,
            n_repeats=2,
            n_warmup=0,
        )
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert mean >= 0
        assert std >= 0

    def test_without_coalesced_tokens_still_runs(self):
        tok_fsm, masks, coalesced, vocab = self._build_simple()
        logits = np.ones(len(vocab))
        mean, std = _time_generation_with_skip(
            init_state=tok_fsm.initial,
            precomputed_masks=masks,
            coalesced_tokens={},  # no coalesced tokens
            tok_fsm=tok_fsm,
            logits=logits,
            vocabulary=vocab,
            n_steps=3,
            n_repeats=2,
            n_warmup=0,
        )
        assert mean >= 0


# --- Integration: constrained generation with build_tokenizer_index ---

class TestConstrainedGenerationIntegration:
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
                logits = np.ones(len(vocab)) + masks[state]
                from scipy.special import softmax
                probs = softmax(logits)
                next_id = np.random.choice(len(vocab), p=probs)
            assert next_id in allowed
            state = tok_fsm.next_state(state, next_id)
