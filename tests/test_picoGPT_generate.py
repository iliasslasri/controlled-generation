"""Unit tests for structured_generation/picoGPT_generate.py."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structured_generation'))
# Purge any cached 'utils' from other test modules (e.g. meta_generation/utils)
# so that picoGPT_generate's `from utils import ...` resolves to structured_generation/utils.
sys.modules.pop("utils", None)
from picoGPT_generate import (
    gelu,
    softmax,
    layer_norm,
    linear,
    ffn,
    attention,
    mha,
    transformer_block,
    gpt2,
    generate_unconstrained,
    generate_constrained,
)


# --- gelu ---

class TestGelu:
    def test_zero(self):
        result = gelu(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    def test_positive(self):
        x = np.array([1.0, 2.0])
        result = gelu(x)
        # GELU(x) ≈ x for large positive x
        assert np.all(result > 0)
        assert result[1] > result[0]

    def test_negative(self):
        result = gelu(np.array([-3.0]))
        # GELU(-3) ≈ 0 (very small)
        assert abs(result[0]) < 0.01

    def test_shape_preserved(self):
        x = np.random.randn(3, 4)
        assert gelu(x).shape == (3, 4)


# --- softmax ---

class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-7)

    def test_all_positive(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = softmax(x)
        assert np.all(result > 0)

    def test_largest_input_gets_highest_prob(self):
        x = np.array([1.0, 5.0, 2.0])
        result = softmax(x)
        assert np.argmax(result) == 1

    def test_2d_softmax(self):
        x = np.array([[1.0, 2.0], [3.0, 1.0]])
        result = softmax(x)
        np.testing.assert_allclose(np.sum(result, axis=-1), [1.0, 1.0], atol=1e-7)

    def test_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-7)


# --- layer_norm ---

class TestLayerNorm:
    def test_output_shape(self):
        x = np.random.randn(5, 8)
        g = np.ones(8)
        b = np.zeros(8)
        result = layer_norm(x, g, b)
        assert result.shape == (5, 8)

    def test_identity_with_unit_params(self):
        """With g=1, b=0, output should be zero-mean, unit-variance."""
        x = np.random.randn(10, 16)
        g = np.ones(16)
        b = np.zeros(16)
        result = layer_norm(x, g, b)
        means = np.mean(result, axis=-1)
        variances = np.var(result, axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(variances, 1.0, atol=1e-3)

    def test_scale_and_shift(self):
        x = np.array([[1.0, 2.0, 3.0]])
        g = np.array([2.0, 2.0, 2.0])
        b = np.array([1.0, 1.0, 1.0])
        result = layer_norm(x, g, b)
        # Should be scaled by 2 and shifted by 1
        assert result.shape == (1, 3)
        # Mean of normalized should be 0, so mean of result ≈ b = 1
        np.testing.assert_allclose(np.mean(result), 1.0, atol=1e-5)


# --- linear ---

class TestLinear:
    def test_output_shape(self):
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 8)
        b = np.random.randn(8)
        result = linear(x, w, b)
        assert result.shape == (3, 8)

    def test_bias_addition(self):
        x = np.zeros((1, 2))
        w = np.zeros((2, 3))
        b = np.array([1.0, 2.0, 3.0])
        result = linear(x, w, b)
        np.testing.assert_array_equal(result, [[1.0, 2.0, 3.0]])


# --- ffn ---

class TestFfn:
    def test_output_shape(self):
        d_model = 8
        d_ff = 32
        x = np.random.randn(5, d_model)
        c_fc = {"w": np.random.randn(d_model, d_ff) * 0.01, "b": np.zeros(d_ff)}
        c_proj = {"w": np.random.randn(d_ff, d_model) * 0.01, "b": np.zeros(d_model)}
        result = ffn(x, c_fc, c_proj)
        assert result.shape == (5, d_model)


# --- attention ---

class TestAttention:
    def test_output_shape(self):
        seq_len = 4
        d_k = 8
        q = np.random.randn(seq_len, d_k)
        k = np.random.randn(seq_len, d_k)
        v = np.random.randn(seq_len, d_k)
        mask = (1 - np.tri(seq_len)) * -1e10
        result = attention(q, k, v, mask)
        assert result.shape == (seq_len, d_k)

    def test_causal_masking(self):
        """First position should only attend to itself."""
        seq_len = 3
        d_k = 4
        q = np.random.randn(seq_len, d_k)
        k = np.random.randn(seq_len, d_k)
        v = np.eye(seq_len, d_k)  # v[i] = one-hot(i)
        mask = (1 - np.tri(seq_len)) * -1e10
        result = attention(q, k, v, mask)
        # First row should be v[0] since it can only attend to position 0
        np.testing.assert_allclose(result[0], v[0], atol=1e-5)


# --- mha ---

class TestMha:
    def test_output_shape(self):
        seq_len = 4
        d_model = 8
        n_head = 2
        x = np.random.randn(seq_len, d_model)
        c_attn = {
            "w": np.random.randn(d_model, 3 * d_model) * 0.01,
            "b": np.zeros(3 * d_model),
        }
        c_proj = {
            "w": np.random.randn(d_model, d_model) * 0.01,
            "b": np.zeros(d_model),
        }
        result = mha(x, c_attn, c_proj, n_head)
        assert result.shape == (seq_len, d_model)


# --- transformer_block ---

class TestTransformerBlock:
    def test_output_shape(self):
        seq_len = 4
        d_model = 8
        n_head = 2
        x = np.random.randn(seq_len, d_model)
        block = {
            "ln_1": {"g": np.ones(d_model), "b": np.zeros(d_model)},
            "ln_2": {"g": np.ones(d_model), "b": np.zeros(d_model)},
            "attn": {
                "c_attn": {
                    "w": np.random.randn(d_model, 3 * d_model) * 0.01,
                    "b": np.zeros(3 * d_model),
                },
                "c_proj": {
                    "w": np.random.randn(d_model, d_model) * 0.01,
                    "b": np.zeros(d_model),
                },
            },
            "mlp": {
                "c_fc": {
                    "w": np.random.randn(d_model, 4 * d_model) * 0.01,
                    "b": np.zeros(4 * d_model),
                },
                "c_proj": {
                    "w": np.random.randn(4 * d_model, d_model) * 0.01,
                    "b": np.zeros(d_model),
                },
            },
        }
        result = transformer_block(x, **block, n_head=n_head)
        assert result.shape == (seq_len, d_model)

    def test_residual_connection(self):
        """Output should differ from input (unless weights are pathological)."""
        d_model = 8
        n_head = 2
        np.random.seed(42)
        x = np.random.randn(3, d_model)
        block = {
            "ln_1": {"g": np.ones(d_model), "b": np.zeros(d_model)},
            "ln_2": {"g": np.ones(d_model), "b": np.zeros(d_model)},
            "attn": {
                "c_attn": {
                    "w": np.random.randn(d_model, 3 * d_model) * 0.1,
                    "b": np.zeros(3 * d_model),
                },
                "c_proj": {
                    "w": np.random.randn(d_model, d_model) * 0.1,
                    "b": np.zeros(d_model),
                },
            },
            "mlp": {
                "c_fc": {
                    "w": np.random.randn(d_model, 4 * d_model) * 0.1,
                    "b": np.zeros(4 * d_model),
                },
                "c_proj": {
                    "w": np.random.randn(4 * d_model, d_model) * 0.1,
                    "b": np.zeros(d_model),
                },
            },
        }
        result = transformer_block(x, **block, n_head=n_head)
        assert not np.allclose(result, x)


# --- gpt2 ---

class TestGpt2:
    def _make_tiny_params(self, n_vocab=10, n_seq=4, d_model=8, n_head=2, n_layer=1):
        np.random.seed(42)
        params = {
            "wte": np.random.randn(n_vocab, d_model).astype(np.float32) * 0.01,
            "wpe": np.random.randn(n_seq, d_model).astype(np.float32) * 0.01,
            "blocks": [],
            "ln_f": {
                "g": np.ones(d_model, dtype=np.float32),
                "b": np.zeros(d_model, dtype=np.float32),
            },
        }
        for _ in range(n_layer):
            params["blocks"].append({
                "ln_1": {
                    "g": np.ones(d_model, dtype=np.float32),
                    "b": np.zeros(d_model, dtype=np.float32),
                },
                "ln_2": {
                    "g": np.ones(d_model, dtype=np.float32),
                    "b": np.zeros(d_model, dtype=np.float32),
                },
                "attn": {
                    "c_attn": {
                        "w": np.random.randn(d_model, 3 * d_model).astype(np.float32) * 0.01,
                        "b": np.zeros(3 * d_model, dtype=np.float32),
                    },
                    "c_proj": {
                        "w": np.random.randn(d_model, d_model).astype(np.float32) * 0.01,
                        "b": np.zeros(d_model, dtype=np.float32),
                    },
                },
                "mlp": {
                    "c_fc": {
                        "w": np.random.randn(d_model, 4 * d_model).astype(np.float32) * 0.01,
                        "b": np.zeros(4 * d_model, dtype=np.float32),
                    },
                    "c_proj": {
                        "w": np.random.randn(4 * d_model, d_model).astype(np.float32) * 0.01,
                        "b": np.zeros(d_model, dtype=np.float32),
                    },
                },
            })
        return params, n_head

    def test_output_shape(self):
        n_vocab, n_seq = 10, 4
        params, n_head = self._make_tiny_params(n_vocab=n_vocab, n_seq=n_seq)
        inputs = [0, 1, 2, 3]
        logits = gpt2(inputs, **params, n_head=n_head)
        assert logits.shape == (n_seq, n_vocab)

    def test_different_inputs_give_different_outputs(self):
        n_vocab, n_seq = 10, 3
        params, n_head = self._make_tiny_params(n_vocab=n_vocab, n_seq=n_seq)
        logits1 = gpt2([0, 1, 2], **params, n_head=n_head)
        logits2 = gpt2([3, 4, 5], **params, n_head=n_head)
        assert not np.allclose(logits1, logits2)


# --- generate_unconstrained ---

class TestGenerateUnconstrained:
    def test_generates_tokens(self):
        n_vocab, n_seq = 10, 6
        np.random.seed(42)
        params = {
            "wte": np.random.randn(n_vocab, 8).astype(np.float32) * 0.01,
            "wpe": np.random.randn(n_seq, 8).astype(np.float32) * 0.01,
            "blocks": [{
                "ln_1": {"g": np.ones(8, dtype=np.float32), "b": np.zeros(8, dtype=np.float32)},
                "ln_2": {"g": np.ones(8, dtype=np.float32), "b": np.zeros(8, dtype=np.float32)},
                "attn": {
                    "c_attn": {"w": np.random.randn(8, 24).astype(np.float32) * 0.01,
                               "b": np.zeros(24, dtype=np.float32)},
                    "c_proj": {"w": np.random.randn(8, 8).astype(np.float32) * 0.01,
                               "b": np.zeros(8, dtype=np.float32)},
                },
                "mlp": {
                    "c_fc": {"w": np.random.randn(8, 32).astype(np.float32) * 0.01,
                             "b": np.zeros(32, dtype=np.float32)},
                    "c_proj": {"w": np.random.randn(32, 8).astype(np.float32) * 0.01,
                               "b": np.zeros(8, dtype=np.float32)},
                },
            }],
            "ln_f": {"g": np.ones(8, dtype=np.float32), "b": np.zeros(8, dtype=np.float32)},
        }
        n_head = 2
        input_ids = [0, 1]
        generated = generate_unconstrained(input_ids, params, n_head, max_tokens=3)
        assert len(generated) > 0
        assert len(generated) <= 3
        assert all(0 <= t < n_vocab for t in generated)

    def test_greedy_is_deterministic(self):
        n_vocab, n_seq = 10, 6
        np.random.seed(42)
        params = {
            "wte": np.random.randn(n_vocab, 8).astype(np.float32) * 0.01,
            "wpe": np.random.randn(n_seq, 8).astype(np.float32) * 0.01,
            "blocks": [{
                "ln_1": {"g": np.ones(8, dtype=np.float32), "b": np.zeros(8, dtype=np.float32)},
                "ln_2": {"g": np.ones(8, dtype=np.float32), "b": np.zeros(8, dtype=np.float32)},
                "attn": {
                    "c_attn": {"w": np.random.randn(8, 24).astype(np.float32) * 0.01,
                               "b": np.zeros(24, dtype=np.float32)},
                    "c_proj": {"w": np.random.randn(8, 8).astype(np.float32) * 0.01,
                               "b": np.zeros(8, dtype=np.float32)},
                },
                "mlp": {
                    "c_fc": {"w": np.random.randn(8, 32).astype(np.float32) * 0.01,
                             "b": np.zeros(32, dtype=np.float32)},
                    "c_proj": {"w": np.random.randn(32, 8).astype(np.float32) * 0.01,
                               "b": np.zeros(8, dtype=np.float32)},
                },
            }],
            "ln_f": {"g": np.ones(8, dtype=np.float32), "b": np.zeros(8, dtype=np.float32)},
        }
        n_head = 2
        input_ids = [0, 1]
        gen1 = generate_unconstrained(input_ids, params, n_head, max_tokens=3, temperature=0.0)
        gen2 = generate_unconstrained(input_ids, params, n_head, max_tokens=3, temperature=0.0)
        assert gen1 == gen2


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
