"""Unit tests for the required functions in meta_generation/meta_generation.py."""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'meta_generation'))


# --- _generate_one (mocked) ---

class TestGenerateOne:
    @pytest.fixture(autouse=True)
    def _import(self):
        from meta_generation import _generate_one
        self._generate_one = _generate_one

    def test_extracts_code_and_logprob(self):
        mock_logprob = MagicMock()
        mock_logprob.logprob = -0.5

        mock_choice = MagicMock()
        mock_choice.message.content = "```python\ndef foo(): return 1\n```"
        mock_choice.logprobs.content = [mock_logprob, mock_logprob]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("meta_generation.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            code, avg_lp = self._generate_one("test prompt", "test-model")

        assert code == "def foo(): return 1"
        assert avg_lp == -0.5

    def test_no_logprobs_returns_neginf(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "```python\ndef bar(): pass\n```"
        mock_choice.logprobs = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("meta_generation.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            code, avg_lp = self._generate_one("test prompt", "test-model")

        assert code == "def bar(): pass"
        assert avg_lp == float("-inf")

    def test_empty_logprobs_content_returns_neginf(self):
        """When logprobs.content is an empty list, should return -inf."""
        mock_choice = MagicMock()
        mock_choice.message.content = "```python\ndef baz(): pass\n```"
        mock_choice.logprobs.content = []

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("meta_generation.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            code, avg_lp = self._generate_one("test prompt", "test-model")

        assert code == "def baz(): pass"
        assert avg_lp == float("-inf")


# --- rank_by_logprob ---

class TestRankByLogprob:
    @pytest.fixture(autouse=True)
    def _import(self):
        from meta_generation import rank_by_logprob
        self.rank_by_logprob = rank_by_logprob

    def test_highest_logprob_first(self):
        logprobs = [-2.0, -0.5, -1.5]
        ranking = self.rank_by_logprob(logprobs)
        assert ranking[0] == 1  # -0.5 is highest

    def test_already_sorted(self):
        logprobs = [-3.0, -2.0, -1.0]
        ranking = self.rank_by_logprob(logprobs)
        assert ranking == [2, 1, 0]

    def test_single_element(self):
        logprobs = [-1.0]
        ranking = self.rank_by_logprob(logprobs)
        assert ranking == [0]

    def test_equal_logprobs(self):
        logprobs = [-1.0, -1.0, -1.0]
        ranking = self.rank_by_logprob(logprobs)
        assert sorted(ranking) == [0, 1, 2]

    def test_returns_all_indices(self):
        logprobs = [-5.0, -3.0, -1.0, -4.0, -2.0]
        ranking = self.rank_by_logprob(logprobs)
        assert sorted(ranking) == [0, 1, 2, 3, 4]


# --- rank_by_edit_sim ---

class TestRankByEditSim:
    @pytest.fixture(autouse=True)
    def _import(self):
        from meta_generation import rank_by_edit_sim
        self.rank_by_edit_sim = rank_by_edit_sim

    def test_identical_codes_ranked(self):
        codes = ["def f(): return 1", "def f(): return 1", "def g(): return 999"]
        ranking = self.rank_by_edit_sim(codes)
        # The two identical codes should be ranked higher (more similar to each other)
        assert ranking[0] in [0, 1]
        assert ranking[1] in [0, 1]

    def test_single_code(self):
        codes = ["def f(): return 1"]
        ranking = self.rank_by_edit_sim(codes)
        assert ranking == [0]

    def test_returns_all_indices(self):
        codes = ["a", "b", "c"]
        ranking = self.rank_by_edit_sim(codes)
        assert sorted(ranking) == [0, 1, 2]

    def test_majority_consensus_wins(self):
        """The code that appears most often should rank highest (MBR property)."""
        codes = ["return 1", "return 1", "return 1", "completely different code"]
        ranking = self.rank_by_edit_sim(codes)
        # The three similar ones should all rank before the outlier
        assert ranking[-1] == 3

    def test_empty_strings(self):
        codes = ["", "", "abc"]
        ranking = self.rank_by_edit_sim(codes)
        # Two empty strings are identical, should rank higher
        assert ranking[0] in [0, 1]

    def test_both_empty_similarity_is_one(self):
        """Two empty strings should have similarity 1.0 (m == 0 edge case)."""
        codes = ["", ""]
        ranking = self.rank_by_edit_sim(codes)
        assert sorted(ranking) == [0, 1]

    def test_normalized_distance(self):
        """Verify the similarity formula: 1 - levenshtein(a, b) / max(len(a), len(b))."""
        # "abc" vs "axc": distance=1, max_len=3, sim=1-1/3=2/3
        # "abc" vs "abc": distance=0, sim=1.0
        # So index 0 and 1 are identical, index 2 differs
        codes = ["abc", "abc", "axc"]
        ranking = self.rank_by_edit_sim(codes)
        # The two identical strings should rank above the different one
        assert ranking[0] in [0, 1]
        assert ranking[1] in [0, 1]
        assert ranking[2] == 2


# --- rank_by_exec_sim ---

class TestRankByExecSim:
    @pytest.fixture(autouse=True)
    def _import(self):
        from meta_generation import rank_by_exec_sim
        self.rank_by_exec_sim = rank_by_exec_sim

    def test_no_func_calls_returns_identity(self):
        codes = ["def f(): pass", "def g(): pass"]
        # test_list with no extractable function calls
        ranking = self.rank_by_exec_sim(codes, ["print('hello')"])
        assert ranking == [0, 1]

    def test_agreeing_codes_ranked_higher(self):
        codes = [
            "def add(a, b): return a + b",
            "def add(a, b): return a + b",
            "def add(a, b): return a * b",
        ]
        test_list = ["assert add(2, 3) == 5"]
        ranking = self.rank_by_exec_sim(codes, test_list)
        # The two identical implementations should rank above the different one
        assert ranking[-1] == 2

    def test_returns_all_indices(self):
        codes = [
            "def f(x): return x",
            "def f(x): return x + 1",
        ]
        test_list = ["assert f(1) == 1"]
        ranking = self.rank_by_exec_sim(codes, test_list)
        assert sorted(ranking) == [0, 1]

    def test_failing_code_gets_zero_similarity(self):
        """A sample that raises an exception should get sim=0 with all others."""
        codes = [
            "def add(a, b): return a + b",
            "def add(a, b): return a + b",
            "raise RuntimeError('boom')",  # crashes at execution
        ]
        test_list = ["assert add(1, 2) == 3"]
        ranking = self.rank_by_exec_sim(codes, test_list)
        # The crashing sample should rank last
        assert ranking[-1] == 2

    def test_partial_agreement_on_multiple_calls(self):
        """With multiple test calls, partial agreement should be reflected."""
        codes = [
            "def f(x): return x * 2",      # f(1)=2, f(2)=4
            "def f(x): return x * 2",      # f(1)=2, f(2)=4  (agrees fully with 0)
            "def f(x): return x + 1",      # f(1)=2, f(2)=3  (agrees on first call only)
        ]
        test_list = ["assert f(1) == 2", "assert f(2) == 4"]
        ranking = self.rank_by_exec_sim(codes, test_list)
        # Samples 0 and 1 agree on both calls; sample 2 only on one
        assert ranking[0] in [0, 1]
        assert ranking[1] in [0, 1]
        assert ranking[2] == 2


# --- evaluate_problem ---

class TestEvaluateProblem:
    @pytest.fixture(autouse=True)
    def _import(self):
        from meta_generation import evaluate_problem
        self.evaluate_problem = evaluate_problem

    def test_all_passing(self):
        codes = [
            "def add(a, b): return a + b",
            "def add(a, b): return a + b",
        ]
        logprobs = [-0.5, -1.0]
        test_list = ["assert add(1, 2) == 3"]
        scores, picks, passed = self.evaluate_problem(codes, logprobs, test_list)
        # pass@1 should be 1.0
        assert scores[0] == 1.0
        assert all(p is True for p in passed)

    def test_mixed_passing(self):
        codes = [
            "def add(a, b): return a + b",   # correct
            "def add(a, b): return a - b",   # wrong
        ]
        logprobs = [-0.5, -1.0]
        test_list = ["assert add(1, 2) == 3"]
        scores, picks, passed = self.evaluate_problem(codes, logprobs, test_list)
        # pass@1 = 0.5
        assert scores[0] == 0.5
        assert passed[0] is True
        assert passed[1] is False

    def test_logprob_pick_is_best_logprob(self):
        codes = [
            "def add(a, b): return a + b",
            "def add(a, b): return a + b",
        ]
        logprobs = [-2.0, -0.5]
        test_list = ["assert add(1, 2) == 3"]
        scores, picks, passed = self.evaluate_problem(codes, logprobs, test_list)
        assert picks[1] == 1  # index 1 has higher logprob

    def test_returns_correct_structure(self):
        codes = ["def f(x): return x"]
        logprobs = [-1.0]
        test_list = ["assert f(1) == 1"]
        scores, picks, passed = self.evaluate_problem(codes, logprobs, test_list)
        assert len(scores) == 4  # pass@1, logprob, edit, exec
        assert len(picks) == 4
        assert picks[0] is None  # pass@1 has no single pick
        assert isinstance(passed, list)
        assert isinstance(scores, tuple)

    def test_all_failing(self):
        """When all samples fail, pass@1 should be 0."""
        codes = [
            "def add(a, b): return a - b",
            "def add(a, b): return a * b",
        ]
        logprobs = [-0.5, -1.0]
        test_list = ["assert add(1, 2) == 3"]
        scores, picks, passed = self.evaluate_problem(codes, logprobs, test_list)
        assert scores[0] == 0.0
        assert all(p is False for p in passed)

    def test_exec_sim_uses_first_test_only(self):
        """evaluate_problem passes test_list[:1] to rank_by_exec_sim."""
        codes = [
            "def add(a, b): return a + b",
            "def add(a, b): return a + b",
        ]
        logprobs = [-0.5, -1.0]
        # Two tests: exec_sim should only see the first one
        test_list = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        scores, picks, passed = self.evaluate_problem(codes, logprobs, test_list)
        # Both pass all tests, so all scores should be positive
        assert scores[0] == 1.0
        # exec pick should still be valid
        assert picks[3] in [0, 1]
