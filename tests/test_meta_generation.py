"""Unit tests for meta_generation/meta_generation.py."""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'meta_generation'))


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


# --- generate_samples (mocked) ---

class TestGenerateSamples:
    @pytest.fixture(autouse=True)
    def _import(self):
        from meta_generation import generate_samples
        self.generate_samples = generate_samples

    def test_returns_n_samples(self):
        mock_logprob = MagicMock()
        mock_logprob.logprob = -1.0

        def fake_create(**kwargs):
            mock_choice = MagicMock()
            mock_choice.message.content = "```python\ndef f(): return 1\n```"
            mock_choice.logprobs.content = [mock_logprob]
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response

        with patch("meta_generation.client") as mock_client:
            mock_client.chat.completions.create.side_effect = fake_create
            codes, logprobs = self.generate_samples(
                "prompt", "model", n=3, max_workers=1
            )

        assert len(codes) == 3
        assert len(logprobs) == 3
        assert all(lp == -1.0 for lp in logprobs)

    def test_filters_failed_samples(self):
        mock_logprob = MagicMock()
        mock_logprob.logprob = -1.0
        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("API error")
            mock_choice = MagicMock()
            mock_choice.message.content = "```python\ndef f(): return 1\n```"
            mock_choice.logprobs.content = [mock_logprob]
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response

        with patch("meta_generation.client") as mock_client:
            mock_client.chat.completions.create.side_effect = fake_create
            codes, logprobs = self.generate_samples(
                "prompt", "model", n=3, max_workers=1
            )

        # One of 3 failed, so we should get 2 valid samples
        assert len(codes) == 2
        assert all(c != "" for c in codes)


# --- Module constants ---

class TestConstants:
    def test_methods_list(self):
        from meta_generation import METHODS
        assert METHODS == ["pass@1", "logprob", "edit", "exec"]
