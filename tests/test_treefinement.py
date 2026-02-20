"""Unit tests for self_correction/self_correction.py."""
import sys
import os
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import openai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'self_correction'))
import self_correction as treefinement


# --- _call_with_retry ---

class TestCallWithRetry:
    def test_succeeds_on_first_try(self):
        fn = MagicMock(return_value="ok")
        result = treefinement._call_with_retry(fn, max_retries=3, backoff=0.0)
        assert result == "ok"
        assert fn.call_count == 1

    def test_retries_on_api_error_then_succeeds(self):
        fn = MagicMock(side_effect=[
            openai.APIConnectionError(request=MagicMock()),
            "recovered",
        ])
        result = treefinement._call_with_retry(fn, max_retries=3, backoff=0.0)
        assert result == "recovered"
        assert fn.call_count == 2

    def test_returns_empty_after_max_retries(self):
        fn = MagicMock(side_effect=openai.APIConnectionError(request=MagicMock()))
        result = treefinement._call_with_retry(fn, max_retries=2, backoff=0.0)
        assert result == ""
        assert fn.call_count == 2

    def test_retries_on_rate_limit(self):
        response = MagicMock()
        response.status_code = 429
        response.headers = {}
        fn = MagicMock(side_effect=[
            openai.RateLimitError(
                message="rate limited",
                response=response,
                body=None,
            ),
            "ok",
        ])
        result = treefinement._call_with_retry(fn, max_retries=3, backoff=0.0)
        assert result == "ok"

    def test_single_retry_on_exhaustion(self):
        fn = MagicMock(side_effect=openai.APIConnectionError(request=MagicMock()))
        result = treefinement._call_with_retry(fn, max_retries=1, backoff=0.0)
        assert result == ""
        assert fn.call_count == 1


# --- initialize ---

class TestInitialize:
    def test_returns_list_of_message_pairs(self):
        programs = ["fn foo() {}", "fn bar() {}"]
        errors = ["error: something", "error: other"]
        result = treefinement.initialize(programs, errors)
        assert len(result) == 2
        for msgs in result:
            assert isinstance(msgs, list)
            assert len(msgs) == 2
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"

    def test_system_prompt_included(self):
        result = treefinement.initialize(["code"], ["err"])
        assert "Verus" in result[0][0]["content"]

    def test_program_and_error_in_user_message(self):
        result = treefinement.initialize(["fn my_func() {}"], ["error: line 5"])
        user_msg = result[0][1]["content"]
        assert "fn my_func() {}" in user_msg
        assert "error: line 5" in user_msg

    def test_empty_input(self):
        result = treefinement.initialize([], [])
        assert result == []


# --- refinement_message ---

class TestRefinementMessage:
    def test_returns_single_user_message(self):
        result = treefinement.refinement_message("error: postcondition failed")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_error_included_in_content(self):
        result = treefinement.refinement_message("error: type mismatch at line 10")
        assert "error: type mismatch at line 10" in result[0]["content"]

    def test_asks_for_fix(self):
        result = treefinement.refinement_message("some error")
        content = result[0]["content"].lower()
        assert "fix" in content


# --- expand_to_batch ---

class TestExpandToBatch:
    def test_empty_input(self):
        progs, errs = treefinement.expand_to_batch([], [], 5)
        assert progs == []
        assert errs == []

    def test_already_large_enough(self):
        progs = ["a", "b", "c"]
        errs = ["e1", "e2", "e3"]
        result_progs, result_errs = treefinement.expand_to_batch(progs, errs, 3)
        assert result_progs == progs
        assert result_errs == errs

    def test_larger_than_n(self):
        progs = ["a", "b", "c", "d"]
        errs = ["e1", "e2", "e3", "e4"]
        result_progs, result_errs = treefinement.expand_to_batch(progs, errs, 2)
        assert result_progs == progs
        assert result_errs == errs

    def test_pads_to_n(self):
        progs = ["a", "b"]
        errs = ["e1", "e2"]
        result_progs, result_errs = treefinement.expand_to_batch(progs, errs, 5)
        assert len(result_progs) == 5
        assert len(result_errs) == 5
        # First k elements should be originals
        assert result_progs[:2] == progs
        assert result_errs[:2] == errs

    def test_padded_elements_are_from_originals(self):
        progs = ["x", "y"]
        errs = ["ex", "ey"]
        random.seed(42)
        result_progs, result_errs = treefinement.expand_to_batch(progs, errs, 10)
        assert all(p in progs for p in result_progs)
        assert all(e in errs for e in result_errs)

    def test_single_element(self):
        progs = ["only"]
        errs = ["err"]
        result_progs, result_errs = treefinement.expand_to_batch(progs, errs, 4)
        assert len(result_progs) == 4
        assert all(p == "only" for p in result_progs)


# --- display_rust_program_with_errors ---

class TestDisplayRustProgramWithErrors:
    def test_contains_rust_code_block(self):
        result = treefinement.display_rust_program_with_errors("fn foo() {}", "no errors")
        assert "```rust" in result
        assert "fn foo() {}" in result

    def test_contains_error_message(self):
        result = treefinement.display_rust_program_with_errors("code", "error: line 5")
        assert "error: line 5" in result
        assert "Error Message" in result

    def test_formatting(self):
        result = treefinement.display_rust_program_with_errors("code", "err")
        assert result.count("```") >= 4  # two code blocks (open + close each)


# --- parse_args ---

class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["self_correction.py"]):
            args = treefinement.parse_args()
        assert args.model == "qwen2.5-coder:7b"
        assert args.batch_size == 32
        assert args.first_stage == 8
        assert args.max_iters == 5
        assert args.temperature == 0.7
        assert args.rebase_temperature == 0.1
        assert args.max_tokens == 2048
        assert args.output == "verified_program.rs"

    def test_custom_args(self):
        with patch("sys.argv", [
            "self_correction.py",
            "--model", "my-model",
            "--batch-size", "5",
            "--first-stage", "20",
            "--max-iters", "3",
            "--temperature", "0.9",
        ]):
            args = treefinement.parse_args()
        assert args.model == "my-model"
        assert args.batch_size == 5
        assert args.first_stage == 20
        assert args.max_iters == 3
        assert args.temperature == 0.9

    def test_input_defaults_to_example(self):
        with patch("sys.argv", ["self_correction.py"]):
            args = treefinement.parse_args()
        assert args.input.endswith("incr_list.rs")


# --- parallel_generator (mocked) ---

class TestParallelGenerator:
    def test_returns_list_of_strings(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "```rust\nfn body() {}\n```"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "model": "test-model",
            "generation_params": {"temperature": 0.7, "max_tokens": 512},
            "batch_size": 2,
            "first_stage": 3,
        }
        results = treefinement.parallel_generator(config, "spec code", mock_client)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all("fn body()" in r for r in results)


# --- refinement_generator (mocked) ---

class TestRefinementGenerator:
    def test_returns_content(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "fixed code"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        state = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix this"},
        ]
        result = treefinement.refinement_generator(
            state, model="m", temperature=0.7, max_tokens=512, client=mock_client
        )
        assert result == "fixed code"


# --- async_refinement_generator (mocked) ---

class TestAsyncRefinementGenerator:
    def test_generates_for_all_states(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "refined"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        history = [
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}],
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u2"}],
        ]
        results = treefinement.async_refinement_generator(
            history, model="m", temperature=0.7, max_tokens=512,
            batch_size=2, client=mock_client,
        )
        assert len(results) == 2
        assert all(r == "refined" for r in results)

    def test_filters_empty_results(self):
        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_choice = MagicMock()
            # Second call returns empty (simulating retry failure)
            mock_choice.message.content = "ok" if call_count != 2 else ""
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            return mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        history = [
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}],
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u2"}],
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u3"}],
        ]
        results = treefinement.async_refinement_generator(
            history, model="m", temperature=0.7, max_tokens=512,
            batch_size=3, client=mock_client,
        )
        # One empty result should be filtered out
        assert len(results) == 2
