"""Unit tests for the decomposed functions in self_correction/self_correction.py:
filter_valid, deduplicate, rebase_select, and iterative_refinement."""
import shutil
import sys
import os
import random
from copy import deepcopy
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'self_correction'))
import self_correction as sc
from examples.valid_not_verified import INPUT_PROGRAM, VALID_NOT_VERIFIED

VERUS_PATH = os.environ.get("VERUS_PATH", "/Users/lelarge/verus/verus")
has_verus = shutil.which(VERUS_PATH) is not None or os.path.isfile(VERUS_PATH)


# --- first_verif (integration with Verus) ---

@pytest.mark.skipif(not has_verus, reason="Verus binary not found")
class TestFirstVerifIntegration:
    """Integration tests that run first_verif against real Verus."""

    def _wrap_as_generations(self, bodies):
        return ["```rust\n" + body + "\n```" for body in bodies]

    def test_all_five_examples_are_valid(self):
        generations = self._wrap_as_generations(VALID_NOT_VERIFIED)
        valid, verified, invalid_count = sc.first_verif(
            generations, INPUT_PROGRAM, VERUS_PATH
        )
        assert len(valid) == 5, (
            f"Expected 5 valid, got {len(valid)} valid, "
            f"{len(verified)} verified, {invalid_count} invalid"
        )
        assert len(verified) == 0
        assert invalid_count == 0

    def test_each_example_is_valid_individually(self):
        for i, body in enumerate(VALID_NOT_VERIFIED):
            generations = self._wrap_as_generations([body])
            valid, verified, invalid_count = sc.first_verif(
                generations, INPUT_PROGRAM, VERUS_PATH
            )
            assert len(valid) == 1, (
                f"Program {i+1} not valid: "
                f"{len(valid)} valid, {len(verified)} verified, {invalid_count} invalid"
            )

    def test_correct_program_is_verified(self):
        correct_body = """\
    let mut result = Vec::with_capacity(l.len());
    for i in 0..l.len()
        invariant
            result.len() == i,
            forall|j: int| 0 <= j < i ==> #[trigger] result[j] == l[j] + 1,
            forall|j: int| 0 <= j < l.len() ==> l[j] + 1 <= i32::MAX,
    {
        result.push(l[i] + 1);
    }
    result
} // End of function
} // verus!
fn main() {}
"""
        generations = self._wrap_as_generations([correct_body])
        valid, verified, invalid_count = sc.first_verif(
            generations, INPUT_PROGRAM, VERUS_PATH
        )
        assert len(verified) == 1, (
            f"Expected verified, got {len(valid)} valid, "
            f"{len(verified)} verified, {invalid_count} invalid"
        )


# --- first_verif (unit tests) ---

class TestFirstVerif:
    @patch("self_correction.utils.check")
    @patch("self_correction.utils.parse_generation")
    def test_separates_valid_verified_invalid(self, mock_parse, mock_check):
        mock_parse.side_effect = lambda inp, gen: f"parsed_{gen}"
        mock_check.side_effect = lambda code, path: {
            "parsed_g0": {"verified": 0, "extracted_code": "c0"},
            "parsed_g1": {"verified": 1, "extracted_code": "c1"},
            "parsed_g2": {"verified": -1, "extracted_code": "c2"},
        }[code]

        valid, verified, invalid_count = sc.first_verif(
            ["g0", "g1", "g2"], "input_prog", "/verus"
        )
        assert len(valid) == 1
        assert valid[0]["extracted_code"] == "c0"
        assert len(verified) == 1
        assert verified[0]["extracted_code"] == "c1"
        assert invalid_count == 1

    @patch("self_correction.utils.check")
    @patch("self_correction.utils.parse_generation")
    def test_empty_generations(self, mock_parse, mock_check):
        valid, verified, invalid_count = sc.first_verif([], "prog", "/verus")
        assert valid == []
        assert verified == []
        assert invalid_count == 0
        mock_parse.assert_not_called()
        mock_check.assert_not_called()

    @patch("self_correction.utils.check")
    @patch("self_correction.utils.parse_generation")
    def test_all_verified(self, mock_parse, mock_check):
        mock_parse.side_effect = lambda inp, gen: gen
        mock_check.return_value = {"verified": 1, "extracted_code": "code"}

        valid, verified, invalid_count = sc.first_verif(
            ["a", "b", "c"], "prog", "/verus"
        )
        assert len(valid) == 0
        assert len(verified) == 3
        assert invalid_count == 0

    @patch("self_correction.utils.check")
    @patch("self_correction.utils.parse_generation")
    def test_all_invalid(self, mock_parse, mock_check):
        mock_parse.side_effect = lambda inp, gen: gen
        mock_check.return_value = {"verified": -1, "extracted_code": "code"}

        valid, verified, invalid_count = sc.first_verif(
            ["a", "b"], "prog", "/verus"
        )
        assert len(valid) == 0
        assert len(verified) == 0
        assert invalid_count == 2

    @patch("self_correction.utils.check")
    @patch("self_correction.utils.parse_generation")
    def test_passes_input_program_and_verus_path(self, mock_parse, mock_check):
        mock_parse.return_value = "parsed"
        mock_check.return_value = {"verified": 0, "extracted_code": "c"}

        sc.first_verif(["gen"], "my_program", "/my/verus")
        mock_parse.assert_called_once_with("my_program", "gen")
        mock_check.assert_called_once_with("parsed", "/my/verus")


# --- filter_valid ---

class TestFilterValid:
    def test_keeps_nonnegative_scores(self):
        states = [["s0"], ["s1"], ["s2"]]
        scored = [(0.5, "err0"), (0.0, "err1"), (-1.0, "err2")]
        result = sc.filter_valid(states, scored)
        assert len(result) == 2
        assert result[0] == (["s0"], 0.5, "err0")
        assert result[1] == (["s1"], 0.0, "err1")

    def test_empty_input(self):
        assert sc.filter_valid([], []) == []

    def test_all_negative(self):
        states = [["a"], ["b"]]
        scored = [(-1.0, "e1"), (-0.1, "e2")]
        assert sc.filter_valid(states, scored) == []

    def test_all_positive(self):
        states = [["a"], ["b"]]
        scored = [(1.0, "e1"), (2.0, "e2")]
        result = sc.filter_valid(states, scored)
        assert len(result) == 2

    def test_preserves_order(self):
        states = [["a"], ["b"], ["c"]]
        scored = [(3.0, "e1"), (-1.0, "e2"), (1.0, "e3")]
        result = sc.filter_valid(states, scored)
        assert result[0][1] == 3.0
        assert result[1][1] == 1.0


# --- deduplicate ---

class TestDeduplicate:
    @patch("self_correction.utils.node_to_code")
    def test_removes_duplicates_keeps_best_score(self, mock_node_to_code):
        mock_node_to_code.side_effect = lambda prog, state: state[0]["code"]
        triples = [
            ([{"code": "A"}], 0.5, "err1"),
            ([{"code": "A"}], 0.8, "err2"),
            ([{"code": "B"}], 0.3, "err3"),
        ]
        result = sc.deduplicate(triples, "prog", iteration_number=0)
        assert len(result) == 2
        codes = [mock_node_to_code("prog", r[0]) for r in result]
        assert "A" in codes
        assert "B" in codes
        # The "A" entry kept should be the one with score 0.8
        a_triple = [r for r in result if r[0][0]["code"] == "A"][0]
        assert a_triple[1] == 0.8

    @patch("self_correction.utils.node_to_code")
    def test_no_duplicates_returns_all(self, mock_node_to_code):
        mock_node_to_code.side_effect = lambda prog, state: state[0]["code"]
        triples = [
            ([{"code": "X"}], 1.0, "e1"),
            ([{"code": "Y"}], 2.0, "e2"),
        ]
        result = sc.deduplicate(triples, "prog", iteration_number=0)
        assert len(result) == 2

    @patch("self_correction.utils.node_to_code")
    def test_all_same_code_keeps_one(self, mock_node_to_code):
        mock_node_to_code.return_value = "same"
        triples = [
            (["s1"], 0.1, "e1"),
            (["s2"], 0.9, "e2"),
            (["s3"], 0.5, "e3"),
        ]
        result = sc.deduplicate(triples, "prog", iteration_number=2)
        assert len(result) == 1
        assert result[0][1] == 0.9

    @patch("self_correction.utils.node_to_code")
    def test_single_element(self, mock_node_to_code):
        mock_node_to_code.return_value = "code"
        triples = [(["s"], 0.5, "err")]
        result = sc.deduplicate(triples, "prog", iteration_number=0)
        assert result == triples


# --- rebase_select ---

class TestRebaseSelect:
    def test_returns_correct_batch_size(self):
        triples = [
            (["s0"], 0.5, "e0"),
            (["s1"], 1.0, "e1"),
            (["s2"], 0.2, "e2"),
        ]
        states, scores, errors = sc.rebase_select(triples, batch_size=5, rebase_temperature=0.1)
        assert len(states) == 5
        assert len(scores) == 5
        assert len(errors) == 5

    def test_scores_come_from_input_triples(self):
        triples = [
            (["s0"], 0.5, "e0"),
            (["s1"], 1.0, "e1"),
        ]
        _, scores, _ = sc.rebase_select(triples, batch_size=4, rebase_temperature=0.1)
        for s in scores:
            assert s in (0.5, 1.0)

    def test_low_temperature_favors_best(self):
        triples = [
            (["low"], 0.0, "e0"),
            (["high"], 10.0, "e1"),
        ]
        random.seed(42)
        _, scores, _ = sc.rebase_select(triples, batch_size=20, rebase_temperature=0.01)
        # With very low temperature, almost all samples should be the high-score one
        assert scores.count(10.0) >= 18

    def test_deepcopies_states(self):
        state = [{"key": "val"}]
        triples = [(state, 1.0, "err")]
        states, _, _ = sc.rebase_select(triples, batch_size=2, rebase_temperature=0.1)
        # Mutating returned state should not affect original
        states[0][0]["key"] = "changed"
        assert state[0]["key"] == "val"

    def test_single_triple(self):
        triples = [(["only"], 0.5, "e")]
        states, scores, errors = sc.rebase_select(triples, batch_size=3, rebase_temperature=0.1)
        assert len(states) == 3
        assert all(s == 0.5 for s in scores)
        assert all(e == "e" for e in errors)


# --- iterative_refinement ---

class TestIterativeRefinement:
    _REFINEMENT_KWARGS = dict(
        model="test-model",
        temperature=0.7,
        max_tokens=512,
        batch_size=2,
    )

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    def test_returns_none_when_all_invalid(self, mock_eval_code, mock_eval_node, mock_gen):
        # Note: @patch decorators apply bottom-up, so params match inner-to-outer
        mock_eval_code.return_value = (0.5, "some error")
        mock_gen.return_value = ["gen1", "gen2"]
        mock_eval_node.return_value = (-1.0, "bad")

        valid = [{"extracted_code": "code1"}, {"extracted_code": "code2"}]
        history, scores = sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            **self._REFINEMENT_KWARGS,
            client=MagicMock(),
            max_iters=2,
            rebase_temperature=0.1,
        )
        assert history is None
        assert scores is None

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    @patch("self_correction.utils.node_to_code")
    def test_returns_on_verified(self, mock_ntc, mock_eval_code, mock_eval_node, mock_gen):
        mock_eval_code.return_value = (0.5, "error msg")
        mock_gen.return_value = ["gen1", "gen2"]
        mock_eval_node.return_value = (1.0, "")
        mock_ntc.side_effect = lambda prog, state: str(id(state))

        valid = [{"extracted_code": "code1"}, {"extracted_code": "code2"}]
        history, scores = sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            **self._REFINEMENT_KWARGS,
            client=MagicMock(),
            max_iters=5,
            rebase_temperature=0.1,
        )
        assert history is not None
        assert any(s >= 1.0 for s in scores)

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    @patch("self_correction.utils.node_to_code")
    def test_runs_max_iters_without_verification(self, mock_ntc, mock_eval_code, mock_eval_node, mock_gen):
        mock_eval_code.return_value = (0.3, "error")
        mock_gen.return_value = ["gen1", "gen2"]
        mock_eval_node.return_value = (0.5, "still errors")
        mock_ntc.side_effect = lambda prog, state: str(id(state))

        valid = [{"extracted_code": "code1"}, {"extracted_code": "code2"}]
        history, scores = sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            **self._REFINEMENT_KWARGS,
            client=MagicMock(),
            max_iters=3,
            rebase_temperature=0.1,
        )
        assert history is None
        assert scores is None

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    @patch("self_correction.utils.node_to_code")
    def test_mixed_valid_and_invalid_within_iteration(self, mock_ntc, mock_eval_code, mock_eval_node, mock_gen):
        """Some states are invalid (score < 0), others valid — only valid survive."""
        mock_eval_code.return_value = (0.3, "error")
        mock_gen.return_value = ["gen1", "gen2"]
        # First state invalid, second valid — alternating per call
        mock_eval_node.side_effect = [(-1.0, "bad"), (0.6, "partial error")] * 5
        mock_ntc.side_effect = lambda prog, state: str(id(state))

        valid = [{"extracted_code": "code1"}, {"extracted_code": "code2"}]
        history, scores = sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            **self._REFINEMENT_KWARGS,
            client=MagicMock(),
            max_iters=2,
            rebase_temperature=0.1,
        )
        # Not verified, but the loop should complete without error
        assert history is None
        assert scores is None

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    @patch("self_correction.utils.node_to_code")
    def test_recovers_after_all_invalid_iteration(self, mock_ntc, mock_eval_code, mock_eval_node, mock_gen):
        """First iteration all invalid (triggers continue), second iteration finds verified."""
        mock_eval_code.return_value = (0.3, "error")
        mock_gen.return_value = ["gen1", "gen2"]
        # Iteration 1: all invalid → continue; Iteration 2: all verified
        mock_eval_node.side_effect = [
            (-1.0, "bad"), (-1.0, "bad"),   # iter 1: all invalid
            (1.0, ""), (1.0, ""),            # iter 2: verified
        ]
        mock_ntc.side_effect = lambda prog, state: str(id(state))

        valid = [{"extracted_code": "code1"}, {"extracted_code": "code2"}]
        history, scores = sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            **self._REFINEMENT_KWARGS,
            client=MagicMock(),
            max_iters=3,
            rebase_temperature=0.1,
        )
        assert history is not None
        assert any(s >= 1.0 for s in scores)

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    @patch("self_correction.utils.node_to_code")
    def test_expand_to_batch_pads_initial_programs(self, mock_ntc, mock_eval_code, mock_eval_node, mock_gen):
        """When len(valid) < batch_size, expand_to_batch should pad to batch_size."""
        mock_eval_code.return_value = (0.3, "error")
        mock_eval_node.return_value = (0.5, "partial")
        mock_ntc.side_effect = lambda prog, state: str(id(state))
        mock_gen.return_value = ["gen1", "gen2", "gen3"]

        # Only 1 valid program but batch_size=3 — should be padded
        valid = [{"extracted_code": "code1"}]
        history, scores = sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            model="test-model",
            temperature=0.7,
            max_tokens=512,
            batch_size=3,
            client=MagicMock(),
            max_iters=1,
            rebase_temperature=0.1,
        )
        # The generator should have been called with a history of length batch_size (3)
        call_args = mock_gen.call_args
        assert len(call_args.kwargs.get("history", call_args[0][0] if call_args[0] else [])) == 3

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    @patch("self_correction.utils.node_to_code")
    def test_refinement_message_appended_to_history(self, mock_ntc, mock_eval_code, mock_eval_node, mock_gen):
        """After each iteration, refinement_message(error) should be appended to each state."""
        mock_eval_code.return_value = (0.3, "init error")
        mock_eval_node.return_value = (0.5, "iter error")
        mock_ntc.side_effect = lambda prog, state: str(id(state))
        mock_gen.return_value = ["gen1", "gen2"]

        valid = [{"extracted_code": "code1"}, {"extracted_code": "code2"}]
        history, scores = sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            **self._REFINEMENT_KWARGS,
            client=MagicMock(),
            max_iters=2,
            rebase_temperature=0.1,
        )
        # On the second call, history should contain refinement messages from iteration 1
        second_call_history = mock_gen.call_args_list[1].kwargs.get(
            "history", mock_gen.call_args_list[1][0][0] if mock_gen.call_args_list[1][0] else []
        )
        for state in second_call_history:
            # Each state should end with a user refinement message
            assert state[-1]["role"] == "user"
            # The refinement message should contain the error from the previous iteration
            assert "iter error" in state[-1]["content"]

    @patch("self_correction.async_refinement_generator")
    @patch("self_correction.utils.evaluate_node")
    @patch("self_correction.utils.evaluate_code")
    @patch("self_correction.utils.node_to_code")
    def test_generator_called_exact_max_iters_times(self, mock_ntc, mock_eval_code, mock_eval_node, mock_gen):
        """The LLM generator should be called exactly max_iters times when nothing verifies."""
        mock_eval_code.return_value = (0.3, "error")
        mock_gen.return_value = ["gen1", "gen2"]
        mock_eval_node.return_value = (0.5, "still errors")
        mock_ntc.side_effect = lambda prog, state: str(id(state))

        valid = [{"extracted_code": "code1"}, {"extracted_code": "code2"}]
        sc.iterative_refinement(
            valid=valid,
            input_program="fn foo() {}",
            verus_path="/fake",
            **self._REFINEMENT_KWARGS,
            client=MagicMock(),
            max_iters=4,
            rebase_temperature=0.1,
        )
        assert mock_gen.call_count == 4
