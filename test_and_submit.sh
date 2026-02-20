#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Sync dependencies and install the project via uv
uv sync --quiet

LOGFILE="test_and_submit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOGFILE") 2>&1
echo "Logging output to $LOGFILE"

echo "=========================================="
echo " Part 1 — Structured Generation Benchmarks"
echo "=========================================="

echo "--- Naive vs. DFA masking benchmark ---"
uv run python structured_generation/deterministic_finite_automaton.py

echo "--- Character-to-token FSM demo ---"
uv run python structured_generation/fsm_token.py

echo "--- Coalescence benchmark on GPT-2 vocab ---"
uv run python structured_generation/coalescence.py

echo "--- End-to-end constrained generation ---"
uv run python structured_generation/picoGPT_generate.py

echo ""
echo "=========================================="
echo " Running Required Tests"
echo "=========================================="

echo "--- Part 1 tests ---"
uv run pytest tests/test_structured_generation_required.py -v

echo "--- Part 2 tests ---"
uv run pytest tests/test_meta_generation_required.py -v

echo "--- Part 3 tests ---"
uv run pytest tests/test_self_correction_required.py -v

echo ""
echo "=========================================="
echo " All benchmarks and tests completed!"
echo "=========================================="
