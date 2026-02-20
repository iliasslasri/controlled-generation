"""
Evaluate selection strategies on a single MBPP problem.

Generates n samples via Ollama (local), then ranks with:
  1. pass@1 (average pass rate)
  2. Best-of-n (logprob)
  3. MBR (edit similarity)
  4. MBR (exec similarity)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import datasets

datasets.logging.set_verbosity_error()
import jellyfish
from openai import OpenAI

from utils import (
    execute_tests,
    execute_codes,
    make_prompt,
    extract_code,
    extract_func_calls,
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11435/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
DEFAULT_MODEL = "qwen2.5-coder:7b"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

METHODS = ["pass@1", "logprob", "edit", "exec"]


# -- Generation ---------------------------------------------------------------


def _generate_one(prompt, model, **kwargs):
    """Generate a single code sample via Ollama and extract its average log-probability.

    Use ``client.chat.completions.create`` with ``logprobs=True`` and
    ``top_logprobs=1``.  Extract the code from the response with
    ``extract_code``.

    Returns:
        tuple[str, float]: ``(code, avg_logprob)`` where *avg_logprob* is the
        mean of per-token log-probabilities, or ``float("-inf")`` when
        log-probabilities are unavailable.
    """
    # TODO: implement this function
    raise NotImplementedError("_generate_one not implemented")


def generate_samples(prompt, model, n, max_workers=8, **kwargs):
    """Generate n samples in parallel (Ollama does not support n>1 per request)."""
    base_seed = kwargs.pop("seed", None)
    codes = [None] * n
    logprobs_per_sample = [None] * n
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {}
        for i in range(n):
            kw = {**kwargs}
            if base_seed is not None:
                kw["seed"] = base_seed + i
            futs[pool.submit(_generate_one, prompt, model, **kw)] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                codes[i], logprobs_per_sample[i] = fut.result()
            except Exception as e:
                print(f"  sample {i} failed: {e}")
                codes[i] = ""
                logprobs_per_sample[i] = float("-inf")
                failed += 1
            done += 1
            print(f"  sample {done}/{n}")
    if failed:
        print(f"  WARNING: {failed}/{n} samples failed")
    # filter out failed samples
    valid = [(c, lp) for c, lp in zip(codes, logprobs_per_sample) if c != ""]
    if not valid:
        raise RuntimeError("All samples failed to generate")
    codes, logprobs_per_sample = map(list, zip(*valid))
    return codes, logprobs_per_sample


# -- Ranking ------------------------------------------------------------------


def rank_by_logprob(logprobs):
    """Rank sample indices by average log-probability in descending order.

    Args:
        logprobs: list of floats, one per sample.

    Returns:
        list[int]: sample indices sorted from highest to lowest log-probability.
    """
    # TODO: implement this function
    raise NotImplementedError("rank_by_logprob not implemented")


def rank_by_edit_sim(codes):
    """Rank samples by MBR with normalized Levenshtein (edit) similarity.

    For each pair ``(i, j)`` compute::

        sim(i, j) = 1 - levenshtein_distance(codes[i], codes[j])
                        / max(len(codes[i]), len(codes[j]))

    When both strings are empty, similarity is 1.0.
    Accumulate a *gain* per sample (sum of similarities to all others)
    and return indices sorted by gain in descending order.

    Args:
        codes: list of code strings.

    Returns:
        list[int]: sample indices sorted from highest to lowest gain.
    """
    # TODO: implement this function
    raise NotImplementedError("rank_by_edit_sim not implemented")


def rank_by_exec_sim(codes, test_list):
    """Rank samples by MBR with execution similarity.

    1. Extract function calls from ``test_list`` with ``extract_func_calls``.
       If there are none, return ``list(range(len(codes)))``.
    2. Execute every sample with ``execute_codes`` (suppress stdout with
       ``contextlib.redirect_stdout(None)``).
    3. For each pair ``(i, j)``, compute::

           sim(i, j) = (matching outputs) / (total function calls)

       If either result is an ``Exception``, ``sim = 0``.
    4. Accumulate gains and return indices sorted by gain descending.

    Args:
        codes: list of code strings.
        test_list: list of assert statements.

    Returns:
        list[int]: sample indices sorted from highest to lowest gain.
    """
    # TODO: implement this function
    raise NotImplementedError("rank_by_exec_sim not implemented")


# -- Per-problem evaluation ---------------------------------------------------


def evaluate_problem(codes, logprobs, test_list):
    """Evaluate all four selection strategies on a single problem.

    1. Run every sample against the full ``test_list`` using ``execute_tests``
       to obtain a ``passed`` list of booleans.
    2. Build ``picks``: ``[None, logprob_pick, edit_pick, exec_pick]``.
       For exec similarity, use only ``test_list[:1]``.
    3. Compute ``scores``: ``(pass@1, logprob_score, edit_score, exec_score)``
       where pass@1 is the average pass rate and the others are 0 or 1
       depending on whether the picked sample passes.

    Args:
        codes: list of code strings.
        logprobs: list of floats (one per sample).
        test_list: list of assert statements.

    Returns:
        tuple: ``(scores, picks, passed)`` where *scores* is a tuple of 4
        floats, *picks* is a list of 4 (None or int), and *passed* is a
        list of booleans.
    """
    # TODO: implement this function
    raise NotImplementedError("evaluate_problem not implemented")


# -- Main ---------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1618)
    p.add_argument(
        "--index",
        type=int,
        default=13,
        help="Index of the MBPP test problem to evaluate",
    )
    p.add_argument("--output", default="mbpp_eval_results.json")
    args = p.parse_args()

    mbpp = datasets.load_dataset("mbpp", split="test")
    example = mbpp[args.index]
    prompt = make_prompt(example)
    test_list = example["test_list"]
    text = example["text"]

    print(f"Model: {args.model}  n={args.n}")
    print(f"Problem [{args.index}]: {text}")
    print("-" * 70)

    start = time.time()
    gen_kw = dict(temperature=args.temperature, top_p=args.top_p, seed=args.seed)

    print("Generating samples...")
    codes, logprobs = generate_samples(prompt, args.model, args.n, **gen_kw)

    print("Evaluating...")
    scores, picks, passed = evaluate_problem(codes, logprobs, test_list)

    print(f"\n{'Method':<12} Score\n" + "-" * 20)
    for m, s in zip(METHODS, scores):
        print(f"{m:<12} {s:.4f}")
    print(f"Total: {time.time()-start:.1f}s")

    for m, s, pick in zip(METHODS, scores, picks):
        if s == 1 and pick is not None:
            print(f"\n--- {m} selected code (sample {pick}) ---")
            print(codes[pick])

    result = dict(
        model=args.model,
        n=args.n,
        task_id=int(example["task_id"]),
        problem=text,
        **{m: s for m, s in zip(METHODS, scores)},
        codes=codes,
        logprobs=logprobs,
        passed=passed,
    )
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
