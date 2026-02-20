# Part 2: Meta-Generation [8 points]

When an LLM generates code, a single sample may or may not be correct. A natural idea is to generate **multiple samples** and then **select the best one**. But how do we decide which sample is "best" without already knowing the answer?

This part explores **meta-generation** strategies: generate n candidate solutions, rank them using different selection criteria, and pick the top-ranked one. We evaluate these strategies on the [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (Mostly Basic Programming Problems) benchmark using a locally-served LLM via [Ollama](https://ollama.com/).

Here is an example of a sample from MBPP:
```
{
    'task_id': 1,
    'text': 'Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].',
    'code': 'R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]',
    'test_list': [
        'assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8',
        'assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12',
        'assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16'],
    'test_setup_code': '',
    'challenge_test_list': []
}
```
From the `text` field, the LLM must generate Python code. For Strategy 3 below, the LLM is allowed to observe the execution of its code on the first test in `test_list`. To check whether the generated code is correct, we run all remaining tests.



## Overview

| Module | Description |
|---|---|
| `meta_generation.py` | Sample generation, ranking strategies, and evaluation |
| `utils.py` | Prompt construction, code extraction, and safe code execution (utility, no exercises) |

---

## The Problem: Best-of-N Selection

Given a coding problem, we:
1. Generate **n** candidate solutions from the LLM (with temperature > 0 for diversity)
2. **Rank** the candidates using a selection strategy
3. **Pick** the top-ranked candidate as our answer

The question is: which ranking strategy gives us the best chance of picking a correct solution?

---

## Baseline: pass@1

The simplest metric is **pass@1**: what fraction of the n samples pass all tests? This measures the overall quality of the model but does not involve any selection — it is the expected success rate if we pick a sample at random.

```
pass@1 = (number of passing samples) / n
```

This serves as the baseline. Any selection strategy that consistently beats pass@1 is adding value.

---

## Strategy 1: Best-of-N by Log-Probability (`rank_by_logprob`)

### Idea

The LLM assigns a probability to each token it generates. We can use this as a **confidence score**: samples where the model was more confident (higher log-probabilities) might be more likely to be correct.

### Algorithm

1. For each sample, compute the **average log-probability** across all generated tokens:
   ```
   avg_logprob = sum(logprob_i for each token i) / number_of_tokens
   ```
2. Rank samples by `avg_logprob` in descending order
3. Pick the top-ranked sample

### Properties

- Fast: no execution needed, the log-probabilities come for free from the generation step
- Relies on the assumption that model confidence correlates with correctness
- Can fail when the model is confidently wrong

### What to implement

- `_generate_one(prompt, model, **kwargs)`: generate a single code sample and extract its average log-probability from the API response.
- `rank_by_logprob(logprobs)`: sort sample indices by log-probability in descending order.

---

## Strategy 2: MBR with Edit Similarity (`rank_by_edit_sim`)

### Idea

**Minimum Bayes Risk (MBR) decoding** selects the sample that is most similar to all other samples. The intuition is: if many independent samples converge to similar code, that code is more likely to be correct (majority vote at the code level).

Here we measure similarity using **Levenshtein (edit) distance** between code strings.

### Algorithm

1. For each pair of samples `(i, j)`, compute the normalized edit similarity:
   ```
   sim(i, j) = 1 - levenshtein_distance(code_i, code_j) / max(len(code_i), len(code_j))
   ```
2. For each sample `i`, compute its **gain** (total similarity to all other samples):
   ```
   gain(i) = sum(sim(i, j) for all j)
   ```
3. Rank samples by `gain` in descending order
4. Pick the top-ranked sample (the one most representative of the consensus)

### Properties

- No execution needed — purely text-based
- Captures syntactic consensus across samples
- O(n^2) pairwise comparisons, but fast since it only compares strings
- Can be fooled by syntactically similar but semantically different code

### What to implement

- `rank_by_edit_sim(codes)`: compute pairwise edit similarities, accumulate gains, and return sorted indices. (Hint: use the `jellyfish` library to compute Levenshtein distance.)

---

## Strategy 3: MBR with Execution Similarity (`rank_by_exec_sim`)

### Idea

Instead of comparing code as text, we compare code by its **behavior**: run each sample on a set of test inputs and compare the outputs. Two samples are similar if they produce the same results on the same inputs.

This is a stronger form of MBR — it captures semantic equivalence rather than syntactic similarity.

### Algorithm

1. Extract function calls from the test cases (e.g., `func(arg1, arg2)` from `assert func(arg1, arg2) == expected`). (Hint: use `extract_func_calls` from `utils.py`.)
2. Execute each code sample with these function calls in sandboxed subprocesses. (Hint: use `execute_codes` from `utils.py`.)
3. For each pair `(i, j)`, compute the execution similarity:
   ```
   sim(i, j) = (number of calls where output_i == output_j) / (total number of calls)
   ```
   If either sample fails to execute, set `sim(i, j) = 0`.
4. Accumulate gains and rank as in Strategy 2.

### Properties

- Most semantically meaningful: measures actual behavioral agreement
- Requires code execution (sandboxed for safety)
- More expensive than text-based methods, but more robust
- Can handle functionally equivalent but syntactically different code

### What to implement

- `rank_by_exec_sim(codes, test_list)`: extract function calls, execute all samples, compute pairwise execution similarities, and return sorted indices. (Hint: use `execute_tests` from `utils.py`)

---

## Evaluation (`evaluate_problem`)

For a single MBPP problem, the evaluation function:

1. Runs all n code samples against the full test suite to determine pass/fail for each
2. Applies each ranking strategy to select one sample
3. Reports whether the selected sample passes:
   - **pass@1**: average pass rate (no selection)
   - **logprob**: does the top-by-logprob sample pass? (0 or 1)
   - **edit**: does the top-by-edit-similarity sample pass? (0 or 1)
   - **exec**: does the top-by-execution-similarity sample pass? (0 or 1)

### What to implement

- `evaluate_problem(codes, logprobs, test_list)`: orchestrate all four evaluation methods and return scores.

---

## Utility Module (`utils.py`)

This module provides infrastructure for prompt construction and safe code execution. It is adapted from [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and [OpenAI's human-eval](https://github.com/openai/human-eval).

Key functions (provided, no exercises):

- `make_prompt(example, n_tests)`: build a zero-shot prompt from an MBPP example
- `extract_code(completion)`: extract Python code from markdown code blocks in LLM responses
- `extract_func_calls(test_list)`: parse function calls from assert statements
- `execute_tests(codes, tests)`: run each code sample against test cases in sandboxed subprocesses (parallel)
- `execute_codes(codes, calls)`: run each code sample with specific function calls and collect outputs (for execution similarity)
- `unsafe_execute(program, conn, timeout)`: isolated execution with timeout, I/O redirection, and `reliability_guard` to disable dangerous system calls

---

## Running the Evaluation

### Prerequisites

You need [Ollama](https://ollama.com/) running locally with a code model. Pull the model first, then start the server:

```bash
ollama pull qwen2.5-coder:7b
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

For faster parallel inference, you can enable parallel requests and flash attention:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=8 OLLAMA_FLASH_ATTENTION=1 ollama serve
```

### Running

```bash
uv run python meta_generation/meta_generation.py
```

### Command-line options

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen2.5-coder:7b` | Ollama model to use |
| `--n` | `15` | Number of samples to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.95` | Top-p (nucleus) sampling |
| `--seed` | `1618` | Random seed for reproducibility |
| `--index` | `13` | MBPP test problem index |
| `--output` | `mbpp_eval_results.json` | Output file for results |

### Example output

```
Method       Score
--------------------
pass@1       0.6000
logprob      1.0000
edit         1.0000
exec         1.0000
```

This shows that while only 60% of individual samples pass, all three selection strategies successfully picked a correct solution. Save your result in the `mbpp_eval_results.json` output file.

---

## Summary

```
MBPP problem
    |
    v
Generate n samples          (parallel, via Ollama)
    |
    v
Rank by 4 strategies:
  1. pass@1                  (baseline: average pass rate)
  2. logprob                 (model confidence)
  3. edit similarity         (textual consensus, MBR)
  4. execution similarity    (behavioral consensus, MBR)
    |
    v
Pick top sample per strategy
    |
    v
Evaluate against test suite
```

