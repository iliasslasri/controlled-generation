# Part 3: Self-Correction [8 points]

Some tasks require **provably correct** code — code that is formally verified by an external tool. When the LLM's first attempt fails verification, can we use the error messages to iteratively fix it?

This part implements a **tree-search refinement pipeline** for generating formally verified Rust code using [Verus](https://github.com/verus-lang/verus), a verification tool for Rust. The LLM generates candidate implementations, Verus checks them, and error feedback drives iterative refinement until a fully verified program is found.

## Overview

| Module | Description |
|---|---|
| `self_correction.py` | Two-stage pipeline: parallel generation + tree-search refinement |
| `utils.py` | Verus integration, code validation, scoring, and error parsing |
| `examples/incr_list.rs` | Example Verus specification for the LLM to complete |

---

## Context: Formal Verification with Verus

Verus is a tool for proving that Rust programs satisfy formal specifications. A Verus program contains:
- **Preconditions** (`requires`): what must be true before the function runs
- **Postconditions** (`ensures`): what must be true after the function returns
- **Loop invariants** (`invariant`): what must hold at every iteration
- **Proof statements** (`assert`, proof blocks): hints to help the verifier

The LLM's task is to **complete the function body** (including any necessary proof statements and loop invariants) so that Verus can verify the postconditions hold.

### Example specification (`examples/incr_list.rs`)

```rust
fn incr_list(l: Vec<i32>) -> (result: Vec<i32>)
    requires
        forall|i: int| 0 <= i < l.len() ==> l[i] + 1 <= i32::MAX,
    ensures
        result.len() == l.len(),
        forall|i: int| 0 <= i < l.len() ==> #[trigger] result[i] == l[i] + 1,
{
    // LLM must complete this function body
}
```

The function must increment every element by 1 and prove that the result satisfies both postconditions.

---

## Stage 1: Parallel Initial Generation (`parallel_generator`)

### Idea

Generate a large number of candidate implementations in parallel. Most will fail verification, but some may partially verify (some assertions pass, some fail). These partially-correct programs seed Stage 2.

### Algorithm

1. We provide a prompt containing:
   - A system prompt with Verus expertise and worked examples
   - The specification (function signature + preconditions + postconditions)
   - Instructions to complete only the function body
2. Generate `first_stage` (default: 50) completions in parallel using a thread pool
3. For each generation, combine it with the specification and run Verus (`check`)
4. Classify results into three categories:
   - **Verified** (score = 1): fully verified, we are done
   - **Valid** (score = 0): partially verified (some assertions pass, some errors remain) — these proceed to Stage 2
   - **Invalid** (score = -1): rejected by validation checks or completely fails

### What to implement

- `first_verif(generations, input_program, verus_path)`: for each generation, parse it with `utils.parse_generation`, run Verus with `utils.check`, and classify the result as verified, valid, or invalid.

---

## Stage 2: Iterative Refinement with Tree Search

### Idea

Take the partially-correct programs from Stage 1, feed Verus error messages back to the LLM, and ask it to fix the code. Repeat for multiple iterations, focusing search effort on the most promising branches.

### The Refinement Loop

Each iteration performs five steps:

#### Step 1 — Expand

For each current conversation state, generate one refinement by calling the LLM with the full conversation history (system prompt + original program + error feedback from previous iterations). The LLM sees the accumulated context and produces a corrected version.

```
State = [system_prompt, user_msg_with_program_and_errors, assistant_response,
         user_msg_with_new_errors, assistant_response, ...]
```

#### Step 2 — Score (Value Function)

Evaluate each refined program with `evaluate_code`:

1. Run Verus on the code
2. Parse the output: count verified assertions and errors
3. Compute a score:
   - Base score = `num_verified / (num_verified + num_errors)`
   - Penalty: `-0.1` per parsed error, `-0.04` per note
   - Score = -1 if code is invalid (fails validation checks)
   - Score = 1 if fully verified (0 errors) and passes trivial-proof detection
4. Return `(score, error_message)`

#### Step 3 — Filter

Remove all states with score < 0 (invalid code). If all states are invalid, retry the iteration.

#### Step 4 — Deduplicate

Collapse states that produced identical code, keeping only the highest-scoring instance of each unique program. This avoids wasting search budget on redundant branches.

#### Step 5 — REBASE Sampling

**REBASE** (Regularized Best-first And Scoring Expansion) resamples the surviving states to maintain a fixed batch size while focusing on the most promising branches:

1. Convert scores to probabilities via softmax with temperature `tau`:
   ```
   P(i) = exp(score_i / tau) / sum_j exp(score_j / tau)
   ```
2. Sample `batch_size` states (with replacement) according to these probabilities
3. Append the error message to each sampled state for the next iteration

A low temperature (default: 0.1) makes the distribution sharp, concentrating effort on the highest-scoring states. Higher temperatures preserve more diversity.

#### Early Stopping

If any state achieves score >= 1 (fully verified), the loop exits immediately.

### What to implement

- `filter_valid(new_states, scored)`: keep only states with non-negative scores, returning `(state, score, error)` triples.
- `deduplicate(valid_triples, input_program, iteration_number)`: collapse states that produced identical code, keeping the best score.
- `rebase_select(valid_triples, batch_size, rebase_temperature)`: sample `batch_size` states proportional to the softmax of their scores.
- `iterative_refinement(valid, input_program, verus_path, ...)`: the main loop orchestrating expand, score, filter, deduplicate, and REBASE steps. Uses the provided helper functions `initialize`, `refinement_message`, `expand_to_batch`, and `async_refinement_generator`.

---

## Code Validation (`_validate_code`)

Before scoring, each program is checked against a set of rejection rules. Code is rejected (score = -1) if it:

- Contains `assume` statements (defeats the purpose of verification)
- Has a trivial postcondition (`ensures true`)
- Has more comments than actual code (>50% comments)
- Has no `ensures` clause
- Contains infinite loops (consecutive `{}`)
- Uses external verifiers (`#[verifier::external]` or `#[verifier::external_body]`)
- Has unreachable statements
- Uses `&mut` on non-`Vec` types

---

## Trivial Proof Detection (`check_pairs`, `check_pairs_loop`)

A subtle failure mode: the LLM may produce code that Verus verifies, but only because the preconditions make the postconditions trivially true (e.g., contradictory preconditions). To detect this:

- **`check_pairs`**: inject `assert(false)` at the start of the function body. If Verus still reports 0 errors, the preconditions are contradictory and the proof is trivial.
- **`check_pairs_loop`**: inject `assert(false)` before closing braces of loop bodies to test whether loop invariants are trivially satisfied.

Programs that pass these tests are rejected as trivially verified.

---

## Scoring Function (`evaluate_code`)

The scoring function serves as the value function for the tree search:

| Condition | Score |
|---|---|
| Fails validation checks | -1 |
| No verified assertions and no errors | -1 |
| More than 20 verified assertions (suspicious) | -1 |
| Trivially verified (`check_pairs` / `check_pairs_loop`) | -1 |
| Fully verified (0 errors, passes trivial checks) | 1 |
| Partial: some verified, some errors | `num_verified / (num_verified + num_errors)` minus penalties |

Error penalties are normalized: each error deducts `0.1 * (1 / total_assertions)` and each note deducts `0.04 * (1 / total_assertions)`.

---

## System Prompt

The system prompt provides the LLM with Verus expertise:

- General notes on Verus syntax (loop invariants, spec vs. exec code, type conversions)
- Three worked examples of increasing complexity:
  1. `has_close_elements` — nested loops with overflow-safe arithmetic
  2. `rolling_max` — recursive spec functions with sequence operations
  3. `intersperse` — quantified specs with proof blocks and `assert_seqs_equal!`

---

## Running the Pipeline

### Prerequisites

- [Verus](https://github.com/verus-lang/verus) installed and accessible, to [install Verus](https://github.com/verus-lang/verus/blob/main/INSTALL.md)
- [Ollama](https://ollama.ai/) running with a code model

```bash
ollama pull qwen2.5-coder:7b
```

### Running

```bash
uv run python self_correction/self_correction.py --verus-path /path/to/verus
```

### Command-line options

| Flag | Default | Description |
|---|---|---|
| `--input` | `examples/incr_list.rs` | Verus specification file |
| `--model` | `qwen2.5-coder:7b` | LLM model |
| `--batch-size` | `32` | Parallel generation batch size |
| `--first-stage` | `8` | Number of initial programs |
| `--max-iters` | `5` | Maximum refinement iterations |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--rebase-temperature` | `0.1` | Softmax temperature for REBASE |
| `--max-tokens` | `2048` | Max tokens per generation |
| `--base-url` | `http://localhost:11435/v1` | API endpoint |
| `--api-key` | `$OPENAI_API_KEY` or `ollama` | API key |
| `--verus-path` | `$VERUS_PATH` | Path to Verus binary |
| `--output` | `verified_program.rs` | Output file |
| `--valid-programs` | `None` | Path to pre-made valid programs (skips Stage 1) |

In my experiments, `qwen2.5-coder:7b` can produce verified programs but requires a very large number of attempts. The pipeline also supports external API calls: with `gpt-4o`, verified programs were found using the default parameters above.

### Bonus

Explore ways to improve the generation of verified programs. To earn extra credit, document your experiments in a markdown file: explain your approach and your results (failures can be as informative as successes!).

---

## Summary: The Full Pipeline

```
Verus specification (requires + ensures)
    |
    v
Stage 1: Generate N candidates in parallel
    |
    v
Classify: verified / valid / invalid
    |
    v
Stage 2: Iterative refinement loop
    |
    +---> Expand:  LLM generates fix from error feedback
    |       |
    |       v
    |     Score:   Verus verifies, compute value function
    |       |
    |       v
    |     Filter:  Remove invalid states (score < 0)
    |       |
    |       v
    |     REBASE:  Softmax resampling toward best branches
    |       |
    |       v
    +--- Append error message, repeat
    |
    v
Output: verified program (or best attempt)
```