import argparse
import importlib.util
import logging
import os
import random
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai

import prompts
import utils

logger = logging.getLogger(__name__)

# --- LLM output logger ---
llm_logger = logging.getLogger("llm_outputs")
llm_logger.setLevel(logging.DEBUG)
_llm_handler = logging.FileHandler("llm_outputs.log", mode="w")
_llm_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
llm_logger.addHandler(_llm_handler)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _call_with_retry(fn: callable, max_retries: int = 3, backoff: float = 1.0) -> str:
    """Call fn() with retries and exponential backoff on API errors.

    Returns "" on final failure so the pipeline can continue with fewer results.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIStatusError,
        ) as e:
            if attempt < max_retries - 1:
                wait = backoff * (2**attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("LLM call failed after %d attempts: %s", max_retries, e)
                return ""


def parallel_generator(
    model: str,
    temperature: float,
    max_tokens: int,
    n: int,
    batch_size: int,
    specification: str,
    client: openai.Client,
) -> list[str]:
    """Generate multiple program completions in parallel.

    Tries the OpenAI `n` parameter first (efficient for GPT). If the API
    ignores `n` and returns only 1 result (e.g. Ollama), falls back to
    individual parallel calls.
    """
    messages = [
        {"role": "system", "content": utils.system_prompt()},
        {
            "role": "user",
            "content": prompts.PARALLEL_GENERATION_PROMPT.format(
                specification=specification
            ),
        },
    ]
    call_params = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        presence_penalty=0.3,
    )

    # Try using the n parameter (works with OpenAI GPT)
    def _call_batch():
        response = client.chat.completions.create(**call_params, n=n)
        results = [choice.message.content for choice in response.choices]
        for r in results:
            llm_logger.info("parallel_generator output:\n%s", r)
        return results

    results = _call_with_retry(_call_batch)
    if len(results) >= n:
        return [r for r in results if r]

    # API ignored n (e.g. Ollama) — fall back to parallel individual calls
    logger.info(
        "API returned %d/%d results; falling back to parallel calls", len(results), n
    )
    remaining = n - len(results)

    def _generate_one(_):
        def _call():
            response = client.chat.completions.create(**call_params)
            content = response.choices[0].message.content
            llm_logger.info("parallel_generator output:\n%s", content)
            return content

        return _call_with_retry(_call)

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        more = list(pool.map(_generate_one, range(remaining)))
    return [r for r in results + more if r]


def refinement_generator(
    state: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    client: openai.Client,
) -> str:
    """Generate a single refinement completion from a conversation state."""

    def _call():
        response = client.chat.completions.create(
            model=model,
            messages=state,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            presence_penalty=0.3,
        )
        content = response.choices[0].message.content
        llm_logger.info(
            "refinement_generator output (temp=%.2f):\n%s", temperature, content
        )
        return content

    return _call_with_retry(_call)


def async_refinement_generator(
    history: list,
    model: str,
    temperature: float,
    max_tokens: int,
    batch_size: int,
    client: openai.Client,
) -> list[str]:
    """Generate refinement completions for all conversation states in parallel."""
    n = len(history)
    # Spread temperatures across [temp-0.2, temp+0.5] for diversity
    temps = [max(0.1, temperature - 0.2 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

    def _generate_one(args):
        state, temp = args
        return refinement_generator(state, model, temp, max_tokens, client)

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        generations = list(pool.map(_generate_one, zip(history, temps)))
    generations = [x for x in generations if len(x) > 0]
    return generations


def initialize(
    programs: list[str],
    error_messages: list[str],
    prompt_template: str = prompts.INITIALIZE_PROMPT,
) -> list[list[dict]]:
    """Create initial message pairs (system prompt + instruction) for each program and its error."""
    messages = []
    for program, error_message in zip(programs, error_messages):
        message = {
            "role": "user",
            "content": prompt_template.format(
                program=program, error_message=error_message
            ),
        }
        messages.append([{"role": "system", "content": utils.system_prompt()}, message])
    return messages


def refinement_message(
    error_message: str,
    prompt_template: str = prompts.REFINEMENT_PROMPT,
) -> list[dict]:
    """Create a follow-up user message with error feedback for the next refinement iteration."""
    return [
        {
            "role": "user",
            "content": prompt_template.format(error_message=error_message),
        }
    ]


def expand_to_batch(
    programs: list[str], error_messages: list[str], n: int
) -> tuple[list[str], list[str]]:
    """Pad programs/errors to size n by random resampling with replacement."""
    k = len(programs)
    if k == 0:
        return [], []
    if k >= n:
        return programs, error_messages
    indices = list(range(k)) + random.choices(range(k), k=n - k)
    return [programs[i] for i in indices], [error_messages[i] for i in indices]


def filter_valid(
    new_states: list[list[dict]],
    scored: list[tuple[float, str]],
) -> list[tuple[list[dict], float, str]]:
    """Keep only states with non-negative scores, returning (state, score, error) triples.

    Args:
        new_states: list of conversation states.
        scored: list of (score, error_message) tuples, one per state.

    Returns:
        List of (state, score, error) triples where score >= 0.
    """
    # TODO: implement this function
    raise NotImplementedError


def deduplicate(
    valid_triples: list[tuple[list[dict], float, str]],
    input_program: str,
    iteration_number: int,
) -> list[tuple[list[dict], float, str]]:
    """Collapse states that produced identical code, keeping the best score.

    For each triple, extract the code with utils.node_to_code(input_program, state).
    When multiple triples produce the same code, keep only the one with the highest score.

    Log duplicates removed using:
        logger.info("Iteration %d: removed %d duplicate programs (%d unique)",
                     iteration_number + 1, n_dupes, n_unique)

    Returns:
        Deduplicated list of (state, score, error) triples.
    """
    # TODO: implement this function
    raise NotImplementedError


def rebase_select(
    valid_triples: list[tuple[list[dict], float, str]],
    batch_size: int,
    rebase_temperature: float,
) -> tuple[list[list[dict]], list[float], list[str]]:
    """Sample batch_size instances from valid_triples proportional to softmax of scores.

    Steps:
    1. Extract scores from valid_triples.
    2. Compute softmax probabilities: _softmax(scores / rebase_temperature).
    3. Sample batch_size indices using random.choices with these weights.
    4. Sort the sampled indices.
    5. Build return lists using deepcopy for states and error messages.

    Returns:
        (new_states, scores, error_messages): three lists of length batch_size.
    """
    # TODO: implement this function
    raise NotImplementedError


def iterative_refinement(
    valid: list[dict],
    input_program: str,
    verus_path: str,
    model: str,
    temperature: float,
    max_tokens: int,
    batch_size: int,
    client: openai.Client,
    max_iters: int,
    rebase_temperature: float,
) -> tuple[list[list[dict]], list[float]]:
    """Run the iterative refinement loop (Stage 2).

    Returns (history, scores) if a verified program is found, or (None, None) otherwise.

    Algorithm:
    1. Extract programs and evaluate them with utils.evaluate_code to get initial error messages.
    2. Pad to batch_size with expand_to_batch, then create initial history with initialize.
    3. For each iteration (up to max_iters):
       a. Expand: call async_refinement_generator to get generations, then build new_states
          by appending each generation as an assistant message to the corresponding history.
       b. Score: evaluate each new state with utils.evaluate_node(input_program, state, verus_path).
       c. Filter: use filter_valid to keep non-negative scores. If none survive, log a warning
          and continue to the next iteration.
       d. Deduplicate: use deduplicate to collapse identical code.
       e. REBASE: use rebase_select to sample batch_size states.
       f. Update history: append refinement_message(error) to each selected state.
       g. If any score >= 1, return (history, scores) immediately.
    4. If no iteration produced a verified program, return (None, None).

    Use the provided helper functions:
    - expand_to_batch(programs, error_messages, batch_size)
    - initialize(programs, error_messages)
    - async_refinement_generator(history=..., model=..., temperature=...,
                                  max_tokens=..., batch_size=..., client=...)
    - refinement_message(error)
    """
    # TODO: implement this function
    raise NotImplementedError


def display_rust_program_with_errors(program: str, errors: str):
    rust_code_md = f"```rust\n{program}\n```"
    error_message_md = f"**Error Message:**\n```\n{errors}\n```"
    return rust_code_md + "\n\n" + error_message_md


def first_verif(generations, input_program, verus_path):
    """Classify each generation as valid, verified, or invalid.

    For each generation:
    1. Parse it with utils.parse_generation(input_program, generation)
    2. Check it with utils.check(parsed, verus_path)
    3. The result dict has a "verified" key:
       - 0  → append to valid list
       - 1  → append to verified list
       - -1 → increment invalid_count

    Returns:
        (valid, verified, invalid_count): lists of result dicts and an int count.
    """
    # TODO: implement this function
    raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-correcting Verus code generation pipeline"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "examples", "incr_list.rs"),
        help="Path to .rs file containing the input program specification (default: examples/incr_list.rs)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="LLM model name (default: qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for parallel generation (default: 32)",
    )
    parser.add_argument(
        "--first-stage",
        type=int,
        default=8,
        help="Number of initial programs to generate (default: 8)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Max refinement iterations (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--rebase-temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for REBASE resampling (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per generation (default: 2048)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11435/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:11435/v1 for Ollama; set to None for OpenAI)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "ollama"),
        help="API key (default: $OPENAI_API_KEY, or 'ollama' for local)",
    )
    parser.add_argument(
        "--verus-path",
        default=os.environ.get("VERUS_PATH", "/Users/lelarge/verus/verus"),
        help="Path to verus binary (default: $VERUS_PATH)",
    )
    parser.add_argument(
        "--output",
        default="verified_program.rs",
        help="Output file for verified program (default: verified_program.rs)",
    )
    parser.add_argument(
        "--valid-programs",
        default=None,
        help="Path to a Python file exporting VALID_NOT_VERIFIED (list of code bodies) and INPUT_PROGRAM. "
        "When provided, Stage 1 is skipped and these programs are used directly for refinement. "
        "(default: None; e.g. examples/valid_not_verified.py)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configure root logger for structured console output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.input) as f:
        input_program = f.read()

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = openai.Client(**client_kwargs)

    logger.info(
        "Config: model=%s, batch_size=%d, first_stage=%d, temperature=%.2f",
        args.model,
        args.batch_size,
        args.first_stage,
        args.temperature,
    )

    if args.valid_programs:
        # --- Skip Stage 1: load pre-made valid programs ---
        valid_programs_path = args.valid_programs
        if not os.path.isabs(valid_programs_path):
            valid_programs_path = os.path.join(
                os.path.dirname(__file__), valid_programs_path
            )
        spec = importlib.util.spec_from_file_location(
            "valid_programs", valid_programs_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        valid = [
            {"extracted_code": mod.INPUT_PROGRAM + body}
            for body in mod.VALID_NOT_VERIFIED
        ]
        logger.info(
            "Skipping Stage 1: loaded %d valid programs from %s",
            len(valid),
            args.valid_programs,
        )
    else:
        # --- Stage 1: Initial generation ---
        logger.info("Stage 1: Generating %d initial programs...", args.first_stage)
        generations = parallel_generator(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.first_stage,
            batch_size=args.batch_size,
            specification=input_program,
            client=client,
        )

        valid, verified, invalid_count = first_verif(
            generations, input_program, args.verus_path
        )

        logger.info(
            "Stage 1 results: %d/%d valid, %d/%d verified, %d/%d invalid",
            len(valid),
            args.first_stage,
            len(verified),
            args.first_stage,
            invalid_count,
            args.first_stage,
        )

        if len(verified) > 0:
            verified_code = verified[0]["extracted_code"]
            with open(args.output, "w") as f:
                f.write(verified_code)
            logger.info(
                "Stage 1: found %d verified program(s), saved to %s",
                len(verified),
                args.output,
            )
            exit(0)

        if len(valid) == 0:
            logger.error("No valid programs generated in first stage, exiting.")
            exit(1)

    # --- Stage 2: Iterative refinement ---
    logger.info(
        "Stage 2: Starting iterative refinement (max %d iterations)...", args.max_iters
    )

    history, scores = iterative_refinement(
        valid=valid,
        input_program=input_program,
        verus_path=args.verus_path,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        client=client,
        max_iters=args.max_iters,
        rebase_temperature=args.rebase_temperature,
    )
    if history is None:
        exit(0)

    # --- Display results ---
    verified_index = np.argmax(scores)
    trajectory = history[verified_index]
    verified_code = utils.node_to_code(input_program, trajectory)

    with open(args.output, "w") as f:
        f.write(verified_code)
    logger.info("Verified program saved to %s", args.output)

    md = "## Successful trajectory\n\n"

    # Display each refinement step by re-evaluating intermediate programs
    for step, i in enumerate(range(2, len(trajectory), 2), start=1):
        program = utils.node_to_code(input_program, trajectory[: i + 1])
        _, errors = utils.evaluate_code(program, args.verus_path)
        md += "### Step %d\n\n" % step
        md += display_rust_program_with_errors(program, errors)
        if i + 2 < len(trajectory):
            md += "\n\n---------------\n\n"

    print(md)
