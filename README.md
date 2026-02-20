# LLM Controlled Generation

This homework explores three complementary approaches to controlling the output of Large Language Models, progressing from token-level constraints to multi-sample selection to iterative self-correction with formal verification feedback.

## Overview

The homework is organized in three parts, each building on concepts from the previous one:

| Part | Folder | Topic | Key idea |
|---|---|---|---|
| 1 | [`structured_generation/`](structured_generation/) | Structured Generation | Constrain each token at generation time using regex-compiled FSMs |
| 2 | [`meta_generation/`](meta_generation/) | Meta-Generation | Generate multiple samples and select the best using ranking strategies |
| 3 | [`self_correction/`](self_correction/) | Self-Correction | Iteratively refine code using verifier error feedback and tree search |


📄 **Complete instructions:** [llm_controlled_generation.pdf](llm_controlled_generation.pdf)

### Part 1 — Structured Generation

LLMs generate text token by token, sampling from the full vocabulary at each step. Structured generation **guarantees** that every generated token is part of a valid output by constraining the sampling at each step. You will build a pipeline that compiles a regex into a token-level finite state machine, then use it to mask logits during generation with a pure-NumPy GPT-2 implementation.

**Modules:** naive vs. DFA masking, character-to-token FSM compilation, coalescence & forced-token optimization, end-to-end constrained generation with picoGPT.

### Part 2 — Meta-Generation

When a single LLM sample may or may not be correct, a natural idea is to generate **multiple samples** and **select the best one**. You will implement and compare four selection strategies — pass@1, best-of-n by log-probability, and two Minimum Bayes Risk (MBR) decoders using edit similarity and execution similarity — evaluated on the MBPP coding benchmark.

**Modules:** parallel sample generation, ranking by model confidence, MBR with Levenshtein distance, MBR with execution agreement.

### Part 3 — Self-Correction

Some tasks require **provably correct** code. You will build a tree-search refinement pipeline that generates formally verified Rust code using Verus. The LLM generates candidates, Verus checks them, and compiler error messages drive iterative refinement. A REBASE sampling strategy focuses search effort on the most promising branches.

**Modules:** parallel initial generation, Verus integration and scoring, code validation and trivial-proof detection, iterative refinement with tree search and REBASE.

## Setup

### Requirements

- Python >= 3.10
- [Ollama](https://ollama.ai/) (for Parts 2 and 3)
- [Verus](https://github.com/verus-lang/verus) (for Part 3 only)

### Installation

Clone the repo and then run:
```bash
cd llm_controlled-generation
uv sync
```

### Dependencies

Installed automatically via `uv sync`:

- `numpy`, `scipy` — numerical computation
- `regex`, `interegular` — regex parsing and DFA construction (Part 1)
- `transformers` — tokenizer and model weights (Part 1)
- `datasets` — MBPP benchmark (Part 2)
- `jellyfish` — Levenshtein distance (Part 2)
- `openai` — OpenAI-compatible API client for Ollama (Parts 2 and 3)

### Ollama setup (Parts 2 and 3)

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

## Running

### Part 1 — Structured Generation

Each module can be run independently:

```bash
uv run python structured_generation/deterministic_finite_automaton.py   # Naive vs. DFA masking benchmark
uv run python structured_generation/fsm_token.py                        # Character-to-token FSM demo
uv run python structured_generation/coalescence.py                      # Coalescence benchmark on GPT-2 vocab
uv run python structured_generation/picoGPT_generate.py                 # End-to-end constrained generation
```

### Part 2 — Meta-Generation

```bash
uv run python meta_generation/meta_generation.py   # Evaluate selection strategies on MBPP
```

### Part 3 — Self-Correction

```bash
uv run python self_correction/self_correction.py --verus-path /path/to/verus
```

## Tests

Run all required tests:

```bash
uv run pytest tests/test_structured_generation_required.py   # Part 1
uv run pytest tests/test_meta_generation_required.py          # Part 2
uv run pytest tests/test_self_correction_required.py          # Part 3
```

Or run all benchmarks and tests at once:

```bash
./test_and_submit.sh
```



## Project Structure

```
llm_controlled-generation/
├── structured_generation/
│   ├── structured_generation.md           # Part 1 description
│   ├── deterministic_finite_automaton.py  # Naive vs. DFA masking
│   ├── fsm_token.py                      # Character-level DFA to token-level FSM
│   ├── coalescence.py                    # Mask deduplication and forced tokens
│   ├── picoGPT_generate.py              # GPT-2 in NumPy + constrained generation
│   └── utils.py                          # GPT-2 weight loading
├── meta_generation/
│   ├── meta_generation.md                 # Part 2 description
│   ├── meta_generation.py               # Sample generation and ranking strategies
│   └── utils.py                          # Prompt construction and safe execution
├── self_correction/
│   ├── self_correction.md                 # Part 3 description
│   ├── self_correction.py                # Tree-search refinement pipeline
│   ├── utils.py                          # Verus integration, validation, scoring
│   └── examples/
│       └── incr_list.rs                  # Example Verus specification
├── tests/                                 # Test suite
├── pyproject.toml
└── README.md                              # This file
```

## License

Apache 2.0
