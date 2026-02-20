"""Prompt templates for the self-correction pipeline."""

PARALLEL_GENERATION_PROMPT = """\
Consider the following verus code:
```{specification}
```

The code contains the relevant spec functions and the preconditions (requires) and postconditions (ensures) for the main function. Your goal is to complete the function, by adding necessary procedure, along with proof statements (such as invariants, asserts, proof blocks etc) to prove the program. Only output the new program and not the entire code. You are not allowed to create new functions, however can use any functions already defined if within the scope. Remember to just output the completion without the function signature, requires and ensures. Only the body of the function is required. Remember to end in:
```rust
}} // End of function
}} // verus!
fn main() {{}}
```
"""

INITIALIZE_PROMPT = """\
Given a Verus program with function signature, preconditions, postconditions, and code, fix the errors present in the code. Your task is to provide only the corrected body of the function, without repeating the function signature, requires, ensures, or specs. Focus on fixing proof statements or adjusting the code to ensure correct compilation. Do not modify function signatures, requires clauses, ensures clauses, or specs.
```rust
<function_body>
}} // End of function
fn main() {{}}
}} // verus!
```

Below is the program::
```rust
{program}
```

The program has following error message:
```
{error_message}
````

Solution Format:
[Thoughts on Error Message]
[Thoughts on Error Resolution]
[Thoughts on Corner Cases, such as Overflow etc.]
```rust
[Function Body, with closing parantheses]
```
"""

INITIALIZE_PROMPT_EASY = """\
Given a Verus program with function signature, preconditions, postconditions, and code, fix the errors present in the code. Your task is to provide only the corrected body of the function, without repeating the function signature, requires, ensures, or specs. Focus on fixing proof statements or adjusting the code to ensure correct compilation. Do not modify function signatures, requires clauses, ensures clauses, or specs.
```rust
<function_body>
}} // End of function
fn main() {{}}
}} // verus!
```

IMPORTANT: In Verus, every loop MUST have an `invariant` clause. Without loop invariants, postconditions cannot be verified. Here is an example of a correct loop with invariants:
```rust
fn rolling_max(numbers: Vec<i32>) -> (result: Vec<i32>)
    ensures
        result.len() == numbers.len(),
        forall|i: int| 0 <= i < numbers.len() ==> result[i] == seq_max(numbers@.take(i + 1)),
{{
    let mut max_so_far = i32::MIN;
    let mut result = Vec::with_capacity(numbers.len());
    for pos in 0..numbers.len()
        invariant
            result.len() == pos,
            max_so_far == seq_max(numbers@.take(pos as int)),
            forall|i: int| 0 <= i < pos ==> result[i] == seq_max(numbers@.take(i + 1)),
    {{
        let number = numbers[pos];
        if number > max_so_far {{
            max_so_far = number;
        }}
        result.push(max_so_far);
        assert(numbers@.take((pos + 1) as int).drop_last() =~= numbers@.take(pos as int));
    }}
    result
}}
```
Notice how the invariant maintains the relationship between the loop index, the result length, and the postcondition property at each iteration.

Below is the program::
```rust
{program}
```

The program has following error message:
```
{error_message}
````

Solution Format:
[Thoughts on Error Message]
[Thoughts on Error Resolution]
[Thoughts on Corner Cases, such as Overflow etc.]
```rust
[Function Body, with closing parantheses]
```"""


REFINEMENT_PROMPT = """\
I got the following errors:
 ```{error_message}```
 Follow the previous format, and fix all the errors. Then give the complete updated code."""
