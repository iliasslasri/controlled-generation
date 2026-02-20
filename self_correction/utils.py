import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple


def extract_code(res: str, add_main: bool = True) -> str:
    """Extract code from a markdown code block in an LLM response."""
    start = res.find('```')
    if start == -1:
        return res.strip()
    end = res[start+3:].find('```')
    if end == -1:
        return res[start + res[start:].find('\n'):].strip()
    # pick line after first ticks, and upto second ticks
    res_to_save = res[start + res[start:].find('\n'):start+3+end].strip()

    if add_main and res_to_save.find('main()')==-1:
        res_to_save += '\n\nfn main() {}'
    return res_to_save


def _run_verus(verus_path: str, code: str, timeout_duration: int = 10) -> Tuple[str, str]:
    """Write code to a temp file, run verus, clean up, and return (stdout, stderr)."""
    fd, path = tempfile.mkstemp(suffix='.rs')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(code)
        try:
            result = subprocess.run(
                [verus_path, path, '--multiple-errors', '100'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_duration,
            )
            return result.stdout.decode('utf-8'), result.stderr.decode('utf-8')
        except subprocess.TimeoutExpired:
            return "", "Process timed out after {} seconds".format(timeout_duration)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# --- Shared validation logic

def _strip_whitespace(s: str) -> str:
    """Remove all whitespace characters from a string."""
    return s.replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '')


def _uncommented_code(code: str) -> str:
    """Return code with blank lines and single-line comments removed."""
    return '\n'.join([line for line in code.splitlines() if line.strip() and not line.strip().startswith("//")])


def _validate_code(code: str, error_output: str) -> Tuple[bool, Optional[str]]:
    """Run all shared rejection checks on code and compiler error output.

    Returns (True, None) if code passes all checks, or (False, reason) if rejected.
    """
    if code.count('assume') > 0:
        return False, 'Assume statements are not allowed'

    if _strip_whitespace(code).find('ensurestrue') != -1:
        return False, 'Cannot create trivial postcondition: ensures true'

    uncommented = _uncommented_code(code)
    non_blank_lines = [x for x in code.splitlines() if x.strip() != '']
    if len(uncommented.splitlines()) < (len(non_blank_lines) // 2):
        return False, 'Too many comments relative to code'

    if uncommented.find('ensures') == -1:
        return False, 'No ensures clause found'

    if _strip_whitespace(uncommented).count('{}') >= 2:
        return False, 'Infinite loops are not allowed'

    if code.find('#[verifier::external]') != -1:
        return False, 'External verifiers are not allowed'

    if code.find('#[verifier::external_body]') != -1:
        return False, 'External verifiers are not allowed'

    if error_output.find('warning: unreachable statement') != -1:
        return False, 'Unreachable statement found'

    if re.search(r'&mut(?!\s*Vec\b)', code):
        return False, 'Mutables for non-Vec type are not allowed'

    return True, None


# --- STAGE 1

def parse_generation(input_program: str, generation: str) -> str:
    """Combine an input program template with an LLM-generated function body."""
    ec = extract_code(generation)
    if ec.strip().startswith('{'):
        ec = ec.strip()[1:]
    code = input_program + ec.strip()
    return code


def check(code: str, verus_path: str) -> dict:
    """Validate generated code via Verus. Returns dict with 'verified' (-1/0/1) and 'extracted_code'."""
    output, err = _run_verus(verus_path, code)

    if err.find('the name `main` is defined multiple times')!=-1:
        # Replace only the last occurrence of 'fn main() {}'
        last_main_idx = code.rindex('fn main() {}')
        code = code[:last_main_idx] + code[last_main_idx + len('fn main() {}'):]
        output, err = _run_verus(verus_path, code)

    extracted_code = code

    valid, _reason = _validate_code(extracted_code, err)
    if not valid:
        return {
            'verified': -1,
            'extracted_code': extracted_code
        }

    num_verified = re.search(r'(\d+) verified', output)
    num_verified = int(num_verified.group(1)) if num_verified else 0

    num_errors = re.search(r'(\d+) errors', output)
    num_errors = int(num_errors.group(1)) if num_errors else 0

    # Reject trivial single-assertion programs
    if num_verified == 1 and num_errors == 0:
        return {
            'verified': -1,
            'extracted_code': extracted_code
        }

    if num_verified > 0 and num_errors == 0:
        return {
            'verified': 1,
            'extracted_code': extracted_code
        }

    if num_verified != 0 or num_errors != 0:
        return {
            'verified': 0,
            'extracted_code': extracted_code
        }

    # error
    return {
        'verified': -1,
        'extracted_code': extracted_code
    }


# --- Stage 2

@dataclass
class ErrorBlock:
    type: str  # 'error' or 'note'
    title: str
    message: str
    file: str
    start_line: int
    end_line: Optional[int] = None

def parse_error_message(error_message: str) -> List[ErrorBlock]:
    blocks = []
    lines = error_message.split('\n')

    block_pattern = re.compile(r'^(error|note): (.+)$')
    file_line_pattern = re.compile(r'^  --> (.+):(\d+):(\d+)$')
    message_start_pattern = re.compile(r'^   \|$')
    message_end_pattern = re.compile(r'^   = .+$')

    current_block = None
    message_lines = []
    in_message = False

    for line in lines:
        block_match = block_pattern.match(line)
        if block_match:
            if current_block:
                current_block.message = '\n'.join(message_lines).strip()
                blocks.append(current_block)

            current_block = ErrorBlock(
                type=block_match.group(1),
                title=block_match.group(2),
                message='',
                file='',
                start_line=0
            )
            message_lines = []
            in_message = False
            continue

        file_line_match = file_line_pattern.match(line)
        if file_line_match and current_block:
            current_block.file = file_line_match.group(1)
            current_block.start_line = int(file_line_match.group(2))
            continue

        if message_start_pattern.match(line):
            in_message = True
            continue

        if message_end_pattern.match(line):
            in_message = False
            continue

        if in_message:
            message_lines.append(line)

        end_line_match = re.match(r'^(\d+) \|', line)
        if end_line_match and current_block:
            current_block.end_line = int(end_line_match.group(1))

    if current_block:
        current_block.message = '\n'.join(message_lines).strip()
        blocks.append(current_block)

    return blocks


def strip_body(code: str) -> Tuple[str, str]:
    """Split code into (signature+preconditions, function body)."""
    idx = code[:code.find('fn main()')].rfind('fn ')
    non_body_code = code[:idx+code[idx:].find('\n{')+2].strip()
    completion = code[idx+code[idx:].find('\n{')+3:].strip()
    return non_body_code, completion

def check_pairs(verus_path: str, prog: str) -> bool:
    """Check if a program is trivially verified by injecting assert(false)."""
    idx = prog.rfind('fn ', 0, prog.rfind('fn '))
    main_idx = prog.find('fn main()')
    # Find first {, replace with "assert false"
    idx2 = prog.find('{', idx)
    prog_require = prog[:idx2] + '{\n\tassert (false);\n' + prog[idx2+1:]
    o, e = _run_verus(verus_path, prog_require)
    if '0 errors' in o:
        return True

    prog_ensure = prog[:idx2] +'{}'+prog[main_idx:]
    o, e = _run_verus(verus_path, prog_ensure)
    num_verified = re.search(r'(\d+) verified', o)
    num_verified = int(num_verified.group(1)) if num_verified else 0
    if '0 errors' in o and num_verified > 0:
        return True
    return False

def check_pairs_loop(verus_path: str, prog: str) -> bool:
    """Check if a loop-based program is trivially verified by injecting assert(false)."""
    prog = '\n'.join([x for x in prog.split('\n') if x!=''])
    idx = prog.rfind('fn ', 0, prog.rfind('fn '))
    main_idx = prog.find('fn main()')
    function_of_concern = prog[idx:main_idx]
    last_bracket = function_of_concern[:function_of_concern.rfind('}')].rfind('}')
    # Now go back 2 lines
    lines = function_of_concern[:last_bracket+1].split('\n')
    final_code = prog[:idx] + '\n'.join(lines[:-2] + ['assert(false);'] + lines[-2:]) + function_of_concern[last_bracket+1:] + prog[main_idx:]
    o, e = _run_verus(verus_path, final_code)
    if '0 errors' in o and '0 verified' not in o:
        return True

    last_bracket = function_of_concern.rfind('}')
    lines = function_of_concern[:last_bracket+1].split('\n')
    final_code = prog[:idx] + '\n'.join(lines[:-2] + ['assert(false);'] + lines[-2:]) + function_of_concern[last_bracket+1:] + prog[main_idx:]
    o, e = _run_verus(verus_path, final_code)
    if '0 errors' in o and '0 verified' not in o:
        return True
    return False


def evaluate_node(input_program: str, state: list, verus_path: str) -> Tuple[float, str]:
    """Evaluate a refinement tree node by converting it to code and scoring it."""
    code = node_to_code(input_program, state)
    return evaluate_code(code, verus_path)

def node_to_code(input_program: str, state: list) -> str:
    """Convert a conversation state (message history) to executable Verus code."""
    generation = state[-1]['content']
    ec = extract_code(generation)
    if ec.strip().startswith('{'):
        ec = ec.strip()[1:]
    code = strip_body(input_program)[0] + ec.strip()
    return code

def evaluate_code(code: str, verus_path: str) -> Tuple[float, str]:
    """Evaluate code correctness. Returns (score, error_message) where score is -1 (invalid), 0-1 (partial), or 1 (verified)."""
    output, error = _run_verus(verus_path, code)
    if error.find('this file contains an unclosed delimiter') != -1:
        code = code + '}'
        output, error = _run_verus(verus_path, code)

    valid, reason = _validate_code(code, error)
    if not valid:
        return -1, reason

    error_msg = error

    num_verified = re.search(r'(\d+) verified', output)
    num_verified = int(num_verified.group(1)) if num_verified else 0

    num_errors = re.search(r'(\d+) errors', output)
    num_errors = int(num_errors.group(1)) if num_errors else 0

    # Num_verified is simply -1 score
    if num_verified == 0 and num_errors == 0:
        return -1, error_msg
    score = num_verified/(num_verified+num_errors)

    normalized_score_for_one = 1/(num_verified+num_errors)

    if num_verified>20:
        # There is some issue
        return -1, error_msg

    if num_errors == 0:
        uncommented = _uncommented_code(code)
        if check_pairs(verus_path, uncommented):
            return -1, 'Trivially verified'
        if check_pairs_loop(verus_path, uncommented):
            return -1, 'Trivially verified'
        return 1, error_msg

    parsed_msgs = parse_error_message(error_msg)
    parsed_errors = [msg for msg in parsed_msgs if msg.type == 'error'][:-1]
    parsed_notes = [msg for msg in parsed_msgs if msg.type == 'note']

    # Every error is a -0.1
    score -= normalized_score_for_one*0.1*len(parsed_errors)
    # Every note is -0.05
    score -= normalized_score_for_one*0.04* len(parsed_notes)
    return score, error_msg


def system_prompt() -> str:
    """Return the system prompt with Verus expertise and examples."""
    return """You are a helpful assistant and an expert in formal verification, Verus, and Rust.

Notes on Verus:
- Remember that all loops in Verus require invariants.
- Variables such as `len` defined prior to the proof typically need an associated invariant:
    - let len: usize = l.len(); ... then in the invariant: len == l.len();
- Specification code and non-specification code differ. For instance:
    - Specification code: some_string@spec_index(i)
    - Non-specification code: some_string.get_char(i)
- Strings:
    - let len: usize = some_string.unicode_len();
- Type conversions:
    - to use a usize `i` as an index in spec code: some_array[i as int]
- Do not generate any helper functions or specs.

Here are some examples:
---
use vstd::math::abs;
use vstd::prelude::*;
use vstd::slice::*;

verus! {

/// Because Verus doesn't support floating point, we need to use integers instead.
fn has_close_elements(numbers: &[i64], threshold: i64) -> (result: bool)
    ensures
        result == exists|i: int, j: int|
            0 <= i < j < numbers@.len() && abs(numbers[i] - numbers[j]) < threshold,
{
    // If `threshold <= 0`, there can't be any elements closer than `threshold`, so we can
    // just return `false`.
    if threshold <= 0 {
        assert(forall|i: int, j: int|
            0 <= i < j < numbers@.len() ==> abs(numbers[i] - numbers[j]) >= 0 >= threshold);
        return false;
    }
    // Now that we know `threshold > 0`, we can safely compute `i64::MAX - threshold` without
    // risk of overflow. (Subtracting a negative threshold would overflow.)

    let max_minus_threshold: i64 = i64::MAX - threshold;
    let numbers_len: usize = numbers.len();
    for x in 0..numbers_len
        invariant
            max_minus_threshold == i64::MAX - threshold,
            numbers_len == numbers@.len(),
            forall|i: int, j: int|
                0 <= i < j < numbers@.len() && i < x ==> abs(numbers[i] - numbers[j]) >= threshold,
    {
        let numbers_x: i64 = *slice_index_get(numbers, x);  // Verus doesn't yet support `numbers[x]` in exec mode.
        for y in x + 1..numbers_len
            invariant
                max_minus_threshold == i64::MAX - threshold,
                numbers_len == numbers@.len(),
                x < numbers@.len(),
                numbers_x == numbers[x as int],
                forall|i: int, j: int|
                    0 <= i < j < numbers@.len() && i < x ==> abs(numbers[i] - numbers[j])
                        >= threshold,
                forall|j: int| x < j < y ==> abs(numbers_x - numbers[j]) >= threshold,
        {
            let numbers_y = *slice_index_get(numbers, y);  // Verus doesn't yet support `numbers[y]` in exec mode.
            if numbers_x > numbers_y {
                // We have to be careful to avoid overflow. For instance, we
                // can't just check whether `numbers_x - numbers_y < threshold`
                // because `numbers_x - numbers_y` might overflow an `i64`.
                if numbers_y > max_minus_threshold {
                    return true;
                }
                if numbers_x < numbers_y + threshold {
                    return true;
                }
            } else {
                if numbers_x > max_minus_threshold {
                    return true;
                }
                if numbers_y < numbers_x + threshold {
                    return true;
                }
            }
        }
    }
    false
}
fn main() {}
} // verus!

---
use vstd::prelude::*;

verus! {

spec fn seq_max(a: Seq<i32>) -> i32
    decreases a.len(),
{
    if a.len() == 0 {
        i32::MIN
    } else if a.last() > seq_max(a.drop_last()) {
        a.last()
    } else {
        seq_max(a.drop_last())
    }
}

fn rolling_max(numbers: Vec<i32>) -> (result: Vec<i32>)
    ensures
        result.len() == numbers.len(),  // result vector should have the same length as input
        forall|i: int| 0 <= i < numbers.len() ==> result[i] == seq_max(numbers@.take(i + 1)),
{
    let mut max_so_far = i32::MIN;
    let mut result = Vec::with_capacity(numbers.len());
    for pos in 0..numbers.len()
        invariant
            result.len() == pos,
            max_so_far == seq_max(numbers@.take(pos as int)),
            forall|i: int| 0 <= i < pos ==> result[i] == seq_max(numbers@.take(i + 1)),
    {
        let number = numbers[pos];
        if number > max_so_far {
            max_so_far = number;
        }
        result.push(max_so_far);
        assert(numbers@.take((pos + 1) as int).drop_last() =~= numbers@.take(pos as int));
    }
    result
}
fn main() {}
} // verus!

---
use vstd::assert_seqs_equal;
use vstd::prelude::*;

verus! {

/// Specification for inserting a number 'delimiter' between every two consecutive elements
/// of input sequence `numbers'
pub open spec fn intersperse_spec(numbers: Seq<u64>, delimiter: u64) -> Seq<u64>
    decreases numbers.len(),
{
    if numbers.len() <= 1 {
        numbers
    } else {
        intersperse_spec(numbers.drop_last(), delimiter) + seq![delimiter, numbers.last()]
    }
}

// We use these two functions to provide valid triggers for the quantifiers in intersperse_quantified
spec fn even(i: int) -> int {
    2 * i
}

spec fn odd(i: int) -> int {
    2 * i + 1
}

// This quantified formulation of intersperse is easier to reason about in the implementation's loop
spec fn intersperse_quantified(numbers: Seq<u64>, delimiter: u64, interspersed: Seq<u64>) -> bool {
    (if numbers.len() == 0 {
        interspersed.len() == 0
    } else {
        interspersed.len() == 2 * numbers.len() - 1
    }) && (forall|i: int| 0 <= i < numbers.len() ==> #[trigger] interspersed[even(i)] == numbers[i])
        && (forall|i: int|
        0 <= i < numbers.len() - 1 ==> #[trigger] interspersed[odd(i)] == delimiter)
}

proof fn intersperse_spec_len(numbers: Seq<u64>, delimiter: u64)
    ensures
        numbers.len() > 0 ==> intersperse_spec(numbers, delimiter).len() == 2 * numbers.len() - 1,
    decreases numbers.len(),
{
    if numbers.len() > 0 {
        intersperse_spec_len(numbers.drop_last(), delimiter);
    }
}

// Show that the two descriptions of intersperse are equivalent
proof fn intersperse_quantified_is_spec(numbers: Seq<u64>, delimiter: u64, interspersed: Seq<u64>)
    requires
        intersperse_quantified(numbers, delimiter, interspersed),
    ensures
        interspersed == intersperse_spec(numbers, delimiter),
    decreases numbers.len(),
{
    let is = intersperse_spec(numbers, delimiter);
    if numbers.len() == 0 {
    } else if numbers.len() == 1 {
        assert(interspersed.len() == 1);
        assert(interspersed[even(0)] == numbers[0]);
    } else {
        intersperse_quantified_is_spec(
            numbers.drop_last(),
            delimiter,
            interspersed.take(interspersed.len() - 2),
        );
        intersperse_spec_len(numbers, delimiter);
        assert_seqs_equal!(is == interspersed, i => {
            if i < is.len() - 2 {
            } else {
                if i % 2 == 0 {
                    assert(is[i] == numbers.last());
                    assert(interspersed[even(i/2)] == numbers[i / 2]);
                    assert(i / 2 == numbers.len() - 1);
                } else {
                    assert(is[i] == delimiter);
                    assert(interspersed[odd((i-1)/2)] == delimiter);
                }
            }
        });
    }
    assert(interspersed =~= intersperse_spec(numbers, delimiter));
}

/// Implementation of intersperse
pub fn intersperse(numbers: Vec<u64>, delimiter: u64) -> (result: Vec<u64>)
    ensures
        result@ == intersperse_spec(numbers@, delimiter),
{
    if numbers.len() <= 1 {
        numbers
    } else {
        let mut result = Vec::new();
        let mut index = 0;
        while index < numbers.len() - 1
            invariant
                numbers.len() > 1,
                0 <= index < numbers.len(),
                result.len() == 2 * index,
                forall|i: int| 0 <= i < index ==> #[trigger] result[even(i)] == numbers[i],
                forall|i: int| 0 <= i < index ==> #[trigger] result[odd(i)] == delimiter,
        {
            result.push(numbers[index]);
            result.push(delimiter);
            index += 1;
            //assert(numbers@.subrange(0, index as int).drop_last() =~= numbers@.subrange(0, index as int - 1));
        }
        result.push(numbers[numbers.len() - 1]);
        proof {
            intersperse_quantified_is_spec(numbers@, delimiter, result@);
        }
        result
    }
}
fn main() {}
} // verus!

---
"""
