"""Five example programs that are valid (structurally correct) but not verified by Verus.

Each program passes _validate_code (has ensures, no assume, proper structure)
and compiles, but has semantic issues that cause Verus verification errors.
These are the kind of programs that enter Stage 2 iterative refinement.
"""

# The input program spec (from examples/incr_list.rs)
INPUT_PROGRAM = """\
use vstd::prelude::*;

verus! {

// Return list with elements incremented by 1.
fn incr_list(l: Vec<i32>) -> (result: Vec<i32>)
    requires
        forall|i: int| 0 <= i < l.len() ==> l[i] + 1 <= i32::MAX,  // avoid overflow

    ensures
        result.len() == l.len(),
        forall|i: int| 0 <= i < l.len() ==> #[trigger] result[i] == l[i] + 1,
{
"""

VALID_NOT_VERIFIED = [
    # 1. Wrong invariant indexing: uses i-1 instead of j
    #    Verus cannot prove the postcondition because the invariant
    #    does not correctly relate result[j] to l[j].
    """\
    let mut result = Vec::with_capacity(l.len());
    for i in 0..l.len()
        invariant
            result.len() == i,
            forall|k: int| 0 <= k < result.len() ==> #[trigger] result[k - 1] == l[k - 1] + 1,
    {
        let incremented = l[i] + 1;
        result.push(incremented);
    }
    result
} // End of function
} // verus!
fn main() {}
""",

    # 2. Missing overflow precondition in loop invariant
    #    The invariant doesn't carry `l[i] + 1 <= i32::MAX` from the
    #    requires clause, so Verus cannot prove the addition is safe.
    """\
    let mut result = Vec::with_capacity(l.len());
    for i in 0..l.len()
        invariant
            result.len() == i,
            forall|j: int| 0 <= j < i ==> #[trigger] result[j] == l[j] + 1,
    {
        let x = l[i];
        assert(x + 1 <= i32::MAX);
        result.push(x + 1);
    }
    result
} // End of function
} // verus!
fn main() {}
""",

    # 3. Incorrect invariant: `l.len() > 0` fails when l is empty
    #    and `0 <= i < l.len()` does not hold at loop entry (i == 0).
    """\
    let mut result = Vec::new();
    for i in 0..l.len()
        invariant
            l.len() > 0,
            0 <= i < l.len(),
            result.len() == i,
            forall|j: int| 0 <= j < i ==> #[trigger] result[j] == l[j] + 1,
    {
        result.push(l[i] + 1);
    }
    result
} // End of function
} // verus!
fn main() {}
""",

    # 4. Invariant too weak: missing `result.len() == i`
    #    Without tracking that result grows with the loop index,
    #    Verus cannot prove `result.len() == l.len()` postcondition.
    """\
    let mut result = Vec::with_capacity(l.len());
    for i in 0..l.len()
        invariant
            forall|j: int| 0 <= j < i ==> #[trigger] result[j] == l[j] + 1,
    {
        result.push(l[i] + 1);
    }
    result
} // End of function
} // verus!
fn main() {}
""",

    # 5. Swapped quantifier variable: uses `k` in trigger but `j` in body
    #    The invariant claims `result[j] == l[k] + 1` which is not what
    #    the postcondition needs (result[i] == l[i] + 1).
    """\
    let mut result = Vec::with_capacity(l.len());
    for i in 0..l.len()
        invariant
            result.len() == i,
            forall|j: int| 0 <= j < i ==> #[trigger] result[j] == l[0] + 1,
    {
        result.push(l[i] + 1);
    }
    result
} // End of function
} // verus!
fn main() {}
""",
]
