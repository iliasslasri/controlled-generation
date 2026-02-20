"""Unit tests for self_correction/utils.py parsing and validation functions."""
import sys
import os
import pytest

# Add self_correction to the path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'self_correction'))
import utils


# --- extract_code ---

class TestExtractCode:
    def test_extracts_from_markdown_block(self):
        text = "Here is code:\n```rust\nfn foo() {}\n```\nDone."
        assert utils.extract_code(text) == "fn foo() {}\n\nfn main() {}"

    def test_no_code_block_returns_stripped(self):
        text = "  just plain text  "
        assert utils.extract_code(text) == "just plain text"

    def test_unclosed_code_block(self):
        text = "```rust\nfn bar() {}\n"
        result = utils.extract_code(text)
        assert "fn bar() {}" in result

    def test_add_main_false(self):
        text = "```\nlet x = 1;\n```"
        result = utils.extract_code(text, add_main=False)
        assert "fn main()" not in result

    def test_does_not_duplicate_main(self):
        text = "```\nfn main() {}\n```"
        result = utils.extract_code(text)
        assert result.count("main()") == 1


# --- _validate_code ---

class TestValidateCode:
    def test_rejects_assume(self):
        valid, reason = utils._validate_code("assume(x > 0);", "")
        assert not valid
        assert "Assume" in reason

    def test_rejects_ensures_true(self):
        valid, reason = utils._validate_code("ensures true", "")
        assert not valid
        assert "ensures true" in reason.lower()

    def test_rejects_ensures_true_with_whitespace(self):
        valid, reason = utils._validate_code("ensures   true", "")
        assert not valid

    def test_rejects_missing_ensures(self):
        code = "fn foo() {\n  let x = 1;\n}"
        valid, reason = utils._validate_code(code, "")
        assert not valid
        assert "ensures" in reason.lower()

    def test_rejects_external_verifier(self):
        valid, reason = utils._validate_code("#[verifier::external]\nensures x > 0", "")
        assert not valid
        assert "External" in reason

    def test_rejects_external_body(self):
        valid, reason = utils._validate_code("#[verifier::external_body]\nensures x > 0", "")
        assert not valid
        assert "External" in reason

    def test_rejects_unreachable(self):
        valid, reason = utils._validate_code("ensures x > 0", "warning: unreachable statement")
        assert not valid
        assert "Unreachable" in reason

    def test_rejects_mut_non_vec(self):
        valid, reason = utils._validate_code("&mut String\nensures x > 0", "")
        assert not valid
        assert "Mutables" in reason

    def test_allows_mut_vec(self):
        valid, reason = utils._validate_code("&mut Vec<i32>\nensures x > 0", "")
        assert valid
        assert reason is None

    def test_rejects_too_many_comments(self):
        code = "// comment1\n// comment2\n// comment3\nensures x > 0"
        valid, reason = utils._validate_code(code, "")
        assert not valid
        assert "comments" in reason.lower()

    def test_rejects_infinite_loops(self):
        code = "ensures x > 0\nloop {} while {}"
        valid, reason = utils._validate_code(code, "")
        assert not valid
        assert "loop" in reason.lower()

    def test_passes_valid_code(self):
        code = "fn foo()\n  ensures\n    result > 0\n{\n  let x = 1;\n  x\n}"
        valid, reason = utils._validate_code(code, "")
        assert valid
        assert reason is None


# --- parse_generation ---

class TestParseGeneration:
    def test_combines_input_and_generation(self):
        inp = "fn foo() {\n    "
        gen = "```rust\nlet x = 1;\n```"
        result = utils.parse_generation(inp, gen)
        assert result.startswith("fn foo() {\n    ")
        assert "let x = 1;" in result

    def test_strips_leading_brace(self):
        inp = "fn foo() {\n    "
        gen = "```rust\n{ let x = 1;\n```"
        result = utils.parse_generation(inp, gen)
        assert not result.endswith("{ let x = 1;")
        assert "let x = 1;" in result


# --- parse_error_message ---

class TestParseErrorMessage:
    def test_parses_single_error(self):
        msg = "error: some error title\n  --> test.rs:10:5\n   |\n10 | let x = 1;\n   |\n"
        blocks = utils.parse_error_message(msg)
        assert len(blocks) == 1
        assert blocks[0].type == "error"
        assert blocks[0].title == "some error title"
        assert blocks[0].file == "test.rs"
        assert blocks[0].start_line == 10

    def test_parses_multiple_blocks(self):
        msg = (
            "error: first error\n  --> a.rs:1:1\n   |\n"
            "note: a note\n  --> a.rs:2:1\n   |\n"
        )
        blocks = utils.parse_error_message(msg)
        assert len(blocks) == 2
        assert blocks[0].type == "error"
        assert blocks[1].type == "note"

    def test_empty_input(self):
        assert utils.parse_error_message("") == []


# --- strip_body ---

class TestStripBody:
    def test_splits_signature_and_body(self):
        code = "use vstd::prelude::*;\nfn foo()\n  ensures x > 0\n{\n  let x = 1;\n}\nfn main() {}"
        sig, body = utils.strip_body(code)
        assert "fn foo()" in sig
        assert "ensures" in sig
        assert "let x = 1;" in body


# --- _strip_whitespace ---

class TestStripWhitespace:
    def test_removes_all_whitespace(self):
        assert utils._strip_whitespace("  a b\tc\nd\r") == "abcd"


# --- _uncommented_code ---

class TestUncommentedCode:
    def test_removes_comments_and_blanks(self):
        code = "// comment\nlet x = 1;\n\n// another\nlet y = 2;"
        result = utils._uncommented_code(code)
        assert "// comment" not in result
        assert "let x = 1;" in result
        assert "let y = 2;" in result

    def test_preserves_code_lines(self):
        code = "let a = 1;\nlet b = 2;"
        result = utils._uncommented_code(code)
        assert result == code

    def test_removes_blank_lines(self):
        code = "let a = 1;\n\n\nlet b = 2;"
        result = utils._uncommented_code(code)
        assert "\n\n" not in result


# --- ErrorBlock ---

class TestErrorBlock:
    def test_fields(self):
        block = utils.ErrorBlock(
            type="error", title="test error", message="details",
            file="test.rs", start_line=5, end_line=10
        )
        assert block.type == "error"
        assert block.title == "test error"
        assert block.message == "details"
        assert block.file == "test.rs"
        assert block.start_line == 5
        assert block.end_line == 10

    def test_end_line_optional(self):
        block = utils.ErrorBlock(
            type="note", title="a note", message="",
            file="a.rs", start_line=1
        )
        assert block.end_line is None


# --- node_to_code ---

class TestNodeToCode:
    def test_extracts_last_message_code(self):
        input_program = "use vstd::prelude::*;\nfn foo()\n  ensures x > 0\n{\n  placeholder\n}\nfn main() {}"
        state = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix this code."},
            {"role": "assistant", "content": "```rust\n  let x = 1;\n  x\n}\nfn main() {}\n```"},
        ]
        result = utils.node_to_code(input_program, state)
        assert "fn foo()" in result
        assert "ensures" in result
        assert "let x = 1;" in result

    def test_strips_leading_brace(self):
        input_program = "use vstd::prelude::*;\nfn bar()\n  ensures r > 0\n{\n  old_body\n}\nfn main() {}"
        state = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "```rust\n{ let y = 2;\n}\nfn main() {}\n```"},
        ]
        result = utils.node_to_code(input_program, state)
        assert "let y = 2;" in result
        # Should not start body with a bare '{'
        sig_part = result.split("let y = 2;")[0]
        assert not sig_part.rstrip().endswith("{{")


# --- system_prompt ---

class TestSystemPrompt:
    def test_returns_nonempty_string(self):
        prompt = utils.system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_verus_keywords(self):
        prompt = utils.system_prompt()
        assert "Verus" in prompt
        assert "invariant" in prompt.lower()
        assert "ensures" in prompt

    def test_contains_examples(self):
        prompt = utils.system_prompt()
        assert "fn main()" in prompt
        assert "verus!" in prompt
