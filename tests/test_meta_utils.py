"""Unit tests for meta_generation/utils.py."""
import sys
import os
import io
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'meta_generation'))
import utils as mg_utils


# --- make_prompt ---

class TestMakePrompt:
    def test_basic_prompt_structure(self):
        example = {
            "prompt": "Write a function to add two numbers.",
            "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        }
        prompt = mg_utils.make_prompt(example)
        assert "Write a function to add two numbers." in prompt
        assert "assert add(1, 2) == 3" in prompt
        assert "```python" in prompt
        assert "triple backticks" in prompt

    def test_uses_text_field_as_fallback(self):
        example = {
            "text": "Write a function to multiply.",
            "test_list": ["assert mul(2, 3) == 6"],
        }
        prompt = mg_utils.make_prompt(example)
        assert "Write a function to multiply." in prompt

    def test_n_tests_limits_included_tests(self):
        example = {
            "prompt": "Do something.",
            "test_list": ["assert f(1) == 1", "assert f(2) == 2", "assert f(3) == 3"],
        }
        prompt = mg_utils.make_prompt(example, n_tests=1)
        assert "assert f(1) == 1" in prompt
        assert "assert f(2) == 2" not in prompt

    def test_n_tests_all(self):
        example = {
            "prompt": "Do something.",
            "test_list": ["assert f(1) == 1", "assert f(2) == 2"],
        }
        prompt = mg_utils.make_prompt(example, n_tests=5)
        assert "assert f(1) == 1" in prompt
        assert "assert f(2) == 2" in prompt


# --- extract_code ---

class TestExtractCode:
    def test_extracts_from_markdown_block(self):
        text = "Here is code:\n```python\ndef foo():\n    return 1\n```\nDone."
        assert mg_utils.extract_code(text) == "def foo():\n    return 1"

    def test_extracts_longest_block(self):
        text = "```python\nx = 1\n```\n\n```python\ndef bar():\n    return 2\n```"
        result = mg_utils.extract_code(text)
        assert "def bar():" in result

    def test_no_block_returns_raw(self):
        text = "just plain text"
        assert mg_utils.extract_code(text) == "just plain text"

    def test_language_tag_ignored_in_content(self):
        text = "```rust\nfn main() {}\n```"
        result = mg_utils.extract_code(text)
        assert result == "fn main() {}"

    def test_empty_block(self):
        text = "```python\n\n```"
        result = mg_utils.extract_code(text)
        assert result == ""


# --- extract_func_calls ---

class TestExtractFuncCalls:
    def test_extracts_simple_call(self):
        tests = ["assert add(1, 2) == 3"]
        calls = mg_utils.extract_func_calls(tests)
        assert calls == ["add(1, 2)"]

    def test_extracts_multiple_calls(self):
        tests = [
            "assert add(1, 2) == 3",
            "assert mul(3, 4) == 12",
        ]
        calls = mg_utils.extract_func_calls(tests)
        assert len(calls) == 2
        assert "add(1, 2)" in calls
        assert "mul(3, 4)" in calls

    def test_no_match_returns_empty(self):
        tests = ["print('hello')"]
        calls = mg_utils.extract_func_calls(tests)
        assert calls == []

    def test_handles_nested_parens(self):
        tests = ["assert foo(bar(1)) == 5"]
        calls = mg_utils.extract_func_calls(tests)
        assert len(calls) == 1
        assert "foo(bar(1))" in calls[0]

    def test_underscore_in_func_name(self):
        tests = ["assert my_func(10) == 20"]
        calls = mg_utils.extract_func_calls(tests)
        assert calls == ["my_func(10)"]


# --- WriteOnlyStringIO ---

class TestWriteOnlyStringIO:
    def test_write_succeeds(self):
        s = mg_utils.WriteOnlyStringIO()
        s.write("hello")
        # getvalue still works (inherited from StringIO)
        assert s.getvalue() == "hello"

    def test_read_raises(self):
        s = mg_utils.WriteOnlyStringIO()
        s.write("data")
        with pytest.raises(OSError):
            s.read()

    def test_readline_raises(self):
        s = mg_utils.WriteOnlyStringIO()
        with pytest.raises(OSError):
            s.readline()

    def test_readlines_raises(self):
        s = mg_utils.WriteOnlyStringIO()
        with pytest.raises(OSError):
            s.readlines()

    def test_readable_returns_false(self):
        s = mg_utils.WriteOnlyStringIO()
        assert s.readable() is False


# --- TimeoutException ---

class TestTimeoutException:
    def test_is_exception(self):
        assert issubclass(mg_utils.TimeoutException, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(mg_utils.TimeoutException):
            raise mg_utils.TimeoutException("test timeout")


# --- chdir ---

class TestChdir:
    def test_changes_and_restores_directory(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            with mg_utils.chdir(tmpdir):
                assert os.getcwd() == os.path.realpath(tmpdir)
            assert os.getcwd() == original

    def test_dot_is_noop(self):
        original = os.getcwd()
        with mg_utils.chdir("."):
            assert os.getcwd() == original

    def test_restores_on_exception(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with mg_utils.chdir(tmpdir):
                    raise ValueError("boom")
            except ValueError:
                pass
            assert os.getcwd() == original


# --- redirect_stdin ---

class TestRedirectStdin:
    def test_redirects_stdin(self):
        stream = mg_utils.WriteOnlyStringIO()
        with mg_utils.redirect_stdin(stream):
            assert sys.stdin is stream


# --- execute_tests (subprocess-based) ---

class TestExecuteTests:
    def test_passing_code(self):
        codes = ["def add(a, b): return a + b"]
        tests = ["assert add(1, 2) == 3"]
        results = mg_utils.execute_tests(codes, tests, timeout=5)
        assert len(results) == 1
        assert results[0]["passed"] is True

    def test_failing_code(self):
        codes = ["def add(a, b): return a - b"]
        tests = ["assert add(1, 2) == 3"]
        results = mg_utils.execute_tests(codes, tests, timeout=5)
        assert len(results) == 1
        assert results[0]["passed"] is False

    def test_multiple_codes(self):
        codes = [
            "def f(x): return x + 1",
            "def f(x): return x * 2",
        ]
        tests = ["assert f(3) == 4"]
        results = mg_utils.execute_tests(codes, tests, timeout=5)
        assert len(results) == 2
        assert results[0]["passed"] is True   # 3 + 1 == 4
        assert results[1]["passed"] is False   # 3 * 2 == 6 != 4

    def test_timeout_code(self):
        codes = ["def f(): \n while True: pass\nf()"]
        tests = ["assert True"]
        results = mg_utils.execute_tests(codes, tests, timeout=1)
        assert len(results) == 1
        assert results[0]["passed"] is False


# --- execute_codes (subprocess-based) ---

class TestExecuteCodes:
    def test_returns_execution_results(self):
        codes = ["def add(a, b): return a + b"]
        calls = ["add(1, 2)", "add(10, 20)"]
        results = mg_utils.execute_codes(codes, calls, timeout=5)
        assert len(results) == 1
        assert results[0] == [3, 30]

    def test_multiple_codes(self):
        codes = [
            "def f(x): return x + 1",
            "def f(x): return x * 2",
        ]
        calls = ["f(5)"]
        results = mg_utils.execute_codes(codes, calls, timeout=5)
        assert len(results) == 2
        assert results[0] == [6]   # 5 + 1
        assert results[1] == [10]  # 5 * 2

    def test_error_returns_exception(self):
        codes = ["def f(x): raise ValueError('oops')"]
        calls = ["f(1)"]
        results = mg_utils.execute_codes(codes, calls, timeout=5)
        assert len(results) == 1
        assert isinstance(results[0], Exception)
