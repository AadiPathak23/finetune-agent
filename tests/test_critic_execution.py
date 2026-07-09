"""Tests for the execution-based correctness gate (critic_execution.py).

The snippets below are real outputs (or faithful reductions of them) from a live
Ollama run — each demonstrates a distinct failure mode the structural Critic
cannot catch.
"""

import pytest

from distillery.critic_execution import (
    Verdict,
    extract_python_code,
    validate_code_item,
)
from distillery.schemas import QAPair


def _item(code: str) -> QAPair:
    return QAPair(question="q", answer=f"```python\n{code}\n```")


# --- EXEC_FAIL: self-contained test whose expected value contradicts the code ---
PROCESS_TEXT = '''
import pytest

def process_text(text):
    if not text:
        return "Empty"
    elif len(text) > 1024:
        return "Too long"
    else:
        return f"Processed: {text[:5]}...{len(text)} characters"

class TestTextProcessing:
    @pytest.mark.parametrize("input,expected", [('', 'Empty'), ('a' * 1025, 'Too long')])
    def test_process_text(self, input, expected):
        assert process_text(input) == expected

    def test_normal_input(self):
        # BUG: text[:5] of 'Hello world' is 'Hello', not 'Hel'
        assert process_text('Hello world') == "Processed: Hel...11 characters"
'''

# --- STATIC_FAIL: uses datetime.* but never imports datetime ---
MISSING_IMPORT = '''
import pytest

class DateProcessor:
    def process_date(self, date_str):
        datetime.datetime.strptime(date_str, '%Y-%m-%d')

@pytest.fixture
def processor():
    return DateProcessor()

def test_invalid_date(processor):
    with pytest.raises(ValueError):
        processor.process_date('bad')
    assert True
'''

# --- PARAMETRIZE_MISMATCH: parametrize says status_code, signature says status_codes ---
PARAM_MISMATCH = '''
import pytest

class TestHttpClient:
    @pytest.mark.parametrize("status_code", [200, 404, 500])
    def test_status_codes(self, status_codes):
        assert status_codes in (200, 404, 500)
        assert True
'''

# --- SKIPPED_EXTERNAL_DEPS: imports a module that doesn't exist here ---
EXTERNAL_DEP = '''
import math
from factorial import calculate_factorial
import pytest

class TestFactorial:
    @pytest.mark.parametrize("input,expected", [(0, 1), (1, 1)])
    def test_calculate_factorial(self, input, expected):
        assert calculate_factorial(input) == expected
        assert isinstance(calculate_factorial(input), int)
'''

# --- OK: a correct, self-contained test ---
GOOD = '''
import pytest

def add(a, b):
    return a + b

class TestAdd:
    @pytest.mark.parametrize("a,b,expected", [(1, 2, 3), (0, 0, 0), (-1, 1, 0)])
    def test_add(self, a, b, expected):
        assert add(a, b) == expected
        assert isinstance(add(a, b), int)
'''


def test_exec_fail_on_self_contradicting_test():
    verdict = validate_code_item(_item(PROCESS_TEXT), "testcase_generation")
    assert verdict.verdict == Verdict.EXEC_FAIL
    assert verdict.is_reject


def test_static_fail_on_missing_import():
    verdict = validate_code_item(_item(MISSING_IMPORT), "testcase_generation")
    assert verdict.verdict == Verdict.STATIC_FAIL
    assert any("datetime" in r for r in verdict.reasons)
    assert verdict.is_reject


def test_parametrize_mismatch():
    verdict = validate_code_item(_item(PARAM_MISMATCH), "testcase_generation")
    assert verdict.verdict == Verdict.PARAMETRIZE_MISMATCH
    assert verdict.is_reject


def test_external_deps_are_skipped_not_rejected():
    verdict = validate_code_item(_item(EXTERNAL_DEP), "testcase_generation")
    assert verdict.verdict == Verdict.SKIPPED_EXTERNAL_DEPS
    assert not verdict.is_reject  # can't judge fairly -> must not reject


def test_good_test_passes():
    verdict = validate_code_item(_item(GOOD), "testcase_generation")
    assert verdict.verdict == Verdict.OK
    assert not verdict.is_reject


def test_non_code_dataset_is_untouched():
    verdict = validate_code_item(_item(PROCESS_TEXT), "bugfixing")
    assert verdict.verdict == Verdict.OK


def test_execute_false_skips_run():
    # With execute=False we stop before running pytest; the self-contradicting
    # test is statically clean, so it should come back OK (not EXEC_FAIL).
    verdict = validate_code_item(_item(PROCESS_TEXT), "testcase_generation", execute=False)
    assert verdict.verdict == Verdict.OK


def test_extract_python_code_joins_blocks():
    answer = "intro\n```python\nx = 1\n```\nmid\n```python\ny = 2\n```"
    code = extract_python_code(answer)
    assert "x = 1" in code and "y = 2" in code


def test_verification_coverage_rollup():
    """agent._summarize_verification aggregates gate verdicts into a coverage stat."""
    from distillery.agent import FinetuneAgent
    from distillery.critic_execution import CodeVerdict, Verdict
    from distillery.llm.mock import MockLLMClient

    agent = FinetuneAgent(llm_client=MockLLMClient())

    class _CriticStub:
        pass

    critic = _CriticStub()
    critic._code_verdicts = {
        "testcase_generation": {
            0: CodeVerdict(Verdict.OK),
            1: CodeVerdict(Verdict.EXEC_FAIL, ["assert failed"]),
            2: CodeVerdict(Verdict.SKIPPED_EXTERNAL_DEPS, ["external"]),
            3: CodeVerdict(Verdict.STATIC_FAIL, ["undefined name"]),
        }
    }
    agent._critic = critic
    debug = {"verification_coverage": {}}
    agent._summarize_verification(debug)

    s = debug["verification_coverage"]["testcase_generation"]
    assert s == {
        "graded": 4,
        "executed": 2,      # OK + EXEC_FAIL actually ran
        "passed": 1,        # OK
        "rejected": 2,      # EXEC_FAIL + STATIC_FAIL
        "skipped_external_deps": 1,
    }
