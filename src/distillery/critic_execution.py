"""Execution-based correctness gate for generated code items.

The rule-based Critic (``critic.py``) only checks STRUCTURE: a fenced code block,
a ``def test_`` function, at least two assertions, and one pytest feature. That
lets *structurally valid but logically broken* tests through — e.g. a test whose
expected value contradicts the code, or one that references an undefined name.

This module adds a CORRECTNESS gate for ``testcase_generation`` items. It grades
each item with a layered verdict (cheap -> expensive) and, crucially, does NOT
punish a test whose only "failure" is importing a module that doesn't exist in
this repo (those can't be judged fairly and are marked skipped, not rejected).

Verdicts:
  * ``OK``                    - passed every applicable check
  * ``SYNTAX_FAIL``           - the code block doesn't parse
  * ``PARAMETRIZE_MISMATCH``  - ``@parametrize`` argnames absent from the test signature
  * ``STATIC_FAIL``           - pyflakes reports an undefined name (missing import, bad scope)
  * ``EXEC_FAIL``             - the snippet is self-contained and ``pytest`` fails/errors
  * ``SKIPPED_EXTERNAL_DEPS`` - imports a non-stdlib/non-pytest module; can't run it here

Trust boundary: the EXEC layer runs LLM-generated code in a subprocess with a
timeout in a throwaway temp directory. It is OPT-IN (callers pass ``execute=True``)
and never runs code that pulls in third-party/unknown modules. Full OS-level
sandboxing is not attempted — this is intended for local, developer-driven runs.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from distillery.schemas import QAPair

# Dataset types this gate applies to.
CODE_DATASET_TYPES = ("testcase_generation", "testcase")

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class Verdict(str, Enum):
    """Outcome of grading a single generated code item."""

    OK = "ok"
    SYNTAX_FAIL = "syntax_fail"
    PARAMETRIZE_MISMATCH = "parametrize_mismatch"
    STATIC_FAIL = "static_fail"
    EXEC_FAIL = "exec_fail"
    SKIPPED_EXTERNAL_DEPS = "skipped_external_deps"


# Verdicts that mean "throw this item out".
_REJECT_VERDICTS = frozenset(
    {
        Verdict.SYNTAX_FAIL,
        Verdict.PARAMETRIZE_MISMATCH,
        Verdict.STATIC_FAIL,
        Verdict.EXEC_FAIL,
    }
)


@dataclass
class CodeVerdict:
    """A verdict plus human-readable reasons."""

    verdict: Verdict
    reasons: list[str] = field(default_factory=list)

    @property
    def is_reject(self) -> bool:
        return self.verdict in _REJECT_VERDICTS


def extract_python_code(answer: str) -> str:
    """Return all fenced python blocks in ``answer`` joined into one snippet."""
    blocks = _CODE_BLOCK_RE.findall(answer)
    return "\n\n".join(b.strip() for b in blocks).strip()


def _attr_chain(node: ast.AST) -> str:
    """Render a dotted attribute/name chain, e.g. ``pytest.mark.parametrize``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _parametrize_argnames(decorator: ast.AST) -> list[str] | None:
    """If ``decorator`` is a ``parametrize`` call, return its declared argnames."""
    if not isinstance(decorator, ast.Call):
        return None
    chain = _attr_chain(decorator.func)
    if not chain.endswith("parametrize"):
        return None
    if not decorator.args:
        return None
    first = decorator.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return [s.strip() for s in first.value.split(",") if s.strip()]
    if isinstance(first, (ast.List, ast.Tuple)):
        names = [
            e.value.strip()
            for e in first.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        return names or None
    return None


def _check_parametrize_signatures(tree: ast.AST) -> list[str]:
    """Flag tests whose ``@parametrize`` names aren't in the function signature."""
    issues: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sig_args = {a.arg for a in (*node.args.args, *node.args.kwonlyargs)}
        for decorator in node.decorator_list:
            names = _parametrize_argnames(decorator)
            if not names:
                continue
            missing = [n for n in names if n not in sig_args]
            if missing:
                issues.append(
                    f"test '{node.name}': parametrize name(s) {missing} not in its "
                    f"parameters {sorted(sig_args - {'self'})}"
                )
    return issues


def _check_undefined_names(tree: ast.AST) -> list[str]:
    """Use pyflakes to catch undefined names (missing imports / bad scope)."""
    try:
        from pyflakes import checker
        from pyflakes import messages as pf_messages
    except ImportError:
        return []  # pyflakes optional at import time; static layer degrades to no-op

    undefined_types = tuple(
        t
        for name in ("UndefinedName", "UndefinedLocal", "UndefinedExport")
        if (t := getattr(pf_messages, name, None)) is not None
    )
    if not undefined_types:
        return []

    result = checker.Checker(tree, filename="<generated-snippet>")
    issues: list[str] = []
    for message in result.messages:
        if isinstance(message, undefined_types):
            rendered = message.message % message.message_args
            issues.append(f"{rendered} (line {message.lineno})")
    return issues


def _external_imports(tree: ast.AST) -> set[str]:
    """Top-level imported modules that are neither stdlib nor pytest."""
    allowed = set(sys.stdlib_module_names) | {"pytest"}
    external: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed:
                    external.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import -> definitely external to a bare snippet
                external.add("." * node.level)
            elif node.module:
                top = node.module.split(".")[0]
                if top not in allowed:
                    external.add(top)
    return external


def _pytest_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def _summarize_pytest_output(output: str) -> str:
    """Pull the most useful line(s) out of pytest's output for the reason string."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    # Prefer the assertion/error line, else the short summary, else the tail.
    for ln in lines:
        if ln.startswith(("E   ", "E ")):
            return ln.lstrip("E ").strip()[:300]
    for ln in reversed(lines):
        if "failed" in ln or "error" in ln:
            return ln[:300]
    return (lines[-1] if lines else "pytest reported failure")[:300]


def _run_pytest(code: str, timeout: float) -> tuple[bool, str]:
    """Run ``code`` as a pytest file in a temp dir. Returns (passed, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "test_generated_snippet.py"
        test_file.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_file),
                    "-q",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"execution timed out after {timeout:.0f}s"
        return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def validate_code_item(
    item: QAPair,
    dataset_type: str,
    *,
    execute: bool = True,
    timeout: float = 20.0,
) -> CodeVerdict:
    """Grade one generated code item.

    Args:
        item: the Q&A pair to grade.
        dataset_type: only ``testcase_generation``/``testcase`` are graded; anything
            else returns ``OK`` untouched.
        execute: run self-contained snippets through pytest (the expensive layer).
        timeout: per-item subprocess timeout for the exec layer.
    """
    if dataset_type not in CODE_DATASET_TYPES:
        return CodeVerdict(Verdict.OK)

    code = extract_python_code(item.answer)
    if not code:
        # No code block at all -> that's the structural critic's job, not ours.
        return CodeVerdict(Verdict.OK)

    # 1. Syntax.
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CodeVerdict(
            Verdict.SYNTAX_FAIL,
            [f"syntax error: {exc.msg} (line {exc.lineno})"],
        )

    # 2. Parametrize/signature consistency (definitive structural defect).
    param_issues = _check_parametrize_signatures(tree)
    if param_issues:
        return CodeVerdict(Verdict.PARAMETRIZE_MISMATCH, param_issues)

    # 3. Static undefined-name analysis.
    static_issues = _check_undefined_names(tree)
    if static_issues:
        return CodeVerdict(Verdict.STATIC_FAIL, static_issues)

    # 4. Execution — only for self-contained snippets we can fairly run.
    if not execute:
        return CodeVerdict(Verdict.OK)

    external = _external_imports(tree)
    if external:
        return CodeVerdict(
            Verdict.SKIPPED_EXTERNAL_DEPS,
            [f"depends on external module(s): {', '.join(sorted(external))}"],
        )
    if not _pytest_available():
        return CodeVerdict(
            Verdict.SKIPPED_EXTERNAL_DEPS,
            ["pytest is not installed; skipped execution"],
        )

    passed, output = _run_pytest(code, timeout=timeout)
    if not passed:
        return CodeVerdict(Verdict.EXEC_FAIL, [_summarize_pytest_output(output)])

    return CodeVerdict(Verdict.OK)
