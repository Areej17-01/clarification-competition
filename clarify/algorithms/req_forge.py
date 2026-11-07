# SPDX-FileCopyrightText: 2026 Areej Mehboob <mehboobareej01@gmail.com>
#
# SPDX-License-Identifier: MIT

"""ReqForge.

One question, then code. A first model call writes the developer's complete
reading of the task (what the function receives, what it returns and the rule)
and names the single point the text leaves unsettled: a condition that is
missing, a contradiction, or a phrase with two readings. The question states
that reading concretely, with the call signature and one worked example, and
asks the user to confirm it or say what differs, naming that point as the one
most in doubt.
Several implementations are then sampled from the text and the answer, each
restating the complete requirement before the code; a set of generated inputs
is run against all of them in one sandbox call and the final program is the
largest group that agrees with the stated examples. The requirement rewrite
follows ClarifyGPT; the question shape follows requirements elicitation
practice of confirming an understanding before asking.

Team: Rubber Duckers
Team Members: Areej Mehboob, Taha Mehboob (TU Ilmenau)
Main Contact: mehboobareej01@gmail.com
"""

from __future__ import annotations

import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)

CODE_RULES = """
Requirements for your answer:
- Define a top-level function named exactly `{entry_point}` with the parameters the problem implies.
- Include every import you need. Do not include tests, example usage, print statements or input().
- Do not add input validation, assertions or raise statements unless the problem asks for an error
  on some input.
- Handle every valid input, whatever its size; never limit the solution to the inputs shown.
- Enclose the code in ```python and ```.
""".strip()


CODE_PROMPT_TEMPLATE = """
Implement the Python function `{entry_point}` that solves the following problem:

{prompt}

{rules}
""".strip()


REGEN_PROMPT_TEMPLATE = """
Implement the Python function `{entry_point}` that solves the following problem:

{prompt}

### Clarifications from the user
{clarifications}

Follow the user's answers exactly, even where they contradict your first reading of the
description or its examples. Keep every other requirement of the description.

Before the code, restate in one paragraph the complete requirement as it now stands: what the
function receives, what it returns and the rule that produces it, taking the user's answer as
authoritative, including any correction the user made to the understanding stated in the
question. Then implement exactly that.

{rules}
""".strip()


READING_PROMPT_TEMPLATE = """
A developer must implement the Python function `{entry_point}` from the problem below and may ask
the user exactly one clarifying question first. The user knows the complete original requirement.
The text below may have lost a condition or a definition, may contradict itself, or may use a
phrase that admits more than one reading.

### Problem
{prompt}

Work in three steps and reply with a JSON object only, no other text:
1. "reading": the complete requirement as the text most plausibly intends it, in one sentence:
   what `{entry_point}` receives, what it returns and the rule that produces it.
2. "gap": the single most critical thing a correct implementation needs that the text does not
   settle: a condition or definition that is missing (a sentence that stops short, a term that is
   never defined, an output whose form is never shown), a contradiction between two statements or
   between a statement and an example, or a phrase with two readings that give different results.
   Prefer a gap whose resolution changes the output on ordinary inputs over one that only affects
   edge cases.
3. "question": one message to the user. State the reading above as the developer's current
   understanding, concretely: how `{entry_point}` is called (each parameter and what it holds, in
   order), what it returns, the rule that produces it, and one worked example call with the
   result that understanding gives. Then ask the user to confirm that understanding or to state
   exactly what differs, naming the gap above as the point most in doubt. Describe behaviour only
   through inputs and outputs, never through code, algorithms, implementations or test cases;
   never touch style, performance, loops, recursion or complexity; and use no backticks, code
   fences, other code formatting or line breaks.
{"reading": "<one sentence>", "gap": "<one sentence>", "question": "<the question>"}
""".strip()


INPUTS_PROMPT_TEMPLATE = """
Write test inputs for the Python function `{entry_point}` described below.

### Problem
{prompt}

Return one ```python block that defines three lists:
- EXAMPLES: every input/output example stated in the problem, as `(args, expected)` pairs, with
  the expected values copied literally and with the same type (an unquoted boolean stays a
  boolean, a quoted value stays a string). Use [] if there are none.
- INPUTS: {num_inputs} typical inputs that exercise the main behaviour and are likely to separate
  different readings of the problem (ties, duplicates, ordering, boundaries of the main rule).
- EDGE_INPUTS: 4 edge cases: an empty or single-element input, zero or negative values, and
  inputs that probe formatting rules the problem leaves open, such as extra whitespace inside
  strings, mixed letter case, duplicates or reversed order.
Every input must be a valid, realistic input for this problem: same argument shape and types as
the signature and the examples, with data that fits the problem's domain. Do not invent inputs
the problem would never receive.
Every `args` is a tuple of positional arguments: `(<first>, <second>)` for two arguments,
`(<only>,)` with a trailing comma for one. If the problem gives no signature, use the most
natural argument shape for `{entry_point}` and keep it the same for every entry.
Use only Python literals. Do not call `{entry_point}` or any other function inside the lists.
""".strip()


HARNESS_TEMPLATE = r"""
import contextlib, copy, io, json, math, signal, time

ENTRY = __ENTRY__
CANDIDATES = __CANDIDATES__
SETUP = __SETUP__
DEADLINE = time.time() + __TIME_BUDGET__


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


_HAS_TIMER = hasattr(signal, "setitimer")
if _HAS_TIMER:
    signal.signal(signal.SIGALRM, _on_alarm)


def _arm(seconds):
    if _HAS_TIMER:
        signal.setitimer(signal.ITIMER_REAL, seconds)


def _key(x):
    if isinstance(x, bool):
        x = int(x)  # the tests compare with ==, so True and 1 (or 8 and 8.0) are the same answer
    if isinstance(x, float):
        if not math.isfinite(x):
            return repr(x)
        return repr(int(x)) if x.is_integer() else repr(round(x, 6))
    if type(x) in (list, tuple):
        inner = ", ".join(_key(v) for v in x)
        return "[" + inner + "]" if type(x) is list else "(" + inner + ")"
    if isinstance(x, (set, frozenset)):
        return "{" + ", ".join(sorted(_key(v) for v in x)) + "}"
    if isinstance(x, dict):
        return "{" + ", ".join(sorted(_key(k) + ": " + _key(v) for k, v in x.items())) + "}"
    return repr(x)


def _expected_key(e):
    if isinstance(e, str) and e.startswith("__EXC__:"):
        return "EXC:" + e[len("__EXC__:"):]
    return _key(e)[:300]


def _args(a):
    return a if isinstance(a, tuple) else (a,)


_ns = {}
try:
    exec(SETUP, _ns)
    EXAMPLES = [e for e in _ns.get("EXAMPLES", []) if isinstance(e, tuple) and len(e) == 2]
    INPUTS = list(_ns.get("INPUTS", []))
    EDGE_INPUTS = list(_ns.get("EDGE_INPUTS", []))
except BaseException:
    EXAMPLES, INPUTS, EDGE_INPUTS = [], [], []

CALLS = [_args(a) for a, _ in EXAMPLES] + [_args(a) for a in INPUTS] + [_args(a) for a in EDGE_INPUTS]
print("@@INPUTS " + json.dumps([
    (ENTRY + "(" + ", ".join(repr(v) for v in a) + ")")[:200] for a in CALLS
]), flush=True)
print("@@EXPECTED " + json.dumps([_expected_key(e) for _, e in EXAMPLES]), flush=True)

for index, source in enumerate(CANDIDATES):
    ns = {"__name__": "candidate"}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _arm(2.0)
            exec(source, ns)
            _arm(0)
        fn = ns[ENTRY]
    except BaseException as e:
        _arm(0)
        print("@@ROW " + json.dumps({"i": index, "load": type(e).__name__}), flush=True)
        continue

    outs = []
    for a in CALLS:
        if time.time() > DEADLINE:
            outs.append("SKIPPED")
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _arm(1.0)
                result = fn(*copy.deepcopy(a))
                _arm(0)
            outs.append(_key(result)[:300])
        except _Timeout:
            outs.append("TIMEOUT")
        except BaseException as e:
            _arm(0)
            outs.append("EXC:" + type(e).__name__)
    print("@@ROW " + json.dumps({"i": index, "out": outs}), flush=True)
"""


USELESS_ANSWERS = ("irrelevant question.", "cannot answer the given question.")


def _strip_reasoning(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL)


def _code_blocks(response: str) -> list[str]:
    response = _strip_reasoning(response)
    blocks = re.findall(r"```python\s*\n(.*?)```", response, flags=re.DOTALL)
    return blocks or re.findall(r"```[a-zA-Z0-9]*\s*\n(.*?)```", response, flags=re.DOTALL)


def _extract_code(response: str, entry_point: str) -> str | None:
    blocks = _code_blocks(response)
    if not blocks:
        return None
    defining = [b for b in blocks if f"def {entry_point}" in b]
    return (defining or blocks)[-1]


def _extract_setup(response: str) -> str | None:
    blocks = _code_blocks(response)
    return "\n".join(blocks) if blocks else None


def _is_script_noise(node: ast.stmt) -> bool:
    if isinstance(node, ast.If):
        return "__name__" in ast.unparse(node.test)
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return isinstance(node.value.func, ast.Name)
    return False


def _finalize(code: str, entry_point: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    tree.body = [node for node in tree.body if not _is_script_noise(node)]
    functions = [
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    code = ast.unparse(tree)
    if entry_point not in functions and entry_point not in assigned and len(functions) == 1:
        code += f"\n\n{entry_point} = {functions[0]}\n"
    return code


def _build_harness(entry_point: str, candidates: list[str], setup: str, time_budget: float) -> str:
    return (
        HARNESS_TEMPLATE.replace("__ENTRY__", repr(entry_point))
        .replace("__CANDIDATES__", repr(candidates))
        .replace("__SETUP__", repr(setup))
        .replace("__TIME_BUDGET__", repr(time_budget))
    )


def _is_failure(output: str) -> bool:
    return output.startswith(("EXC:", "TIMEOUT", "SKIPPED"))


def _parse_json(response: str) -> dict[str, Any] | None:
    text = _strip_reasoning(response)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    text = text[start : end + 1].replace(", ...", "").replace("...", "")
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class _Execution:
    def __init__(
        self,
        calls: list[str],
        expected: list[str],
        rows: list[list[str] | None],
        prefer_examples: bool,
    ):
        self.calls = calls
        self.expected = expected
        self.rows = rows
        self.loaded = sum(row is not None for row in rows)

        groups: dict[tuple[str, ...], list[int]] = {}
        for index, outputs in enumerate(rows):
            if outputs is not None:
                groups.setdefault(tuple(outputs), []).append(index)

        def rank(item):
            outputs, members = item
            failures = sum(_is_failure(o) for o in outputs)
            all_failed = bool(outputs) and failures == len(outputs)
            passes = self.example_passes(outputs)
            if prefer_examples:
                return (not all_failed, passes, len(members), -failures)
            return (not all_failed, len(members), passes, -failures)

        self.clusters = sorted(groups.items(), key=rank, reverse=True)

    def example_passes(self, outputs: tuple[str, ...] | list[str]) -> int:
        return sum(o == e for o, e in zip(outputs, self.expected, strict=False))

    def best(self) -> int | None:
        return self.clusters[0][1][0] if self.clusters else None


class ReqForge(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "num_samples": 4,  # implementations sampled after the answer
        "num_inputs": 8,  # test inputs generated on top of the stated examples
        "parallel_calls": 8,  # LLM calls in flight at once
        "llm_attempts": 3,  # retries per LLM call
        "exec_time_budget": 20.0,  # seconds per sandbox run
    }

    def complete(self, env: ClarificationEnvironment, prompt: str) -> str | None:
        delay = 1.0
        for attempt in range(self.config["llm_attempts"]):
            try:
                return _strip_reasoning(env.llm(prompt))
            except LimitsExceededException:
                return None
            except Exception:
                if attempt + 1 < self.config["llm_attempts"]:
                    time.sleep(delay)
                    delay *= 2
        return None

    def complete_many(self, env: ClarificationEnvironment, prompts: list[str]) -> list[str | None]:
        workers = max(1, min(self.config["parallel_calls"], len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda p: self.complete(env, p), prompts))

    def code_prompt(self, problem: dict[str, Any], clarifications: list[tuple[str, str]]) -> str:
        rules = CODE_RULES.replace("{entry_point}", problem["entry_point"])
        template = REGEN_PROMPT_TEMPLATE if clarifications else CODE_PROMPT_TEMPLATE
        prompt = (
            template.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{rules}", rules)
        )
        if clarifications:
            block = "\n".join(f"Question: {q}\nAnswer: {a}" for q, a in clarifications)
            prompt = prompt.replace("{clarifications}", block)
        return prompt

    def parse_candidates(self, responses: list[str | None], entry_point: str) -> list[str]:
        candidates = []
        for response in responses:
            code = _extract_code(response, entry_point) if response else None
            if code:
                candidates.append(_finalize(code, entry_point))
        return candidates

    def sample(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        count: int,
        extra_prompts: list[str] | None = None,
    ) -> tuple[list[str | None], list[str]]:
        extra = list(extra_prompts or [])
        prompt = self.code_prompt(problem, clarifications)
        responses = self.complete_many(env, extra + [prompt] * count)
        candidates = self.parse_candidates(responses[len(extra) :], problem["entry_point"])
        if not candidates:
            responses = responses[: len(extra)] + self.complete_many(env, [prompt] * count)
            candidates = self.parse_candidates(responses[len(extra) :], problem["entry_point"])
        return responses[: len(extra)], candidates

    def execute(
        self,
        env: ClarificationEnvironment,
        entry_point: str,
        candidates: list[str],
        setup: str | None,
        prefer_examples: bool,
    ) -> _Execution | None:
        if not candidates or not setup:
            return None

        script = _build_harness(entry_point, candidates, setup, self.config["exec_time_budget"])
        try:
            output = env.exec_code(script)
        except Exception:
            return None

        calls, expected, rows = None, [], [None] * len(candidates)
        for line in output.splitlines():
            try:
                if line.startswith("@@INPUTS "):
                    calls = json.loads(line[len("@@INPUTS ") :])
                elif line.startswith("@@EXPECTED "):
                    expected = json.loads(line[len("@@EXPECTED ") :])
                elif line.startswith("@@ROW "):
                    row = json.loads(line[len("@@ROW ") :])
                    if "out" in row:
                        rows[row["i"]] = row["out"]
            except (ValueError, KeyError, IndexError, TypeError):
                continue

        if not calls:
            return None
        return _Execution(calls, expected, rows, prefer_examples)

    @staticmethod
    def fenced(code: str) -> str:
        return f"```python\n{code}\n```"

    def reading_question(
        self, env: ClarificationEnvironment, problem: dict[str, Any]
    ) -> str | None:
        prompt = READING_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
            "{entry_point}", problem["entry_point"]
        )
        for _ in range(2):
            data = _parse_json(self.complete(env, prompt) or "")
            question = data.get("question") if data else None
            if isinstance(question, str) and question.strip():
                return " ".join(question.replace("`", "").split())[:600]
        return None

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem["entry_point"]
        clarifications: list[tuple[str, str]] = []
        if env.can_ask():
            question = self.reading_question(env, problem)
            if question:
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    answer = ""
                if answer.strip() and answer.strip().lower() not in USELESS_ANSWERS:
                    clarifications.append((question, answer))

        inputs_prompt = (
            INPUTS_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry_point)
            .replace("{num_inputs}", str(self.config["num_inputs"]))
        )
        extra, candidates = self.sample(
            env, problem, clarifications, self.config["num_samples"], extra_prompts=[inputs_prompt]
        )
        if not candidates:
            return self.fenced(
                f"def {entry_point}(*args, **kwargs):\n    raise NotImplementedError"
            )
        setup = _extract_setup(extra[0]) if extra and extra[0] else None
        execution = self.execute(env, entry_point, candidates, setup, prefer_examples=True)
        best = (
            candidates[execution.best()]
            if execution and execution.best() is not None
            else candidates[0]
        )
        return self.fenced(best)
