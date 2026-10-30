# SPDX-FileCopyrightText: 2026 Areej Mehboob <mehboobareej01@gmail.com>
#
# SPDX-License-Identifier: MIT

"""ReqForge.

Execution-guided clarification. We sample several implementations plus a set
of test inputs, run everything in one sandbox call and group the programs by
behaviour. The input where the groups disagree most, plus a short audit of the
prompt for contradictions, feeds a drafting step that writes a handful of
candidate questions (return format, the observed split, an edge case, a vague
rule). A reviewer applying the user's own quality criteria picks one; drafts
the prompt already answers are dropped. The answer is converted into the
expected output on the witness input, fresh programs are sampled with the Q/A
in context, and the final program is the largest group that honours the
answer and the stated examples. If the user cannot answer, we fall back to the
pre-question majority. Ideas borrowed from ClarifyGPT, SpecFix, TiCoder,
CodeT and Self-Debugging.

Team: Rubber Duckers
Team Members: Areej Mehboob, Taha Mehboob (TU Ilmenau)
Main Contact: mehboobareej01@gmail.com
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from collections import Counter
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

{rules}
""".strip()

REPAIR_PROMPT_TEMPLATE = """
Implement the Python function `{entry_point}` that solves the following problem:

{prompt}
{clarifications}
A previous implementation returns the wrong value on the problem's own examples:

{failures}

The expected values are correct. Work out which reading of the problem produces exactly these
values (for instance which boundaries, orderings or counting rules it implies) and
implement that reading for all inputs. Do not special-case the listed inputs. The previous
implementation, for reference:
```python
{code}
```

{rules}
""".strip()

INPUTS_PROMPT_TEMPLATE = """
Write test inputs for the Python function `{entry_point}` described below.

### Problem
{prompt}

Return one ```python block that defines three lists:
- EXAMPLES: every input/output example stated in the problem, as `(args, expected)` pairs, with
  the expected values copied literally and with the same type (a bare `True` stays a boolean, a
  quoted `'42'` stays a string). Use [] if there are none.
- INPUTS: {num_inputs} typical inputs that exercise the main behaviour and are likely to separate
  different readings of the problem (ties, duplicates, ordering, boundaries of the main rule).
- EDGE_INPUTS: 4 edge cases: an empty or single-element input, zero or negative values, and
  inputs that probe formatting rules the problem leaves open, such as extra whitespace inside
  strings, mixed letter case, duplicates or reversed order.
Every input must be a valid, realistic input for this problem: same argument shape and types as
the signature and the examples, with data that fits the problem's domain. Do not invent inputs
the problem would never receive.
Every `args` is a tuple of positional arguments, e.g. `([1, 2, 3], 2)` for two arguments or
`([1, 2, 3],)` for a single list argument. If the problem gives no signature, use the most
natural argument shape for `{entry_point}` and keep it the same for every entry.
Use only Python literals. Do not call `{entry_point}` or any other function inside the lists.
""".strip()

QUESTION_PROMPT_TEMPLATE = """
You are helping to implement `{entry_point}` for the problem below. Before the final code is
written, the user can be asked exactly ONE clarifying question; you draft the candidates for it.

### Problem
{prompt}

### What we observed (internal, the user has not seen any of this)
{observations}
{history}
Draft exactly {num_drafts} candidate questions, one per slot, in this order:
1. the return value: its type and exact format, whenever the problem states neither and shows
   no example output (a yes/no versus a count versus a specific message, a list versus a
   string, an integer versus a float). If the problem does settle it, use this slot for the
   most important missing constraint instead;
2. the point the observations single out: the input where the readings diverge, or the conflict;
3. an edge case the problem leaves open: empty input, zero, a single element, duplicates,
   invalid input, error handling;
4. a vague term or missing rule: allowed operations, inclusive or exclusive bounds, index base,
   case sensitivity, order of the result, tie-breaking, rounding or precision.
Every slot must be filled with a different point; a reviewer picks one of them afterwards.
Priorities, highest first:
0. the return type or format, whenever the problem states neither and shows no example output;
1. a statement that conflicts with the signature, an example or another statement;
2. the shape of the arguments (one list versus separate values), but only when the problem gives
   no function signature and the observations report a disagreement about it;
3. a missing constraint (bound, ordering, tie-breaking, duplicates, empty or invalid input,
   error handling, return type or format);
4. a vague term with two readings that give different outputs.

Rules for the question:
- Ask about exactly one fact, in one sentence. Never bundle several questions.
- Ask a closed question that states the two possible readings of the requirement in words, each
  with its result on one realistic input. A question that only asks what to return for an input,
  without naming the rule being decided, is not allowed: it reads as a request for a test case.
  Shape: Does <rule> mean <reading A>, so that f(<input>) returns <A>, or <reading B>, so that it
  returns <B>? Take the input and both results from the observations when they name one. When a
  third reading is plausible, end with ", or another rule?" so the user can state the actual
  rule instead of rejecting both.
- Describe behaviour only through inputs and outputs. Never mention implementations, candidates,
  code, variables, algorithms or "what we observed": the user has not seen any of it.
- Write every input and output as plain text. Never use backticks, code fences or any other code
  formatting anywhere in the question: the user's reply copies the question's formatting, and
  formatted parts of the reply are lost.
- Never ask about anything the problem already states; if a point is settled by the text, move on
  to the next point.
- Do not ask about style, performance, loops, recursion or complexity, and never ask for the
  implementation, the algorithm or test cases.
- Ignore typos and formatting noise; they do not change the meaning.

Examples of the shape (one per kind of unclear point; the topics are illustrative only):
- Conflict: The description says to return a count, but the example shows a list; should
  f([1, 2, 2]) return the count 3 or the list [1, 2, 2]?
- Missing constraint: Is the bound inclusive, so that f([1, 2, 3], 2) returns [2, 3], or
  exclusive, so that it returns [3]?
- Vague term: Does "largest" mean largest by value, so that f(["b", "aaa"]) returns "b", or
  longest, so that it returns "aaa"?

{reply_format}
""".strip()

REPLY_ASK = (
    "Reply with a JSON object only, no other text, holding the list of candidate questions: "
    '{"questions": ["<first question>", "<second question>", "<third question>"]}'
)
REPLY_ASK_OR_SKIP = (
    "Reply with a JSON object only, no other text, holding the list of candidate questions: "
    '{"questions": ["<first question>", "<second question>", "<third question>"]}. Only if '
    'the problem is completely unambiguous, reply {"questions": []} instead.'
)
REPLY_RETRY = (
    "Your previous reply did not follow the rules. Reply with a JSON object only, "
    '{"questions": ["<first question>", "<second question>"]}, every question phrased only '
    "through inputs and outputs, and nothing else."
)

GRADE_PROMPT_TEMPLATE = """
A developer may ask the user ONE clarifying question before implementing `{entry_point}`. Grade
each candidate question below against the user's own criteria.

### Problem
{prompt}

### Candidate questions
{questions}

The user scores a question as good only if it meets all of: CRITICALITY (a real blocker to a
correct solution, not cosmetic), SEARCH SPACE (the answer cannot be reliably inferred from the
problem; it must come from the user), LEAKAGE (it asks for a specific fact, not the algorithm,
the implementation or test cases), ATOMICITY (a precise answer is exactly one fact) and
OBJECTIVITY (a single unambiguous resolution).
First list every candidate whose answer the problem already settles, quoting exactly the
sentence, signature or example that settles it (typos and formatting noise settle nothing).
Then pick, among the others, the ONE candidate most likely to reveal a requirement the problem
leaves out. When nothing in the problem shows what the function returns, the question about
the return value's type and format wins; otherwise prefer the question whose answer changes the
returned value on the most ordinary inputs.
Reply with a JSON object only, no other text, like
{"settled": [{"index": 2, "quote": "<the quoted part of the problem>"}], "best": 0}
with an empty "settled" list when nothing is settled.
""".strip()

CONFLICT_PROMPT_TEMPLATE = """
Read the problem for the Python function `{entry_point}` and look only for internal conflicts:
places where the description, the signature or type annotations, and the examples do not agree
about what the function receives or returns (for example the text says "return the count" while an
example shows a list, or the text states a rule that an example violates).

### Problem
{prompt}

Reply with a JSON object only, no other text:
{"conflicts": [{"a": "<what one part says, quoted>", "b": "<what the other part says, quoted>"}]}
with at most three entries, the most consequential first (a conflict about the returned value or
its type outranks one about wording), or {"conflicts": []} if everything agrees. Ignore typos and
formatting noise. Do not report missing information, only disagreements between parts that are
present.
""".strip()

EXPECTED_PROMPT_TEMPLATE = """
The user clarified a requirement for the Python function `{entry_point}`.

### Problem
{prompt}

### Clarification
Question: {question}
Answer: {answer}

Based on the problem and the user's answer, what exactly must `{call}` return?
Reply with a JSON object only, no other text:
{"value": "<the returned value as a Python literal in a string, for example '[1, 2]', or null>",
 "raises": "<ExceptionName if the call must raise, or null>"}
Set both to null when the answer does not determine the result.
""".strip()

DEFINITE_PROMPT_TEMPLATE = """
A developer asked the user a clarifying question about the Python function `{entry_point}`.

### Question
{question}

### Answer
{answer}

Does the answer commit to one reading or state a concrete value, so that the developer can rely
on it? It does not if the user says the point is unspecified, ambiguous, a design choice or up to
the developer, or only gestures at an answer without settling it.
Reply with a JSON object only, no other text: {"definite": true or false}
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
print("@@EDGE_FROM " + str(len(EXAMPLES) + len(INPUTS)), flush=True)
print("@@INPUTS " + json.dumps([
    (ENTRY + "(" + ", ".join(repr(v) for v in a) + ")")[:200] for a in CALLS
]), flush=True)
print("@@ARGS " + json.dumps([repr(a)[:400] for a in CALLS]), flush=True)
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
    """All fenced blocks of the inputs reply, joined, so EXAMPLES and INPUTS both survive."""
    blocks = _code_blocks(response)
    return "\n".join(blocks) if blocks else None


def _is_script_noise(node: ast.stmt) -> bool:
    """Top-level statements that would run (and possibly fail) when the solution is imported."""
    if isinstance(node, ast.If):
        return "__name__" in ast.unparse(node.test)
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return isinstance(node.value.func, ast.Name)
    return False


def _finalize(code: str, entry_point: str) -> str:
    """Drops demo code and aliases the entry point if the model renamed the function."""
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


def _entry_signature(code: str, entry_point: str) -> dict[str, Any] | None:
    """Positional arity of the entry point, read statically (no execution)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            args = node.args
            positional = list(args.posonlyargs) + list(args.args)
            names = [a.arg for a in positional]
            if args.vararg:
                names.append("*" + args.vararg.arg)
            return {
                "required": len(positional) - len(args.defaults),
                "optional": len(args.defaults),
                "varargs": args.vararg is not None,
                "text": f"{entry_point}({', '.join(names)})",
            }
    return None


def _has_explicit_signature(prompt: str, entry_point: str) -> bool:
    return re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", prompt) is not None


def _setup_literals(setup: str) -> tuple[list | None, list | None]:
    """EXAMPLES and INPUTS of the setup block as literals, or None when they are not literal."""
    try:
        tree = ast.parse(setup)
    except SyntaxError:
        return None, None
    found: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("EXAMPLES", "INPUTS")
        ):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
    examples = found.get("EXAMPLES")
    inputs = found.get("INPUTS")
    return (examples if isinstance(examples, list) else None), (
        inputs if isinstance(inputs, list) else None
    )


def _call_arity(args: Any) -> int:
    return len(args) if isinstance(args, tuple) else 1


def _with_witness(setup: str, args_source: str, expected_source: str) -> str:
    """Prepends the clarified (input, expected) pair to EXAMPLES, keeping the original setup."""
    body = "\n".join("    " + line for line in setup.splitlines()) or "    pass"
    return (
        f"try:\n{body}\nexcept BaseException:\n    pass\n"
        f"EXAMPLES = [({args_source}, {expected_source})] + list(globals().get('EXAMPLES', []))\n"
    )


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
    """The JSON object in a reply; anything around it (a code fence, a sentence) is ignored."""
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


def _parse_questions(response: str) -> list[str] | None:
    """Candidate questions from a {"questions": [...]} reply; None when the reply is unreadable."""
    data = _parse_json(response)
    if data is None or not isinstance(data.get("questions"), list):
        return None
    questions = [str(q).strip() for q in data["questions"] if isinstance(q, str) and q.strip()]
    return [q for q in questions if len(q) <= 600]


def _parse_conflicts(response: str) -> list[str]:
    """Conflicts as "A | B" strings, in the order the model ranked them."""
    data = _parse_json(response) or {}
    items = data.get("conflicts")
    conflicts = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("a") and item.get("b"):
            conflicts.append(f"{str(item['a']).strip()} | {str(item['b']).strip()}"[:300])
    return conflicts[:3]


def _parse_grades(response: str, count: int) -> tuple[int | None, set[int]]:
    """Return (reviewer's pick, indices the prompt already answers).

    "Settled" needs a quote from the prompt; without one we ignore the claim, because the
    reviewer says it far too often. Unreadable reply -> pick nothing, settle nothing.
    """
    data = _parse_json(response) or {}
    settled: set[int] = set()
    items = data.get("settled")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        index, quote = item.get("index"), item.get("quote")
        if isinstance(index, int) and 0 <= index < count:
            if isinstance(quote, str) and len(quote.strip()) > 8:
                settled.add(index)
    best = data.get("best")
    return (best if isinstance(best, int) and 0 <= best < count else None), settled


def _parse_expected(response: str) -> str | None:
    """Source of the expected literal for the witness call, or None when undetermined."""
    data = _parse_json(response) or {}
    raises = data.get("raises")
    if isinstance(raises, str) and raises.strip().isidentifier():
        return repr("__EXC__:" + raises.strip())
    value = data.get("value")
    if value is None:
        return None
    text = value.strip() if isinstance(value, str) else repr(value)
    try:
        return repr(ast.literal_eval(text))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _parse_definite(response: str) -> bool:
    """Whether the user's answer commits to a reading; an unreadable reply counts as definite."""
    data = _parse_json(response)
    if data is None or "definite" not in data:
        return True
    return bool(data["definite"])


class _Execution:
    """Behaviour of a set of candidates on the shared inputs, grouped into clusters.

    Clusters are ranked by size, or first by agreement with the problem's own examples when
    `prefer_examples` is set. `priority` marks how many leading examples come from the user's
    clarification; matching those outranks everything else.
    """

    def __init__(
        self,
        calls: list[str],
        args: list[str],
        expected: list[str],
        rows: list[list[str] | None],
        prefer_examples: bool,
        priority: int = 0,
        edge_from: int | None = None,
    ):
        self.calls = calls
        self.args = args
        self.expected = expected
        self.rows = rows
        self.edge_from = len(calls) if edge_from is None else edge_from
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
            clarified = self.example_passes(outputs[:priority], priority)
            if prefer_examples:
                return (not all_failed, clarified, passes, len(members), -failures)
            return (not all_failed, clarified, len(members), passes, -failures)

        self.clusters = sorted(groups.items(), key=rank, reverse=True)

    def example_passes(self, outputs: tuple[str, ...] | list[str], limit: int | None = None) -> int:
        expected = self.expected if limit is None else self.expected[:limit]
        return sum(o == e for o, e in zip(outputs, expected, strict=False))

    def best(self) -> int | None:
        return self.clusters[0][1][0] if self.clusters else None

    def dominance(self) -> float:
        if not self.clusters or not self.loaded:
            return 0.0
        return len(self.clusters[0][1]) / self.loaded

    def all_failed(self) -> bool:
        return self.loaded > 0 and all(
            all(_is_failure(o) for o in row) for row in self.rows if row is not None
        )

    def top_is_clean(self) -> bool:
        """The leading cluster has no failures and passes every stated example."""
        if not self.clusters or self.loaded < 2 or len(self.calls) < 3:
            return False
        outputs = self.clusters[0][0]
        if any(_is_failure(o) for o in outputs):
            return False
        return self.example_passes(outputs) == len(self.expected)

    def witness(self) -> int | None:
        """The input the candidates disagree on most (max entropy), preferring value-vs-value
        splits over crash-vs-value ones and penalising the edge inputs a little.
        """
        if len(self.clusters) < 2:
            return None
        rows = [row for row in self.rows if row is not None]
        best_index, best_score = None, (-1.0, 0)
        for index in range(len(self.calls)):
            outputs = [row[index] for row in rows if index < len(row)]
            counts = Counter(outputs)
            if len(counts) < 2:
                continue
            total = len(outputs)
            entropy = -sum(c / total * math.log(c / total) for c in counts.values())
            clean = sum(1 for o in counts if not _is_failure(o)) >= 2
            edge_penalty = 0.6 if index >= self.edge_from else 0.0
            score = (entropy + (1.0 if clean else 0.0) - edge_penalty, -index)
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    def witness_groups(self, index: int) -> list[tuple[str, int, int]]:
        """(output, count, representative candidate) at one input, most common first."""
        first_seen: dict[str, int] = {}
        counts: Counter[str] = Counter()
        for candidate, row in enumerate(self.rows):
            if row is None or index >= len(row):
                continue
            counts[row[index]] += 1
            first_seen.setdefault(row[index], candidate)
        return [(output, n, first_seen[output]) for output, n in counts.most_common()]


class ReqForge(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "num_candidates": 8,
        "num_final": 8,
        "num_drafts": 4,
        "num_inputs": 8,
        "parallel_calls": 8,
        "llm_attempts": 3,
        "max_questions": 3,
        "skip_dominance": 0.8,
        "dominance": 0.6,
        "exec_time_budget": 20.0,
        "num_repairs": 3,
    }

    def complete(self, env: ClarificationEnvironment, prompt: str) -> str | None:
        """One model call; transient failures are retried with backoff, budget errors are not."""
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
        """Samples `count` implementations (plus any extra prompts first); retries once if none parse."""
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
        priority: int = 0,
    ) -> _Execution | None:
        if not candidates or not setup:
            return None

        script = _build_harness(entry_point, candidates, setup, self.config["exec_time_budget"])
        try:
            output = env.exec_code(script)
        except Exception:
            return None

        calls, args, expected, rows = None, [], [], [None] * len(candidates)
        edge_from = None
        for line in output.splitlines():
            try:
                if line.startswith("@@EDGE_FROM "):
                    edge_from = int(line[len("@@EDGE_FROM ") :])
                elif line.startswith("@@INPUTS "):
                    calls = json.loads(line[len("@@INPUTS ") :])
                elif line.startswith("@@ARGS "):
                    args = json.loads(line[len("@@ARGS ") :])
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
        return _Execution(calls, args, expected, rows, prefer_examples, priority, edge_from)

    def signature_notes(
        self, problem: dict[str, Any], candidates: list[str], setup: str | None
    ) -> list[str]:
        """Static evidence about the argument shape, only when the prompt leaves it open."""
        entry_point = problem["entry_point"]
        if _has_explicit_signature(problem["prompt"], entry_point):
            return []

        signatures = [s for s in (_entry_signature(c, entry_point) for c in candidates) if s]
        texts = Counter(s["text"] for s in signatures)
        arities = {s["required"] for s in signatures if not s["varargs"]}

        examples, inputs = _setup_literals(setup or "")
        calls = [pair[0] for pair in (examples or []) if isinstance(pair, tuple) and len(pair) == 2]
        calls += list(inputs or [])
        call_arities = Counter(_call_arity(a) for a in calls)

        notes = []
        if len(arities) > 1:
            summary = "; ".join(f"{n} define `{t}`" for t, n in texts.most_common(3))
            notes.append(f"Implementations disagree on the argument shape: {summary}.")
        if call_arities and arities:
            common = call_arities.most_common(1)[0][0]
            if common not in arities:
                notes.append(
                    f"The generated inputs call `{entry_point}` with {common} argument(s), but the "
                    f"implementations expect {sorted(arities)}; the argument shape is unclear."
                )
        return notes

    def audit(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> list[str]:
        """Internal conflicts between prose, signature and examples (Larbi-style contradictions)."""
        prompt = CONFLICT_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
            "{entry_point}", problem["entry_point"]
        )
        return _parse_conflicts(self.complete(env, prompt) or "")

    def describe(
        self,
        problem: dict[str, Any],
        candidates: list[str],
        setup: str | None,
        execution: _Execution | None,
        conflicts: list[str],
    ) -> str:
        lines = []
        if conflicts:
            lines.append(
                "Conflicts inside the problem statement. These are the most critical ambiguity: "
                "ask about the FIRST one listed, asking which side is intended."
            )
            lines += [f"- {conflict}" for conflict in conflicts]
        lines += self.signature_notes(problem, candidates, setup)
        if execution is None:
            lines.append("Running the candidate implementations was not possible.")
            return "\n".join(lines)

        if execution.all_failed():
            lines.append(
                "Every implementation failed on every input (for example with TypeError), so the "
                "implementations and the inputs assume different argument shapes."
            )

        top_outputs = execution.clusters[0][0] if execution.clusters else ()
        mismatches = [
            (execution.calls[i], expected, top_outputs[i])
            for i, expected in enumerate(execution.expected)
            if i < len(top_outputs) and top_outputs[i] != expected
        ]
        for call, expected, got in mismatches[:3]:
            lines.append(
                f"The problem's own example `{call}` expects `{expected}`, "
                f"but the leading implementation returns `{got}`."
            )
        if mismatches:
            lines.append(
                "Ask about this conflict first: say what the description requires, say what the "
                "example shows, and ask which of the two is intended."
            )

        witness = execution.witness()
        if witness is None:
            lines.append(
                f"{execution.loaded} independent implementations behaved identically "
                f"on {len(execution.calls)} test inputs."
            )
            return "\n".join(lines)

        groups = execution.witness_groups(witness)
        lines.append(f"Independent implementations disagree on `{execution.calls[witness]}`:")
        for output, count, _ in groups[:4]:
            lines.append(f"- {count} implementation(s) return `{output}`")
        for label, (_, _, representative) in zip("AB", groups[:2], strict=False):
            lines.append(
                f"\nImplementation {label}:\n```python\n{candidates[representative][:1500]}\n```"
            )
        return "\n".join(lines)

    def is_definite(
        self, env: ClarificationEnvironment, problem: dict[str, Any], question: str, answer: str
    ) -> bool:
        """Whether the user's answer settles the point; a hedge is not used as a constraint."""
        prompt = (
            DEFINITE_PROMPT_TEMPLATE.replace("{entry_point}", problem["entry_point"])
            .replace("{question}", question)
            .replace("{answer}", answer)
        )
        return _parse_definite(self.complete(env, prompt) or "")

    def formulate_question(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        candidates: list[str],
        setup: str | None,
        execution: _Execution | None,
        clarifications: list[tuple[str, str]],
        unanswered: list[str],
        conflicts: list[str],
    ) -> str | None:
        notes = self.signature_notes(problem, candidates, setup)
        may_skip = bool(clarifications) or (
            execution is not None
            and execution.expected
            and execution.top_is_clean()
            and execution.dominance() >= self.config["skip_dominance"]
            and not notes
            and not conflicts
        )
        history = ""
        if clarifications:
            history = "\n### Already clarified (never ask about these facts again)\n" + "\n".join(
                f"Q: {q}\nA: {a}" for q, a in clarifications
            )
        if unanswered:
            history += (
                "\n### Already asked, but the user could not answer (ask about something else)\n"
            )
            history += "\n".join(f"Q: {q}" for q in unanswered)
        base_prompt = (
            QUESTION_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace(
                "{observations}", self.describe(problem, candidates, setup, execution, conflicts)
            )
            .replace("{history}", history + "\n" if history else "")
            .replace("{num_drafts}", str(self.config["num_drafts"]))
            .replace("{reply_format}", REPLY_ASK_OR_SKIP if may_skip else REPLY_ASK)
        )

        drafts = _parse_questions(self.complete(env, base_prompt) or "")
        if drafts is None:
            drafts = _parse_questions(self.complete(env, base_prompt + "\n\n" + REPLY_RETRY) or "")
        if not drafts:
            return None
        drafts = drafts[: self.config["num_drafts"]]

        best, settled = self.grade_questions(env, problem, drafts)
        viable = [index for index in range(len(drafts)) if index not in settled]
        if not viable:
            return None
        index = best if best in viable else viable[0]
        return drafts[index].replace("`", "").strip()

    def grade_questions(
        self, env: ClarificationEnvironment, problem: dict[str, Any], drafts: list[str]
    ) -> tuple[int | None, set[int]]:
        """Reviewer's pick among the drafts, and the drafts the problem already settles."""
        listing = "\n".join(f"{index}. {draft}" for index, draft in enumerate(drafts))
        prompt = (
            GRADE_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{questions}", listing)
        )
        return _parse_grades(self.complete(env, prompt) or "", len(drafts))

    def witness_expectation(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        execution: _Execution | None,
        question: str,
        answer: str,
    ) -> tuple[str, str] | None:
        """(args source, expected source) for the witness input, derived from the user's answer."""
        if execution is None:
            return None
        index = execution.witness()
        if index is None or index >= len(execution.args):
            return None
        args_source = execution.args[index]
        try:
            ast.literal_eval(args_source)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
        prompt = (
            EXPECTED_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{question}", question)
            .replace("{answer}", answer)
            .replace("{call}", execution.calls[index])
        )
        expected_source = _parse_expected(self.complete(env, prompt) or "")
        return (args_source, expected_source) if expected_source else None

    def repair(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        candidates: list[str],
        setup: str | None,
        execution: _Execution | None,
        priority: int = 0,
    ) -> tuple[list[str], _Execution | None]:
        """One repair round when the leader fails the example we got from the user.

        Only the first `priority` examples (the ones derived from the answer) are trusted;
        examples printed in the prompt can be the corrupted part of a variant. We show the
        model the failing call with expected vs actual and let it rewrite. Repaired programs
        join the pool and the normal example-first ranking decides.
        """
        if execution is None or priority <= 0 or not execution.expected or not setup:
            return candidates, execution
        best = execution.best()
        if best is None or best >= len(candidates) or execution.rows[best] is None:
            return candidates, execution
        failures = [
            f"- {call} should return {expected} but returns {got}"
            for call, expected, got in zip(
                execution.calls[:priority],
                execution.expected[:priority],
                execution.rows[best][:priority],
                strict=False,
            )
            if got != expected
        ]
        if not failures:
            return candidates, execution

        block = ""
        if clarifications:
            answers = "\n".join(f"Question: {q}\nAnswer: {a}" for q, a in clarifications)
            block = (
                "\n### Clarifications from the user\n"
                f"{answers}\nFollow the user's answers exactly.\n"
            )
        entry_point = problem["entry_point"]
        prompt = (
            REPAIR_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry_point)
            .replace("{clarifications}", block)
            .replace("{failures}", "\n".join(failures))
            .replace("{rules}", CODE_RULES.replace("{entry_point}", entry_point))
            .replace("{code}", candidates[best][:3000])
        )
        responses = self.complete_many(env, [prompt] * self.config["num_repairs"])
        repaired = self.parse_candidates(responses, entry_point)
        if not repaired:
            return candidates, execution
        pool = candidates + repaired
        pooled = self.execute(
            env, entry_point, pool, setup, prefer_examples=True, priority=priority
        )
        if pooled is None:
            return candidates, execution
        return pool, pooled

    @staticmethod
    def fenced(code: str) -> str:
        return f"```python\n{code}\n```"

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        entry_point = problem["entry_point"]

        inputs_prompt = (
            INPUTS_PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry_point)
            .replace("{num_inputs}", str(self.config["num_inputs"]))
        )
        extra, candidates = self.sample(
            env, problem, [], self.config["num_candidates"], extra_prompts=[inputs_prompt]
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

        clarifications: list[tuple[str, str]] = []
        unanswered: list[str] = []
        conflicts = self.audit(env, problem) if env.can_ask() else []
        for turn in range(self.config["max_questions"]):
            if not env.can_ask():
                break
            retry = bool(unanswered) and not clarifications
            if (
                turn > 0
                and not retry
                and (
                    execution is None
                    or execution.witness() is None
                    or execution.dominance() >= self.config["dominance"]
                )
            ):
                break

            question = self.formulate_question(
                env, problem, candidates, setup, execution, clarifications, unanswered, conflicts
            )
            if not question:
                break
            try:
                answer = env.ask_human(question)
            except TooManyQuestionException:
                break
            useless = not answer.strip() or answer.strip().lower() in USELESS_ANSWERS
            if useless or not self.is_definite(env, problem, question, answer):
                if unanswered:
                    break
                unanswered.append(question)
                continue
            clarifications.append((question, answer))

            witness = self.witness_expectation(env, problem, execution, question, answer)
            _, finals = self.sample(env, problem, clarifications, self.config["num_final"])
            if not finals:
                break

            final_setup, priority = setup, 0
            if witness and setup:
                final_setup, priority = _with_witness(setup, *witness), 1
            pool = finals + [best]
            execution = self.execute(
                env, entry_point, pool, final_setup, prefer_examples=True, priority=priority
            )
            candidates, execution = self.repair(
                env, problem, clarifications, pool, final_setup, execution, priority
            )
            best = (
                candidates[execution.best()]
                if execution and execution.best() is not None
                else candidates[0]
            )

        return self.fenced(best)
