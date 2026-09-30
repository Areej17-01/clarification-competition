# SPDX-FileCopyrightText: 2026 Areej Mehboob <mehboobareej01@gmail.com>
#
# SPDX-License-Identifier: MIT

"""ReqForge.

Work in progress. The LLM is asked for a solution to the coding problem.
If it responds with a question instead of code, the question is forwarded
to the user and the LLM is queried again with the clarification.

Team: Rubber Duckers
Team Members: Areej Mehboob, Taha Mehboob (TU Ilmenau)
Main Contact: mehboobareej01@gmail.com
"""

from __future__ import annotations

from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import ClarificationEnvironment, TooManyQuestionException
from clarify.runtime import _validate_and_parse_evalplus_result

PROMPT_TEMPLATE = """
Please provide a self-contained Python script implementing `{entry_point}` that solves the following problem:

{prompt}

Enclose your solution in ```python and ```.
""".strip()


class ReqForge(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {}

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    PROMPT_TEMPLATE.replace("{prompt}", problem["prompt"]).replace(
                        "{entry_point}", problem["entry_point"]
                    )
                ),
            }
        ]

        response = env.llm(messages)

        messages += [{"role": "assistant", "content": response}]

        while True:
            try:
                return _validate_and_parse_evalplus_result(response)
            except ValueError:
                # The response might not contain an answer, assume a question is raised.
                try:
                    answer = env.ask_human(response)
                    messages += [{"role": "user", "content": answer}]
                    response = env.llm(messages)
                    messages += [{"role": "assistant", "content": response}]
                except TooManyQuestionException:
                    return response
