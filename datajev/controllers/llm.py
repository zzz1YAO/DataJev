"""LLMController: the LLM-only baseline.

It answers the *same five control questions* as Jev, but with a generative
model. Same typed output shape, so runs are directly comparable in the trace:
same analyst, same data, different control policy.
"""

from __future__ import annotations

import time
from typing import Any

from datajev.controllers.base import Controller, ControllerError
from datajev.llm import LLMClient, LLMError, resolve_llm_config
from datajev.state import AnalysisState
from datajev.types import (
    ACTIONS,
    TOOLS,
    ControllerDecision,
    argmax_option,
    confidence_from_probabilities,
    normalize_probabilities,
)

SYSTEM_PROMPT = (
    "You are the control policy of a data-analysis agent. You do not analyze data "
    "yourself; you look at the agent's compressed analytical state and decide what it "
    "should do next. Answer with a single JSON object and nothing else."
)

USER_TEMPLATE = """\
ANALYTICAL STATE
{state}

TASK
Return exactly one JSON object with these keys:

{{
  "action": {{"continue": p, "switch": p, "verify": p, "stop": p, "other": p}},
  "tool": {{"python": p, "sql": p, "none": p, "other": p}},
  "retain_insight": probability that the latest result is worth keeping,
  "needs_verification": probability that an important finding must be checked,
  "completeness": {{"0": p, "1": p, "2": p, "3": p}}
}}

Rules:
- action/switch means explore a substantially different direction; do not let the
  analysis stop after a single global summary if the goal asks whether that summary
  is sufficient.
- retain_insight and needs_verification are probabilities between 0 and 1.
- completeness probabilities are for these levels:
  0: major parts of the goal remain unanswered;
  1: some evidence exists but important gaps remain;
  2: most important aspects are covered;
  3: the goal is comprehensively supported by evidence.
- Each probability group must sum to 1.
"""


class LLMController(Controller):
    """Baseline controller that asks a generative model for the decisions."""

    name = "llm"
    description = "LLM-only baseline: a generative model makes the control decisions."

    def __init__(self, client: LLMClient | None = None, **kwargs: Any) -> None:
        self.client = client or LLMClient(resolve_llm_config(**kwargs))
        if not self.client.available:
            raise ControllerError(
                "LLMController needs an LLM API key (DATAJEV_LLM_API_KEY, OPENAI_API_KEY "
                "or deepseek_api). Use --controller heuristic for an offline run."
            )

    def decide(self, state: AnalysisState) -> ControllerDecision:
        user = USER_TEMPLATE.format(state=state.render_text())
        started = time.perf_counter()
        try:
            response = self.client.json(SYSTEM_PROMPT, user, max_tokens=700)
        except LLMError as exc:
            raise ControllerError(str(exc)) from exc
        latency = time.perf_counter() - started
        if response.parse_error:
            raise ControllerError(
                "LLM controller returned unparseable structured output: "
                f"{response.parse_error}; raw response: {response.text[:3000]}"
            )
        data = response.data

        action_block = data.get("action") if isinstance(data.get("action"), dict) else {}
        action_probs = normalize_probabilities(action_block)
        action = argmax_option(action_probs, "continue")
        if action not in ACTIONS:
            action = "other"

        tool_block = data.get("tool") if isinstance(data.get("tool"), dict) else {}
        tool_probs = normalize_probabilities(tool_block)
        tool = argmax_option(tool_probs, "python")
        if tool not in TOOLS:
            tool = "other"

        completeness_block = (
            data.get("completeness") if isinstance(data.get("completeness"), dict) else {}
        )
        completeness_probs = normalize_probabilities(completeness_block)
        completeness = 0.0
        for level, probability in completeness_probs.items():
            try:
                completeness += float(level) * probability
            except (TypeError, ValueError):
                continue

        return ControllerDecision(
            source="llm",
            action=action,
            action_probabilities=action_probs,
            action_confidence=confidence_from_probabilities(action_probs),
            tool=tool,
            tool_probabilities=tool_probs,
            tool_confidence=confidence_from_probabilities(tool_probs),
            retain_insight=_clamp01(data.get("retain_insight")),
            needs_verification=_clamp01(data.get("needs_verification")),
            completeness=round(completeness, 4),
            completeness_probabilities=completeness_probs,
            completeness_legend=COMPLETENESS_LEGEND,
            completeness_confidence=confidence_from_probabilities(completeness_probs),
            latency_s=round(latency, 4),
            model=self.client.config.model,
            usage=response.usage,
            raw=data,
        )


COMPLETENESS_LEGEND = {
    "0": "Major parts of the goal remain unanswered.",
    "1": "Some useful evidence exists but important gaps remain.",
    "2": "Most important aspects are covered.",
    "3": "The goal is comprehensively supported by evidence.",
}


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(1.0, max(0.0, numeric)), 6)
