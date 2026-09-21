"""JevController: TypeSafe Jev as the control policy.

One state, five judgments, one HTTP call:

* ``action``           Choice - continue / switch / verify / stop / other
* ``tool``             Choice - python / sql / none / other
* ``retain_insight``   Noul   - is the latest result worth keeping?
* ``needs_verification`` Noul - must an important finding be checked?
* ``completeness``     Score  - how complete is the analysis for the goal?

The full probability distributions are copied into the typed decision, which is
what the trace stores so a future UI can visualise Jev's decisions.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from datajev.controllers.base import Controller, ControllerError
from datajev.state import AnalysisState
from datajev.types import (
    ACTIONS,
    TOOLS,
    ControllerDecision,
    argmax_option,
    confidence_from_probabilities,
    normalize_probabilities,
)

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


def build_questions() -> dict[str, dict[str, Any]]:
    """The five control questions asked over one state, in one call."""
    return {
        "action": {
            "type": "choice",
            "instructions": (
                "Given `goal`, `latest_result`, `retained_insights` and "
                "`recent_trajectory`, what should the data analyst do next?"
            ),
            "criteria": {
                "continue": "Deepen the current analytical direction with another step.",
                "switch": "Explore a substantially different direction that the analysis has not covered.",
                "verify": "Verify or stress-test an important current finding before relying on it.",
                "stop": "Enough evidence already exists to answer the goal.",
                "other": "None of these actions is appropriate.",
            },
        },
        "tool": {
            "type": "choice",
            "instructions": (
                "Which tool would most improve the next step of the analysis of `goal`, "
                "given `current_direction`, `latest_result` and `dataset`?"
            ),
            "criteria": {
                "python": "Numerical, statistical or code-based analysis.",
                "sql": "Structured querying or aggregation is needed.",
                "none": "No additional tool is needed.",
                "other": "Another tool is more appropriate.",
            },
        },
        "retain_insight": {
            "type": "noul",
            "instructions": (
                "Does `latest_result` provide meaningful, non-redundant evidence toward "
                "answering `goal`, beyond what is already listed in `retained_insights`?"
            ),
            "criteria": {
                "true": "The result adds a distinct, decision-relevant finding.",
                "false": "The result is redundant, inconclusive or irrelevant.",
            },
        },
        "needs_verification": {
            "type": "noul",
            "instructions": (
                "Should an important current finding be independently checked before it is "
                "relied upon, given `goal`, `latest_result` and `recent_trajectory`?"
            ),
            "criteria": {
                "true": "A load-bearing finding is unchecked or fragile.",
                "false": "Existing findings are already well supported.",
            },
        },
        "completeness": {
            "type": "score",
            "instructions": "How complete is the analysis so far with respect to `goal`?",
            "criteria": [
                "Major parts of the goal remain unanswered.",
                "Some useful evidence exists but important gaps remain.",
                "Most important aspects are covered.",
                "The goal is comprehensively supported by evidence.",
            ],
        },
    }


class JevController(Controller):
    """TypeSafe Jev (System One) as a DataJev controller."""

    name = "jev"
    description = "TypeSafe Jev: one state, five typed control judgments per step."

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ControllerError(
                "JevController needs TYPESAFE_API_KEY (or --typesafe-api-key). "
                "Use --controller heuristic for an offline run."
            )
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_s)

    # ------------------------------------------------------------------ #
    # Request construction (pure, easy to test)
    # ------------------------------------------------------------------ #

    def build_request(self, state: AnalysisState) -> dict[str, Any]:
        return {
            "state": state.to_jev_state(),
            "model": self.model,
            "questions": build_questions(),
        }

    def decide(self, state: AnalysisState) -> ControllerDecision:
        payload = self.build_request(state)
        started = time.perf_counter()
        response = self._post(payload)
        latency = time.perf_counter() - started

        answers = response.get("answers") or {}
        if not answers:
            raise ControllerError("Jev returned no answers.")

        action = answers.get("action", {})
        action_probs = normalize_probabilities(action.get("probabilities") or {})
        action_choice = action.get("choice") or argmax_option(action_probs, "continue")
        if action_choice not in ACTIONS:
            action_probs = normalize_probabilities({**action_probs, "other": action_probs.get("other", 0.0)})
            action_choice = argmax_option(action_probs, "continue")

        tool = answers.get("tool", {})
        tool_probs = normalize_probabilities(tool.get("probabilities") or {})
        tool_choice = tool.get("choice") or argmax_option(tool_probs, "python")
        if tool_choice not in TOOLS:
            tool_choice = "other"

        retain = answers.get("retain_insight", {})
        verify = answers.get("needs_verification", {})
        completeness = answers.get("completeness", {})
        completeness_probs = normalize_probabilities(completeness.get("probabilities") or {})

        return ControllerDecision(
            source="jev",
            action=action_choice,
            action_probabilities=action_probs,
            action_confidence=action.get("confidence", confidence_from_probabilities(action_probs)),
            tool=tool_choice,
            tool_probabilities=tool_probs,
            tool_confidence=tool.get("confidence", confidence_from_probabilities(tool_probs)),
            retain_insight=float(retain.get("noul", 0.0) or 0.0),
            needs_verification=float(verify.get("noul", 0.0) or 0.0),
            completeness=float(completeness.get("score", 0.0) or 0.0),
            completeness_probabilities=completeness_probs,
            completeness_legend={str(k): str(v) for k, v in (completeness.get("legend") or {}).items()},
            completeness_confidence=completeness.get(
                "confidence", confidence_from_probabilities(completeness_probs)
            ),
            latency_s=round(latency, 4),
            model=response.get("model") or self.model,
            usage=response.get("usage"),
            raw=answers,
        )

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = "unknown error"
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(self.endpoint, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"connection error: {exc}"
            else:
                if response.status_code < 400:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in RETRY_STATUSES:
                    raise ControllerError(f"Jev request failed ({last_error})")
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt * 0.5)
        raise ControllerError(f"Jev request failed after {self.max_retries} attempts ({last_error})")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
