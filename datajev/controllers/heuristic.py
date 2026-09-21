"""HeuristicController: a deterministic, offline control policy.

It exists so the full vertical slice can run and be tested with no API keys at
all. It emits the same typed decision shape (with explicit probability
distributions) as the Jev and LLM controllers, so the trace and the CLI render
identically.
"""

from __future__ import annotations

import math

from datajev.controllers.base import Controller
from datajev.state import AnalysisState
from datajev.types import ControllerDecision, argmax_option, normalize_probabilities

COMPLETENESS_LEGEND = {
    "0": "Major parts of the goal remain unanswered.",
    "1": "Some useful evidence exists but important gaps remain.",
    "2": "Most important aspects are covered.",
    "3": "The goal is comprehensively supported by evidence.",
}
LEVELS = ("0", "1", "2", "3")


def _softmax(scores: dict[str, float], temperature: float = 0.7) -> dict[str, float]:
    peak = max(scores.values())
    exps = {key: math.exp((value - peak) / temperature) for key, value in scores.items()}
    return normalize_probabilities(exps)


class HeuristicController(Controller):
    """Rule-based controller used for offline runs and tests."""

    name = "heuristic"
    description = "Deterministic rules over the analytical state (offline fallback)."

    def decide(self, state: AnalysisState) -> ControllerDecision:
        steps = state.step_count
        retained = len(state.retained_insights())
        verified = state.verified
        seen_actions = set(state.action_history)

        # ---------------------------------------------------------------- #
        # action scores
        # ---------------------------------------------------------------- #
        scores = {"continue": 1.0, "switch": 0.55, "verify": 0.35, "stop": -0.6, "other": -1.5}
        if steps == 0:
            scores.update({"continue": 2.0, "switch": 0.6, "verify": 0.1, "stop": -2.0, "other": -2.0})
        else:
            if steps == 1:
                # A first global pass is rarely a sufficient answer to an
                # open-ended goal: prefer a genuinely different direction.
                scores["switch"] += 0.9
            if retained <= 1:
                scores["continue"] += 0.4
            if not verified:
                scores["verify"] += 0.35
                scores["stop"] -= 0.6
            if "verify" not in seen_actions and steps >= 2:
                scores["verify"] += 0.6
            if "switch" not in seen_actions and steps >= 2:
                scores["switch"] += 0.25
            # Stop only becomes plausible once findings are checked.
            if verified and steps >= 3:
                scores["stop"] += 1.4 + 0.25 * min(steps, 8)
            if verified and retained >= 3:
                scores["stop"] += 0.4
        action_probs = _softmax(scores)
        action = argmax_option(action_probs, "continue")

        # ---------------------------------------------------------------- #
        # tool scores
        # ---------------------------------------------------------------- #
        tool_scores = {"python": 1.0, "sql": -0.8, "none": -1.2, "other": -1.5}
        if action == "verify":
            tool_scores["python"] += 0.6
        tool_probs = _softmax(tool_scores)
        tool = argmax_option(tool_probs, "python")

        # ---------------------------------------------------------------- #
        # Nouls and the completeness Score
        # ---------------------------------------------------------------- #
        has_result = bool(state.latest_summary or state.latest_result)
        retain = 0.88 if (has_result and steps >= 1) else 0.15
        needs_verification = 0.0
        if steps >= 2 and not verified:
            needs_verification = 0.78
        elif steps >= 1 and not verified:
            needs_verification = 0.6
        elif verified:
            needs_verification = 0.12

        completeness_score = min(
            3.0, 0.6 * steps + 1.0 * (1.0 if verified else 0.0) + 0.15 * retained
        )
        completeness_probs = _triangular(completeness_score)

        decision = ControllerDecision(
            source="heuristic",
            action=action,
            action_probabilities=action_probs,
            action_confidence=max(action_probs.values()) if action_probs else None,
            tool=tool,
            tool_probabilities=tool_probs,
            tool_confidence=max(tool_probs.values()) if tool_probs else None,
            retain_insight=round(retain, 4),
            needs_verification=round(needs_verification, 4),
            completeness=round(completeness_score, 4),
            completeness_probabilities=completeness_probs,
            completeness_legend=COMPLETENESS_LEGEND,
            completeness_confidence=max(completeness_probs.values()) if completeness_probs else None,
            model=None,
            usage=None,
            raw=None,
            notes=["rules: offline heuristic controller"],
        )
        return decision


def _triangular(score: float) -> dict[str, float]:
    """Spread a Score value over its four levels."""
    weights = {}
    for index, level in enumerate(LEVELS):
        weights[level] = max(0.0, 1.0 - abs(score - index) / 1.5) + 1e-6
    return normalize_probabilities(weights)
