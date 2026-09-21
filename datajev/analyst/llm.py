"""LLM analyst: the System-2 half, driven by a generative model.

Two calls per step: one to write the code, one to read the result. The LLM never
decides what direction to take next.
"""

from __future__ import annotations

from typing import Any

from datajev.analyst.base import Analyst
from datajev.analyst.prompts import (
    FINAL_SYSTEM,
    FINAL_USER,
    INTERPRET_SYSTEM,
    INTERPRET_USER,
    PLAN_SYSTEM,
    PLAN_USER,
)
from datajev.llm import LLMClient, LLMError, resolve_llm_config
from datajev.state import AnalysisState, clip
from datajev.types import (
    AnalysisPlan,
    ControllerDecision,
    ExecutionResult,
    FinalAnswer,
    Interpretation,
)


class AnalystError(RuntimeError):
    """Raised when the analyst cannot produce a usable step."""


class LLMAnalyst(Analyst):
    """Writes code and reads results with an OpenAI-compatible model."""

    name = "llm"
    description = "A generative model writes and interprets each analysis step."

    def __init__(self, client: LLMClient | None = None, **kwargs: Any) -> None:
        self.client = client or LLMClient(resolve_llm_config(**kwargs))
        if not self.client.available:
            raise AnalystError(
                "LLMAnalyst needs an LLM API key (DATAJEV_LLM_API_KEY, OPENAI_API_KEY "
                "or deepseek_api). Use --analyst scripted for an offline run."
            )

    # ------------------------------------------------------------------ #

    def plan(self, state: AnalysisState, decision: ControllerDecision) -> AnalysisPlan:
        artifacts = ", ".join(state.artifact_names()[-8:]) or "(none yet)"
        user = PLAN_USER.format(
            state=state.render_text(),
            action=decision.action,
            tool=decision.tool,
            completeness=decision.completeness,
            needs_verification=decision.needs_verification,
            artifacts=artifacts,
        )
        try:
            response = self.client.json(PLAN_SYSTEM, user, max_tokens=2200)
        except LLMError as exc:
            raise AnalystError(str(exc)) from exc

        if response.parse_error:
            first_raw = response.text
            repair_user = (
                f"{user}\n\n"
                "Your previous response was not valid JSON. Re-emit the complete response "
                "as one JSON object, with the Python code represented as a JSON string "
                "with escaped newlines. Do not use markdown fences or truncate the object.\n\n"
                "PREVIOUS RAW RESPONSE\n"
                f"{clip(first_raw, 12000)}"
            )
            try:
                repaired = self.client.json(PLAN_SYSTEM, repair_user, max_tokens=4000)
            except LLMError as exc:
                raise AnalystError(
                    "Analyst structured output could not be parsed and the repair request "
                    f"failed. First parse error: {response.parse_error}. "
                    f"Raw response: {clip(first_raw, 3000)}"
                ) from exc
            if repaired.parse_error:
                finish = f"; finish_reason={response.finish_reason}" if response.finish_reason else ""
                raise AnalystError(
                    "Analyst structured output could not be parsed after one repair attempt "
                    f"(first: {response.parse_error}{finish}; repair: {repaired.parse_error}). "
                    f"Raw response: {clip(first_raw, 3000)}. "
                    f"Repair response: {clip(repaired.text, 3000)}"
                )
            response = repaired

        data = response.data
        raw_code = data.get("code")
        code = raw_code.strip() if isinstance(raw_code, str) else ""
        code = _strip_fence(code)
        if not code:
            keys = ", ".join(sorted(str(key) for key in data)) or "none"
            raise AnalystError(
                "Analyst generated no code in parsed structured output "
                f"(keys: {keys}). Raw response: {clip(response.text, 3000)}"
            )

        question = str(data.get("question") or state.current_question or "Untitled step").strip()
        return AnalysisPlan(
            question=question,
            code=code,
            rationale=str(data.get("rationale") or "").strip(),
            tool=decision.tool,
        )

    def interpret(
        self, state: AnalysisState, plan: AnalysisPlan, execution: ExecutionResult
    ) -> Interpretation:
        user = INTERPRET_USER.format(
            goal=state.goal,
            question=plan.question,
            code=clip(plan.code, 3000),
            ok=execution.ok,
            output=clip(execution.output, 6000) or "(no output)",
        )
        try:
            response = self.client.json(INTERPRET_SYSTEM, user, max_tokens=900)
        except LLMError as exc:
            return Interpretation(
                summary=f"Interpretation failed: {exc}",
                insights=[],
                finding="",
            )

        if response.parse_error:
            return Interpretation(
                summary=(
                    "Interpretation structured output could not be parsed: "
                    f"{response.parse_error}. Raw response: {clip(response.text, 1200)}"
                ),
                insights=[],
                finding="",
                usage=response.usage,
            )

        data = response.data
        insights = [str(item).strip() for item in (data.get("insights") or []) if str(item).strip()]
        return Interpretation(
            summary=str(data.get("summary") or "").strip(),
            insights=insights[:3],
            finding=str(data.get("finding") or "").strip(),
            usage=response.usage,
        )

    def finalize(self, state: AnalysisState, stop_reason: str) -> FinalAnswer:
        retained = state.retained_insights()
        insights = "\n".join(f"- {i.text}" for i in retained) or "(none retained)"
        trajectory = "\n".join(state.recent_trajectory(limit=10)) or "(no steps)"
        user = FINAL_USER.format(
            goal=state.goal,
            state=state.render_text(),
            insights=insights,
            trajectory=trajectory,
            stop_reason=stop_reason,
        )
        try:
            response = self.client.json(FINAL_SYSTEM, user, max_tokens=8192)
        except LLMError as exc:
            return FinalAnswer(
                text=_fallback_answer(state, f"LLM final answer failed: {exc}"), source="fallback"
            )
        if response.parse_error:
            first_raw = response.text
            repair_user = (
                f"{user}\n\n"
                "Your previous final answer was truncated or was not valid JSON. Re-emit "
                "the complete answer as one JSON object with an escaped Markdown string "
                "under the `answer` key. Do not use markdown fences around the JSON, and "
                "keep the answer concise enough to finish completely.\n\n"
                "PREVIOUS RAW RESPONSE\n"
                f"{clip(first_raw, 12000)}"
            )
            try:
                repaired = self.client.json(FINAL_SYSTEM, repair_user, max_tokens=8192)
            except LLMError as exc:
                return FinalAnswer(
                    text=_fallback_answer(
                        state,
                        "LLM final answer structured output could not be parsed and the "
                        f"repair request failed: {exc}. Raw response: {clip(first_raw, 1200)}",
                    ),
                    usage=response.usage,
                    source="fallback",
                )
            if repaired.parse_error:
                finish = f"; finish_reason={response.finish_reason}" if response.finish_reason else ""
                return FinalAnswer(
                    text=_fallback_answer(
                        state,
                        "LLM final answer structured output could not be parsed after one "
                        f"repair attempt (first: {response.parse_error}{finish}; "
                        f"repair: {repaired.parse_error}). "
                        f"Raw response: {clip(first_raw, 1200)}",
                    ),
                    usage=repaired.usage or response.usage,
                    source="fallback",
                )
            response = repaired
        answer = str(response.data.get("answer") or "").strip()
        if not answer:
            answer = _fallback_answer(state, "LLM returned no final answer.")
        return FinalAnswer(text=answer, usage=response.usage, source="llm")

    def usage(self) -> dict | None:
        return dict(self.client.total_usage)


def _strip_fence(code: str) -> str:
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return code


def _fallback_answer(state: AnalysisState, note: str) -> str:
    retained = state.retained_insights()
    body = "\n".join(f"- {i.text}" for i in retained) or "- (no retained insights)"
    return f"# Final answer\n\n{note}\n\n## Retained insights\n{body}\n"
