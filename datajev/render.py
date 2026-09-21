"""Terminal rendering.

Plain ANSI, no UI framework: the point is to make the controller's probability
distributions visible in a terminal, the same way a future web UI will.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from datajev.types import ControllerDecision, Insight, Step

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
GREY = "\033[90m"

ACTION_COLORS = {
    "continue": CYAN,
    "switch": MAGENTA,
    "verify": YELLOW,
    "stop": GREEN,
    "other": GREY,
}


def _supports_color(stream: TextIO) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


class Renderer:
    """Small, dependency-free renderer for one run."""

    def __init__(self, color: bool | None = None, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self.color = _supports_color(self.stream) if color is None else color
        self.indent = "  "

    # ------------------------------------------------------------------ #

    def _paint(self, text: str, *codes: str) -> str:
        if not self.color:
            return text
        return "".join(codes) + text + RESET

    def line(self, text: str = "") -> None:
        print(text, file=self.stream)

    def rule(self, width: int = 62) -> None:
        self.line(self._paint("─" * width, GREY))

    def bar(self, probability: float, width: int = 18) -> str:
        probability = max(0.0, min(1.0, float(probability)))
        filled = int(round(probability * width))
        return "█" * filled + "░" * (width - filled)

    # ------------------------------------------------------------------ #

    def banner(self) -> None:
        self.line()
        self.line(self._paint("⚡ DataJev", BOLD))
        self.line(self._paint("System-1 control for System-2 data agents", DIM))

    def run_header(self, payload: dict[str, Any]) -> None:
        dataset = payload["dataset"]
        config = payload["config"]
        self.line()
        self.line(self._paint("GOAL", BOLD))
        self.line(f"  {payload['goal']}")
        self.line()
        self.line(self._paint("DATA", BOLD))
        self.line(
            f"  {dataset.rows} rows × {len(dataset.columns)} columns  "
            f"({', '.join(dataset.columns[:8])}{'…' if len(dataset.columns) > 8 else ''})"
        )
        self.line()
        self.line(self._paint("CONTROL", BOLD))
        self.line(f"  controller : {payload['controller']}")
        self.line(f"  analyst    : {payload['analyst']}")
        self.line(f"  max steps  : {config.max_steps}")
        self.line(f"  run dir    : {payload['run_dir']}")
        self.line()

    # ------------------------------------------------------------------ #

    def decision(self, step_index: int, decision: ControllerDecision) -> None:
        label = self._paint(f"CONTROLLER ({decision.source.upper()})", BOLD)
        self.line(f"{label} {self._paint(f'· for step {step_index:02d}', GREY)}")
        action_color = ACTION_COLORS.get(decision.action, CYAN)
        self.line(
            self.indent
            + self._paint("NEXT ACTION", DIM)
            + "  "
            + self._paint(decision.action.upper(), action_color, BOLD)
        )
        for option, probability in _sorted_probs(decision.action_probabilities):
            self.line(self.indent * 2 + self._prob_row(option, probability, 16))
        if decision.tool_probabilities:
            self.line(self.indent + self._paint("TOOL", DIM))
            for option, probability in _sorted_probs(decision.tool_probabilities):
                self.line(self.indent * 2 + self._prob_row(option, probability, 16))
        self.line(
            self.indent
            + self._paint("Retain insight", DIM)
            + f"          {decision.retain_insight:.2f}"
        )
        self.line(
            self.indent
            + self._paint("Needs verification", DIM)
            + f"      {decision.needs_verification:.2f}"
        )
        self.line(
            self.indent
            + self._paint("Analysis completeness", DIM)
            + f"   {decision.completeness:.2f} / 3"
        )
        if decision.completeness_probabilities:
            for level, probability in _sorted_probs(
                decision.completeness_probabilities, by_key=True
            ):
                legend = decision.completeness_legend.get(level, "")
                self.line(
                    self.indent * 2
                    + self._prob_row(f"L{level}", probability, 16)
                    + (self._paint(f"  {legend[:44]}", GREY) if legend else "")
                )
        for note in decision.notes:
            self.line(self.indent + self._paint(f"note: {note}", GREY))
        if decision.latency_s:
            self.line(
                self.indent
                + self._paint(
                    f"{decision.model or ''} · {decision.latency_s * 1000:.0f} ms", GREY
                )
            )

    def _prob_row(self, label: str, probability: float, width: int) -> str:
        return f"{label:<16}{self.bar(probability, width)}  {probability * 100:5.1f}%"

    # ------------------------------------------------------------------ #

    def step(self, step: Step, insights: list[Insight], analyst: str = "analyst") -> None:
        self.rule()
        status = self._paint("ok", GREEN) if step.ok else self._paint("failed", RED)
        self.line(
            self._paint(f"STEP {step.index:02d}", BOLD)
            + f" · {step.question} "
            + self._paint(f"[{step.action}/{step.tool}] {status} {step.duration_s:.1f}s", GREY)
        )
        if step.interpretation and step.interpretation.summary:
            self.line(self.indent + self._paint(analyst.upper(), DIM))
            self.line(self.indent + "→ " + _wrap(step.interpretation.summary, 96, self.indent + "  "))
        for insight in insights:
            marker = "•" if insight.kind == "finding" else "✓"
            self.line(self.indent + f"{marker} {insight.text}")
        if step.artifacts:
            self.line(
                self.indent
                + self._paint("artifacts: " + ", ".join(step.artifacts), GREY)
            )
        if not step.ok and step.error:
            self.line(self.indent + self._paint("error: " + step.error, RED))

    # ------------------------------------------------------------------ #

    def final(self, answer: str, trace_path: str, stop_reason: str, insights: list[Insight]) -> None:
        self.rule()
        retained = [i for i in insights if i.retained]
        self.line(
            self._paint("RETENTION", BOLD)
            + f"  {len(retained)} retained / {len(insights)} proposed"
        )
        for insight in retained:
            probability = (
                f" (retain p={insight.retain_probability:.2f})"
                if insight.retain_probability is not None
                else ""
            )
            self.line(self.indent + f"• {insight.text}{self._paint(probability, GREY)}")
        self.rule()
        self.line(self._paint("FINAL ANSWER", BOLD))
        self.line(answer.strip())
        self.rule()
        self.line(self._paint(f"stop reason : {stop_reason}", DIM))
        self.line(self._paint(f"trace       : {trace_path}", DIM))

    def error(self, message: str) -> None:
        print(self._paint(f"error: {message}", RED), file=sys.stderr)


def _sorted_probs(
    probabilities: dict[str, float], by_key: bool = False
) -> list[tuple[str, float]]:
    items = [(str(k), float(v)) for k, v in (probabilities or {}).items()]
    if by_key:
        return sorted(items, key=lambda kv: kv[0])
    return sorted(items, key=lambda kv: (-kv[1], kv[0]))


def _wrap(text: str, width: int, subsequent_indent: str) -> str:
    import textwrap

    return textwrap.fill(text, width=width, subsequent_indent=subsequent_indent)
