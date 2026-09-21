"""Run traces.

A trace is the full, lossless record of one run: every controller decision with
its probability distributions, every analysis step with its code and output,
every insight, and the final answer. It is written as JSON next to the
generated artifacts so a future UI can replay and visualize the run.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datajev.types import ControllerDecision, Insight, Step, to_dict
from datajev.version import __version__

TRACE_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Trace:
    run_dir: Path
    run_id: str
    goal: str
    config: dict[str, Any]
    dataset: dict[str, Any]
    started_at: str
    steps: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    #: Insight objects are held by reference: the controller judges them *after*
    #: the step that produced them, so they must be serialized at save time.
    insights: list[Insight]
    final_answer: str = ""
    stop_reason: str = ""
    finished_at: str = ""
    retention_fallback: bool = False
    controller_usage: dict[str, Any] | None = None
    analyst_usage: dict[str, Any] | None = None
    errors: list[str] | None = None

    @classmethod
    def create(
        cls,
        run_dir: Path,
        run_id: str,
        goal: str,
        config: dict[str, Any],
        dataset: dict[str, Any],
    ) -> "Trace":
        return cls(
            run_dir=Path(run_dir),
            run_id=run_id,
            goal=goal,
            config=config,
            dataset=dataset,
            started_at=utc_now(),
            steps=[],
            decisions=[],
            insights=[],
            errors=[],
        )

    # ------------------------------------------------------------------ #

    def record_decision(self, step_index: int, decision: ControllerDecision) -> None:
        self.decisions.append(
            {
                "for_step": step_index,
                "recorded_at": utc_now(),
                **to_dict(decision),
            }
        )

    def record_step(self, step: Step, insights: list[Insight]) -> None:
        self.steps.append(step.as_dict())
        self.insights.extend(insights)

    def mark_retention_fallback(self) -> None:
        self.retention_fallback = True

    def set_final(self, answer: str) -> None:
        self.final_answer = answer

    def finish(
        self,
        stop_reason: str,
        controller_usage: dict[str, Any] | None = None,
        analyst_usage: dict[str, Any] | None = None,
    ) -> None:
        self.stop_reason = stop_reason
        self.finished_at = utc_now()
        self.controller_usage = controller_usage
        self.analyst_usage = analyst_usage

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_version": TRACE_VERSION,
            "run_id": self.run_id,
            "datajev_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "goal": self.goal,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stop_reason": self.stop_reason,
            "config": self.config,
            "dataset": self.dataset,
            "steps": self.steps,
            "decisions": self.decisions,
            "insights": [to_dict(insight) for insight in self.insights],
            "final_answer": self.final_answer,
            "retention_fallback": self.retention_fallback,
            "usage": {
                "controller": self.controller_usage,
                "analyst": self.analyst_usage,
            },
            "errors": self.errors or [],
            "artifacts": sorted(
                str(p.relative_to(self.run_dir))
                for p in (self.run_dir / "artifacts").rglob("*")
                if p.is_file()
            ),
        }

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.run_dir / "trace.json"
        trace_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (self.run_dir / "final_answer.md").write_text(
            self.final_answer + "\n", encoding="utf-8"
        )
        return trace_path


def load_trace(path: str | Path) -> dict[str, Any]:
    """Load a saved trace from a run directory or a trace.json path."""
    target = Path(path)
    if target.is_dir():
        target = target / "trace.json"
    if not target.exists():
        raise FileNotFoundError(f"No trace found at {target}")
    return json.loads(target.read_text(encoding="utf-8"))
