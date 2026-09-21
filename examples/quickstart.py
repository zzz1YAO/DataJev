"""Minimal DataJev example.

    python examples/quickstart.py --offline
    python examples/quickstart.py                      # uses Jev/LLM when keys are set

The offline path needs no API keys at all: the scripted analyst writes and runs
real pandas code, and the heuristic controller decides what to do next.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datajev.agent import RunConfig, build_agent
from datajev.render import Renderer

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "temporal_regimes.csv"
DEFAULT_GOAL = (
    "How are MEANGAM and MEANGBZ related over time, and is a single global "
    "description sufficient to characterize their relationship?"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal DataJev example")
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--offline", action="store_true", help="force heuristic + scripted")
    args = parser.parse_args()

    renderer = Renderer()
    config = RunConfig(
        csv_path=args.csv,
        goal=args.goal,
        out_dir=args.out,
        controller="heuristic" if args.offline else "auto",
        analyst="scripted" if args.offline else "auto",
        max_steps=args.max_steps,
    )

    renderer.banner()
    agent = build_agent(config)
    result = agent.run(
        on_event=lambda name, payload: (
            renderer.run_header(payload)
            if name == "run_start"
            else renderer.decision(payload["step_index"], payload["decision"])
            if name == "decision"
            else renderer.step(payload["step"], payload["insights"], payload.get("analyst", "analyst"))
            if name == "step"
            else None
        ),
    )
    renderer.final(
        answer=result.final_answer.text,
        trace_path=str(result.trace_path),
        stop_reason=result.stop_reason,
        insights=result.insights,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
