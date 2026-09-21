"""DataJev command line interface.

    datajev analyze data.csv --goal "What drives revenue?"
    datajev show runs/20260101-120000-data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from datajev import __version__
from datajev.agent import RunConfig, build_agent
from datajev.controllers.base import ControllerError
from datajev.render import Renderer
from datajev.trace import load_trace

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _common_parser() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--no-color",
        action="store_true",
        default=argparse.SUPPRESS,
        help="disable ANSI colors",
    )
    return common


def _load_dotenv() -> None:
    """Read a local .env if python-dotenv is installed (never overrides env)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datajev",
        description="System-1 control for System-2 data agents.",
    )
    parser.add_argument("--version", action="version", version=f"datajev {__version__}")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    subparsers = parser.add_subparsers(dest="command")
    common = _common_parser()

    analyze = subparsers.add_parser(
        "analyze", help="run an analysis loop on a CSV", parents=[common]
    )
    analyze.add_argument("csv", help="path to a CSV file")
    analyze.add_argument("--goal", required=True, help="the analytical goal, in natural language")
    analyze.add_argument(
        "--controller",
        default="auto",
        choices=["auto", "jev", "llm", "heuristic"],
        help="control policy (default: jev when TYPESAFE_API_KEY is set, else heuristic)",
    )
    analyze.add_argument(
        "--analyst",
        default="auto",
        choices=["auto", "llm", "scripted"],
        help="analyst backend (default: llm when an LLM key is set, else scripted)",
    )
    analyze.add_argument("--model", default=None, help="LLM model for the analyst / LLM controller")
    analyze.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    analyze.add_argument("--jev-model", default="jev-latest", help="TypeSafe model id")
    analyze.add_argument("--max-steps", type=int, default=6, help="maximum analysis steps")
    analyze.add_argument("--min-steps", type=int, default=1, help="minimum analysis steps")
    analyze.add_argument("--timeout", type=float, default=90.0, help="per-step timeout in seconds")
    analyze.add_argument("--out", default="runs", help="directory for run traces (default: runs)")
    analyze.add_argument("--run-name", default=None, help="name suffix for the run directory")
    analyze.add_argument("--quiet", action="store_true", help="only print the final answer and run dir")
    analyze.add_argument(
        "--json",
        dest="json_summary",
        action="store_true",
        help="print a JSON summary of the run to stdout",
    )

    show = subparsers.add_parser("show", help="print a saved run trace", parents=[common])
    show.add_argument("run", help="run directory or trace.json path")
    show.add_argument(
        "--steps", action="store_true", help="include full step stdout and decision distributions"
    )
    return parser


# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "show":
        return _cmd_show(args)
    parser.print_help()
    return EXIT_USAGE


def _cmd_analyze(args: argparse.Namespace) -> int:
    renderer = Renderer(color=False if args.no_color else None)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        renderer.error(f"CSV not found: {csv_path}")
        return EXIT_USAGE

    config = RunConfig(
        csv_path=str(csv_path),
        goal=args.goal,
        out_dir=args.out,
        controller=args.controller,
        analyst=args.analyst,
        model=args.model,
        base_url=args.base_url,
        jev_model=args.jev_model,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        timeout_s=args.timeout,
        run_name=args.run_name,
    )

    try:
        agent = build_agent(config)
    except (ControllerError, ValueError) as exc:
        renderer.error(str(exc))
        return EXIT_ERROR

    if not args.quiet and not args.json_summary:
        renderer.banner()

    def on_event(name: str, payload: dict[str, Any]) -> None:
        if args.quiet or args.json_summary:
            return
        if name == "run_start":
            renderer.run_header(payload)
        elif name == "decision":
            renderer.decision(payload["step_index"], payload["decision"])
        elif name == "step":
            renderer.step(payload["step"], payload["insights"], payload.get("analyst", "analyst"))
        elif name == "controller_error":
            renderer.error(payload["message"])

    try:
        result = agent.run(on_event=on_event)
    except KeyboardInterrupt:
        renderer.error("interrupted")
        return EXIT_ERROR
    except Exception as exc:  # surfaced to the user, not swallowed
        if args.json_summary:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            renderer.error(f"run failed: {exc}")
        return EXIT_ERROR

    if args.json_summary:
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_dir": str(result.run_dir),
                    "trace_path": str(result.trace_path),
                    "controller": result.controller,
                    "analyst": result.analyst,
                    "steps": result.steps,
                    "stop_reason": result.stop_reason,
                    "retained_insights": [i.text for i in result.insights if i.retained],
                    "final_answer": result.final_answer.text,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK

    if args.quiet:
        print(result.final_answer.text)
        print(f"\nrun dir: {result.run_dir}")
        return EXIT_OK

    renderer.final(
        answer=result.final_answer.text,
        trace_path=str(result.trace_path),
        stop_reason=result.stop_reason,
        insights=result.insights,
    )
    return EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    renderer = Renderer(color=False if args.no_color else None)
    try:
        trace = load_trace(args.run)
    except FileNotFoundError as exc:
        renderer.error(str(exc))
        return EXIT_USAGE

    renderer.line()
    renderer.line(f"run_id      : {trace.get('run_id')}")
    renderer.line(f"goal        : {trace.get('goal')}")
    config = trace.get("config") or {}
    renderer.line(f"controller  : {config.get('controller')}")
    renderer.line(f"analyst     : {config.get('analyst')}")
    renderer.line(f"stop reason : {trace.get('stop_reason')}")
    renderer.line(f"steps       : {len(trace.get('steps', []))}")
    renderer.line(f"started     : {trace.get('started_at')}  finished: {trace.get('finished_at')}")
    renderer.line()

    for entry in trace.get("decisions", []):
        if not args.steps:
            action = entry.get("action")
            completeness = entry.get("completeness", 0.0)
            renderer.line(
                f"  decision for step {entry.get('for_step'):>2}: {action:<8} "
                f"tool={entry.get('tool'):<13} completeness={completeness:.2f}"
            )
            continue
        renderer.line(f"DECISION for step {entry.get('for_step')}")
        renderer.line(f"  action probabilities : {_fmt_probs(entry.get('action_probabilities'))}")
        renderer.line(f"  tool probabilities   : {_fmt_probs(entry.get('tool_probabilities'))}")
        renderer.line(f"  retain_insight       : {entry.get('retain_insight')}")
        renderer.line(f"  needs_verification   : {entry.get('needs_verification')}")
        renderer.line(f"  completeness         : {entry.get('completeness')}")
        renderer.line(f"  completeness probs   : {_fmt_probs(entry.get('completeness_probabilities'))}")
        renderer.line(f"  confidence           : action={entry.get('action_confidence')} tool={entry.get('tool_confidence')}")
        renderer.line(f"  model/latency        : {entry.get('model')} / {entry.get('latency_s')}s")
        renderer.line()

    if args.steps:
        for step in trace.get("steps", []):
            renderer.line(f"STEP {step.get('index')}: {step.get('question')}")
            renderer.line(f"  action={step.get('action')} tool={step.get('tool')} ok={step.get('ok')}")
            renderer.line("  code:")
            for line in (step.get("code") or "").splitlines():
                renderer.line("    " + line)
            renderer.line("  stdout:")
            for line in (step.get("stdout") or "").splitlines():
                renderer.line("    " + line)
            renderer.line()
    else:
        for step in trace.get("steps", []):
            renderer.line(f"  step {step.get('index'):>2}: [{step.get('action')}/{step.get('tool')}] {step.get('question')}")
    renderer.line()
    renderer.line("RETAINED INSIGHTS")
    for insight in trace.get("insights", []):
        if insight.get("retained"):
            renderer.line(f"  • {insight.get('text')}")
    renderer.line()
    renderer.line("FINAL ANSWER")
    renderer.line((trace.get("final_answer") or "").strip())
    renderer.line(f"\nartifacts: {trace.get('artifacts')}")
    return EXIT_OK


def _fmt_probs(probabilities: dict[str, float] | None) -> str:
    if not probabilities:
        return "(none)"
    return "  ".join(f"{k}={v:.3f}" for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
