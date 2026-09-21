# DataJev

**System-1 control for System-2 data agents.**
Let LLMs analyze. Let Jev decide what to analyze next.

> Status: phase 1 — backend + CLI vertical slice only. No web UI, no benchmark
> numbers, no polished marketing README yet.

## What it does

A CSV and a goal go in. A generative analyst writes and runs each analysis step.
A controller reads the *compressed analytical state* after every step and
decides what to do next: continue, switch direction, verify, or stop.

```text
CSV + goal
    │
    ▼
Analyst (System 2, LLM or scripted) ──► code ──► execution ──► result + insights
    ▲                                                              │
    │                                                    compressed state
    │                                                              │
    └──────────── next typed decision ◄──── Controller (System 1) ──┘
                        continue / switch / verify / stop
                        tool + retain + needs_verification + completeness
```

The controller never sees the CSV, only goal, schema, latest result, retained
insights and recent trajectory.

## Install

```bash
uv sync
```

## Use

```bash
# offline, no API keys: deterministic scripted analyst + rule-based controller
datajev analyze examples/temporal_regimes.csv \
  --goal "How are MEANGAM and MEANGBZ related over time, and is a single global description sufficient?" \
  --controller heuristic --analyst scripted

# Jev controls, an LLM analyzes
export TYPESAFE_API_KEY=...
export DATAJEV_LLM_API_KEY=...        # any OpenAI-compatible endpoint
export DATAJEV_LLM_MODEL=gpt-4o-mini  # or deepseek-flash, etc.
datajev analyze data.csv --goal "What are the most important drivers of revenue?"

# replay a saved run
datajev show runs/20260921-131502-temporal-regimes --steps
```

Without `--controller`/`--analyst`, DataJev picks Jev when `TYPESAFE_API_KEY`
is set and an LLM when a key is available, otherwise it falls back to
`heuristic` + `scripted` so the loop always runs.

| Controller | What it is |
| --- | --- |
| `jev` | TypeSafe Jev. One state, five typed judgments, one HTTP call. |
| `llm` | Baseline: a generative model answers the same five questions. |
| `heuristic` | Deterministic rules over the state (offline fallback, tests). |

| Analyst | What it is |
| --- | --- |
| `llm` | OpenAI-compatible model writes each step and reads its result. |
| `scripted` | Deterministic, schema-driven pandas/matplotlib steps. |

## Runs and traces

Each run writes to `runs/<timestamp>-<name>/`:

```text
trace.json          every decision with full probability distributions
final_answer.md     the closing answer
artifacts/          plots and files produced by the steps
steps/step_XX.py    the exact code that ran, plus its output
```

`trace.json` keeps Jev's raw answers, so `action`/`tool`/`completeness`
distributions and the two Noul probabilities are available for a future UI.

## Development

```bash
uv run pytest          # all tests run without API keys or network
python examples/quickstart.py --offline
python examples/make_temporal_regimes.py   # regenerate the demo CSV
```

## Not in this phase

Web UI, benchmarks, packaging/publishing, richer tools (SQL), multi-agent,
memory, sandboxing.
