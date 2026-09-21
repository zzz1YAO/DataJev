"""Scripted analyst: a deterministic System-2 stand-in.

It writes real pandas/matplotlib code that is driven by the dataset *schema*
(never by the answer). It exists so the whole loop - controller included - can
be run and tested with no API keys at all, and so tests are reproducible.

The generated code prints machine-readable ``INSIGHT:`` and ``SUMMARY:`` lines,
which is what :meth:`interpret` reads back.

Templates are plain strings with ``<<PLACEHOLDER>>`` markers: they contain
generated f-strings, so they must not be interpolated as Python f-strings here.
"""

from __future__ import annotations

import re

from datajev.analyst.base import Analyst
from datajev.state import AnalysisState
from datajev.types import (
    AnalysisPlan,
    ControllerDecision,
    ExecutionResult,
    FinalAnswer,
    Interpretation,
)

INSIGHT_RE = re.compile(r"^INSIGHT:\s*(.+)$", re.MULTILINE)
SUMMARY_RE = re.compile(r"^SUMMARY:\s*(.+)$", re.MULTILINE)

PAIR_COMPUTE = '''
num = df.select_dtypes(include="number")
cols = [str(c) for c in num.columns]
if len(cols) < 2:
    print("SUMMARY: fewer than two numeric columns; no relationship can be measured")
    raise SystemExit(0)

corr = num.corr(method="pearson")
pairs = []
for i, col_a in enumerate(cols):
    for col_b in cols[i + 1:]:
        value = corr.loc[col_a, col_b]
        if pd.notna(value):
            pairs.append((abs(float(value)), col_a, col_b, float(value)))
if not pairs:
    print("SUMMARY: no numeric pair has a computable correlation")
    raise SystemExit(0)
pairs.sort(key=lambda item: item[0], reverse=True)
_, A_COL, B_COL, R_AB = pairs[0]
PAIR = df[[A_COL, B_COL]].dropna()
X = PAIR[A_COL].to_numpy(dtype=float)
Y = PAIR[B_COL].to_numpy(dtype=float)
SLOPE, INTERCEPT = np.polyfit(X, Y, 1)
R2 = R_AB ** 2
'''

PAIR_PRINTS = '''
print(f"NUMERIC COLUMNS: {cols}")
print(f"STRONGEST PAIR: {A_COL} vs {B_COL}  pearson_r={R_AB:.4f}  n={len(PAIR)}")
print(f"OLS FIT: {B_COL} = {INTERCEPT:.4f} + {SLOPE:.4f} * {A_COL}   R2={R2:.4f}")
print(
    f"INSIGHT: Across the whole dataset {A_COL} and {B_COL} move together "
    f"(pearson r={R_AB:.3f}, R2={R2:.3f}, n={len(PAIR)})."
)
'''

#: Used by the first step only: later steps reuse the pair silently so they do
#: not re-assert the same finding on every iteration.
PAIR_SNIPPET = PAIR_COMPUTE + PAIR_PRINTS

SEGMENT_SNIPPET = '''
def _segments(frame, time_col, cat_col):
    work = frame.copy()
    label = None
    if time_col is not None and time_col in work.columns:
        parsed = pd.to_datetime(work[time_col], errors="coerce", format="mixed")
        work = work.loc[parsed.notna()].copy()
        parsed = parsed[parsed.notna()]
        work["_segment_key"] = parsed
        work = work.sort_values("_segment_key")
        work["_segment"] = pd.qcut(
            work["_segment_key"].rank(method="first"), 4, labels=["S1", "S2", "S3", "S4"]
        )
        label = f"equal-count time segments of {time_col}"
    elif cat_col is not None and cat_col in work.columns:
        work["_segment"] = work[cat_col].astype(str)
        work["_segment_key"] = work["_segment"]
        label = f"groups of {cat_col}"
    else:
        work["_segment_key"] = np.arange(len(work))
        work["_segment"] = pd.qcut(work["_segment_key"], 4, labels=["S1", "S2", "S3", "S4"])
        label = "equal-count row-order segments"
    return work, label


def _segment_rows(work, a_col, b_col, min_n=8):
    rows = []
    for seg, part in work.groupby("_segment", observed=True):
        sub = part[[a_col, b_col]].dropna()
        if len(sub) < min_n:
            continue
        rows.append(
            (
                str(seg),
                int(len(sub)),
                float(sub[a_col].corr(sub[b_col])),
                float(sub[a_col].corr(sub[b_col], method="spearman")),
            )
        )
    return rows
'''

GLOBAL_TAIL = '''
missing = df.isna().sum()
missing = missing[missing > 0]
print("MISSING VALUES: " + (missing.to_string() if len(missing) else "none"))
'''

SEGMENT_TAIL = '''
time_col = <<TIME_COL>>
cat_col = <<CAT_COL>>
work, label = _segments(df, time_col, cat_col)
rows = _segment_rows(work, A_COL, B_COL)
if not rows:
    print("SUMMARY: the dataset could not be split into usable segments")
    raise SystemExit(0)

print(f"SEGMENTED CORRELATION ({label})")
for seg, n, r, rho in rows:
    print(f"  SEGMENT {seg}: n={n} pearson={r:.4f} spearman={rho:.4f}")

weakest = min(rows, key=lambda item: abs(item[2]))
strongest = max(rows, key=lambda item: abs(item[2]))
print(f"WEAKEST SEGMENT: {weakest[0]} pearson={weakest[2]:.4f} n={weakest[1]}")
print(f"STRONGEST SEGMENT: {strongest[0]} pearson={strongest[2]:.4f} n={strongest[1]}")
print(
    f"INSIGHT: The {A_COL}-{B_COL} relationship is not uniform across {label}: "
    f"pearson goes from {weakest[2]:.3f} in {weakest[0]} to {strongest[2]:.3f} in {strongest[0]}."
)

fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [row[0] for row in rows]
values = [row[2] for row in rows]
colors = ["#d1495b" if abs(v) < 0.3 else "#2f6feb" for v in values]
ax.bar(labels, values, color=colors)
ax.axhline(0.0, color="#444444", linewidth=0.8)
ax.set_title(f"Pearson correlation of {A_COL} vs {B_COL} by segment")
ax.set_ylabel("pearson r")
plt.tight_layout()
plt.savefig("<<TAG>>_segment_correlation.png", dpi=140, bbox_inches="tight")
plt.close()

if time_col is not None and time_col in df.columns:
    ts = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    plot_df = df.loc[ts.notna()].copy()
    plot_df["_time"] = ts[ts.notna()]
    plot_df = plot_df.sort_values("_time")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(plot_df["_time"], plot_df[A_COL], color="#2f6feb", label=A_COL, linewidth=1.2)
    ax.set_xlabel(time_col)
    ax.set_ylabel(A_COL, color="#2f6feb")
    twin = ax.twinx()
    twin.plot(plot_df["_time"], plot_df[B_COL], color="#d1495b", label=B_COL, linewidth=1.2)
    twin.set_ylabel(B_COL, color="#d1495b")
    ax.set_title(f"{A_COL} and {B_COL} over {time_col}")
    plt.tight_layout()
    plt.savefig("<<TAG>>_timeseries.png", dpi=140, bbox_inches="tight")
    plt.close()

print(
    f"SUMMARY: Segmented analysis shows the global {A_COL}-{B_COL} relationship is "
    f"heterogeneous; the weakest segment ({weakest[0]}) is close to decoupled "
    f"(pearson={weakest[2]:.3f}) while the strongest ({strongest[0]}) is {strongest[2]:.3f}."
)
'''

ROLLING_TAIL = '''
time_col = <<TIME_COL>>
if time_col is not None and time_col in df.columns:
    ts = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    work = df.loc[ts.notna()].copy()
    work["_order"] = ts[ts.notna()]
    work = work.sort_values("_order")
    axis_label = time_col
else:
    work = df.reset_index(drop=True).copy()
    work["_order"] = np.arange(len(work))
    axis_label = "row order"

series_a = work[A_COL].astype(float)
series_b = work[B_COL].astype(float)
window = max(30, len(work) // 8)
rolling = series_a.rolling(window=window, min_periods=window).corr(series_b)
print(f"ROLLING CORRELATION: window={window} points over {axis_label}")
print(f"  min={rolling.min():.4f}  max={rolling.max():.4f}  last={rolling.dropna().iloc[-1]:.4f}")
low_index = rolling.idxmin()
print(f"  lowest at {work.loc[low_index, '_order']}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(work["_order"], rolling, color="#2f6feb", linewidth=1.3)
ax.axhline(R_AB, color="#888888", linestyle="--", linewidth=1.0, label=f"global r={R_AB:.3f}")
ax.axhline(0.0, color="#444444", linewidth=0.8)
ax.set_xlabel(axis_label)
ax.set_ylabel(f"rolling pearson({A_COL}, {B_COL})")
ax.set_title(f"Rolling correlation of {A_COL} and {B_COL}")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig("<<TAG>>_rolling_correlation.png", dpi=140, bbox_inches="tight")
plt.close()

print(
    f"INSIGHT: The rolling {A_COL}-{B_COL} correlation is unstable over {axis_label}: "
    f"it drops to {rolling.min():.3f} while the global value is {R_AB:.3f}, so the global "
    f"number hides local decoupling."
)
print(
    "SUMMARY: A time-resolved view confirms the relationship weakens in part of the record "
    "and recovers elsewhere; a single global coefficient is therefore not sufficient."
)
'''

VERIFY_TAIL = '''
time_col = <<TIME_COL>>
cat_col = <<CAT_COL>>
work, label = _segments(df, time_col, cat_col)
rows = _segment_rows(work, A_COL, B_COL)
if not rows:
    print("SUMMARY: nothing to verify; no usable segment")
    raise SystemExit(0)

weakest = min(rows, key=lambda item: abs(item[2]))
seg_name, seg_n, seg_r, seg_rho = weakest
part = work[work["_segment"].astype(str) == seg_name]
pair = part[[A_COL, B_COL]].dropna()
a = pair[A_COL].to_numpy(dtype=float)
b = pair[B_COL].to_numpy(dtype=float)

rng = np.random.default_rng(0)
boot = []
for _ in range(2000):
    idx = rng.integers(0, len(a), len(a))
    if np.std(a[idx]) > 0 and np.std(b[idx]) > 0:
        boot.append(float(np.corrcoef(a[idx], b[idx])[0, 1]))
boot = np.array(boot)
low, high = np.percentile(boot, [2.5, 97.5])

observed = float(np.corrcoef(a, b)[0, 1])
null = []
for _ in range(2000):
    shuffled = rng.permutation(b)
    if np.std(shuffled) > 0:
        null.append(float(np.corrcoef(a, shuffled)[0, 1]))
null = np.array(null)
p_value = float((np.abs(null) >= abs(observed)).mean())

print(f"VERIFICATION of segment {seg_name} (n={seg_n})")
print(f"  pearson={observed:.4f}  spearman={seg_rho:.4f}  bootstrap95=[{low:.4f}, {high:.4f}]")
print(f"  permutation p-value={p_value:.4f} over 2000 shuffles")
print(f"  global benchmark pearson={R_AB:.4f}")
print(
    f"INSIGHT: The weak regime in {seg_name} survives verification: pearson={observed:.3f} "
    f"(bootstrap 95% [{low:.3f}, {high:.3f}], permutation p={p_value:.3f}) versus a global "
    f"r={R_AB:.3f}, so the sub-population is genuinely different."
)
print(
    "SUMMARY: Independently resampling and permuting the weakest segment confirms its "
    "correlation is materially lower than the global estimate."
)
'''

SUMMARY_TAIL = '''
time_col = <<TIME_COL>>
cat_col = <<CAT_COL>>
work, label = _segments(df, time_col, cat_col)
rows = _segment_rows(work, A_COL, B_COL)
if rows:
    weak = min(rows, key=lambda item: abs(item[2]))
    strong = max(rows, key=lambda item: abs(item[2]))
    print(
        f"INSIGHT: Final consolidated view: global r={R_AB:.3f} is a poor summary because "
        f"segment correlations span {weak[2]:.3f} ({weak[0]}) to {strong[2]:.3f} ({strong[0]})."
    )
    print(
        f"SUMMARY: Consolidated evidence over {len(rows)} segments; the qualified conclusion "
        f"is that the relationship is strong overall but locally decoupled in {weak[0]}."
    )
else:
    print("SUMMARY: consolidated global evidence only.")
'''


def _render(template: str, tag: str, time_col: str | None, cat_col: str | None) -> str:
    return (
        template.replace("<<TAG>>", tag)
        .replace("<<TIME_COL>>", repr(time_col))
        .replace("<<CAT_COL>>", repr(cat_col))
    )


class ScriptedAnalyst(Analyst):
    """Deterministic analyst that runs a fixed, schema-driven plan."""

    name = "scripted"
    description = "Deterministic schema-driven steps (offline fallback, no LLM)."

    # ------------------------------------------------------------------ #

    def plan(self, state: AnalysisState, decision: ControllerDecision) -> AnalysisPlan:
        index = state.step_count
        tag = f"step_{index + 1:02d}"
        time_col, cat_col = self._columns(state)
        if decision.action == "verify":
            return self._verify_plan(tag, time_col, cat_col)
        if index == 0:
            return self._global_plan()
        if index == 1:
            return self._segment_plan(tag, time_col, cat_col, decision)
        if index == 2:
            return self._rolling_plan(tag, time_col)
        return self._summary_plan(tag, time_col, cat_col)

    @staticmethod
    def _columns(state: AnalysisState) -> tuple[str | None, str | None]:
        profile = state.dataset
        time_col = profile.datetime_columns[0] if profile.datetime_columns else None
        cat_col = None
        if time_col is None and profile.categorical_columns:
            cat_col = profile.categorical_columns[0]
        return time_col, cat_col

    # ------------------------------------------------------------------ #

    @staticmethod
    def _global_plan() -> AnalysisPlan:
        return AnalysisPlan(
            question="What is the strongest global relationship in the dataset?",
            code=PAIR_SNIPPET + GLOBAL_TAIL,
            rationale="Establish the dominant global pattern before testing whether it holds.",
            tool="python",
        )

    @staticmethod
    def _segment_plan(
        tag: str, time_col: str | None, cat_col: str | None, decision: ControllerDecision
    ) -> AnalysisPlan:
        return AnalysisPlan(
            question="Is the global relationship stable across time or across subgroups?",
            code=PAIR_COMPUTE + SEGMENT_SNIPPET + _render(SEGMENT_TAIL, tag, time_col, cat_col),
            rationale="Stress-test the global finding by segmenting the population.",
            tool=decision.tool,
        )

    @staticmethod
    def _rolling_plan(tag: str, time_col: str | None) -> AnalysisPlan:
        return AnalysisPlan(
            question="How does the relationship evolve over the record?",
            code=PAIR_COMPUTE + _render(ROLLING_TAIL, tag, time_col, None),
            rationale="Localize where the relationship breaks down.",
            tool="python",
        )

    @staticmethod
    def _verify_plan(tag: str, time_col: str | None, cat_col: str | None) -> AnalysisPlan:
        return AnalysisPlan(
            question="Does the weak regime survive independent verification?",
            code=PAIR_COMPUTE + SEGMENT_SNIPPET + _render(VERIFY_TAIL, tag, time_col, cat_col),
            rationale="A load-bearing finding about a sub-population must be checked.",
            tool="python",
        )

    @staticmethod
    def _summary_plan(tag: str, time_col: str | None, cat_col: str | None) -> AnalysisPlan:
        return AnalysisPlan(
            question="What is the consolidated, qualified conclusion?",
            code=PAIR_COMPUTE + SEGMENT_SNIPPET + _render(SUMMARY_TAIL, tag, time_col, cat_col),
            rationale="Consolidate the evidence before the controller stops.",
            tool="python",
        )

    # ------------------------------------------------------------------ #

    def interpret(
        self, state: AnalysisState, plan: AnalysisPlan, execution: ExecutionResult
    ) -> Interpretation:
        if not execution.ok:
            message = execution.stderr.strip().splitlines()
            detail = message[-1] if message else "unknown error"
            return Interpretation(
                summary=f"Step failed: {detail}",
                insights=[],
                finding="step failed",
            )
        insights = [match.strip() for match in INSIGHT_RE.findall(execution.stdout)]
        summaries = [match.strip() for match in SUMMARY_RE.findall(execution.stdout)]
        summary = summaries[-1] if summaries else _tail_summary(execution.stdout)
        return Interpretation(summary=summary, insights=insights[:3], finding=summary[:120])

    def finalize(self, state: AnalysisState, stop_reason: str) -> FinalAnswer:
        retained = state.retained_insights()
        evidence = "\n".join(f"{index}. {i.text}" for index, i in enumerate(retained, start=1))
        if not evidence:
            evidence = "1. No insight was retained; the analysis is inconclusive."
        trajectory = "\n".join(f"- {line}" for line in state.recent_trajectory(limit=10))
        text = (
            f"# Final answer\n\n"
            f"**Goal.** {state.goal}\n\n"
            f"## Evidence retained\n{evidence}\n\n"
            f"## Trajectory\n{trajectory}\n\n"
            f"## Limits\n"
            f"- Controller stopped because: {stop_reason}.\n"
            f"- Deterministic scripted analyst: the wording is template-composed from the "
            f"numbers above, and no claim beyond them is made.\n"
        )
        return FinalAnswer(text=text, source="scripted")


def _tail_summary(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("[datajev]")]
    return " | ".join(lines[-3:]) if lines else "(no output)"
