"""Local Python execution for one analysis step.

Deliberately simple for the first slice: one subprocess, a CSV pre-loaded as
``df``, a working directory for artifacts, a timeout. No Docker, no database.

Every step script is written to disk before it runs, so the trace keeps the
exact code that produced each result.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from datajev.types import ExecutionResult

ARTIFACT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html", ".csv", ".json", ".txt"}

PREAMBLE = '''# --- DataJev step script -------------------------------------------------
# Generated code. `df` is pre-loaded: never re-read the CSV yourself.
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)

df = pd.read_csv(os.environ["DATAJEV_CSV"])
print(f"[datajev] df.shape={df.shape}")
# --- end preamble --------------------------------------------------------
'''


@dataclass
class StepExecutor:
    """Runs generated analysis code in a subprocess."""

    csv_path: str
    run_dir: Path
    timeout_s: float = 90.0
    python_executable: str = sys.executable

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).resolve()
        self.csv_path = str(Path(self.csv_path).resolve())
        self.artifacts_dir = self.run_dir / "artifacts"
        self.steps_dir = self.run_dir / "steps"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.steps_dir.mkdir(parents=True, exist_ok=True)

    def run(self, code: str, step_index: int) -> ExecutionResult:
        script_path = (self.steps_dir / f"step_{step_index:02d}.py").resolve()
        script_path.write_text(PREAMBLE + "\n" + code.rstrip() + "\n", encoding="utf-8")

        before = self._snapshot_artifacts()
        env = dict(os.environ)
        env["DATAJEV_CSV"] = self.csv_path
        env["MPLBACKEND"] = "Agg"
        # Keep matplotlib's font cache in the project so every step does not
        # rebuild it in a throwaway temp directory.
        cache_dir = self.run_dir.parent / ".mplcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(cache_dir)
        env["PYTHONWARNINGS"] = "ignore"

        started = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                [self.python_executable, str(script_path)],
                cwd=str(self.artifacts_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr) + f"\n[datajev] step timed out after {self.timeout_s:.0f}s"
            returncode = -1
        duration = time.perf_counter() - started

        artifacts = sorted(self._snapshot_artifacts() - before)
        (self.steps_dir / f"step_{step_index:02d}.out.txt").write_text(
            stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8"
        )

        return ExecutionResult(
            ok=(returncode == 0 and not timed_out),
            stdout=stdout,
            stderr=stderr,
            artifacts=artifacts,
            duration_s=round(duration, 3),
            timed_out=timed_out,
            script_path=str(script_path),
        )

    # ------------------------------------------------------------------ #

    def _snapshot_artifacts(self) -> set[str]:
        found: set[str] = set()
        for path in self.artifacts_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES:
                found.add(str(path.relative_to(self.artifacts_dir)))
        return found


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
