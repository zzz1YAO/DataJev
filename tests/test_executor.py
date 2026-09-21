"""Python executor: artifacts, stdout capture, failure and timeout handling."""

from __future__ import annotations

from pathlib import Path

from datajev.analyst.executor import StepExecutor


def test_executor_runs_code_and_collects_artifacts(tiny_csv, tmp_path):
    executor = StepExecutor(csv_path=tiny_csv, run_dir=tmp_path / "run", timeout_s=60)
    code = (
        "print('rows', len(df))\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot(df['alpha'], df['beta'])\n"
        "plt.savefig('step_01_scatter.png', dpi=60)\n"
        "plt.close()\n"
        "print('mean', round(float(df['alpha'].mean()), 4))\n"
    )
    result = executor.run(code, 1)

    assert result.ok is True
    assert result.timed_out is False
    assert "(120, 4)" in result.stdout
    assert "mean" in result.stdout
    assert result.artifacts == ["step_01_scatter.png"]
    assert (tmp_path / "run" / "artifacts" / "step_01_scatter.png").exists()

    script = Path(result.script_path)
    assert script.exists()
    assert "DATAJEV_CSV" in script.read_text()
    assert (tmp_path / "run" / "steps" / "step_01.out.txt").exists()


def test_executor_reports_failure(tiny_csv, tmp_path):
    executor = StepExecutor(csv_path=tiny_csv, run_dir=tmp_path / "run", timeout_s=30)
    result = executor.run("raise ValueError('boom')", 2)
    assert result.ok is False
    assert "boom" in result.stderr


def test_executor_times_out(tiny_csv, tmp_path):
    executor = StepExecutor(csv_path=tiny_csv, run_dir=tmp_path / "run", timeout_s=0.5)
    result = executor.run("import time\nprint('started', flush=True)\ntime.sleep(5)", 3)
    assert result.timed_out is True
    assert result.ok is False
    assert "timed out" in result.stderr


def test_executor_only_reports_new_artifacts(tiny_csv, tmp_path):
    executor = StepExecutor(csv_path=tiny_csv, run_dir=tmp_path / "run", timeout_s=30)
    executor.run("import matplotlib.pyplot as plt\nplt.plot([1,2],[2,3])\nplt.savefig('first.png')\nplt.close()", 1)
    result = executor.run("print('no new artifact')", 2)
    assert result.artifacts == []
