"""Shared test fixtures.

All fixtures are API-free: a tiny CSV plus helpers to build states. Nothing in
the default test suite makes a network call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datajev.dataset import profile_csv
from datajev.state import AnalysisState


@pytest.fixture(scope="session")
def tiny_csv(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A 120-row CSV with a strong global relationship and one decoupled half."""
    rng = np.random.default_rng(7)
    n = 120
    t = np.arange(n)
    common = np.sin(2 * np.pi * t / 40.0) + 0.01 * t
    x = common * 2.0 + rng.normal(0, 0.3, n)
    y = np.where(t < n // 2, common * 1.6, 0.2 * common) + rng.normal(0, 0.3, n)
    frame = pd.DataFrame(
        {
            "day": (pd.Timestamp("2020-01-01") + pd.to_timedelta(t, unit="D")).strftime(
                "%Y-%m-%d"
            ),
            "alpha": np.round(x, 4),
            "beta": np.round(y, 4),
            "region": np.where(t < n // 2, "north", "south"),
        }
    )
    path = tmp_path_factory.mktemp("data") / "tiny.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def tiny_state(tiny_csv: str) -> AnalysisState:
    profile = profile_csv(tiny_csv)
    return AnalysisState(goal="How are alpha and beta related over time?", dataset=profile)
