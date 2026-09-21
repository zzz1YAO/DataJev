"""CSV profiling.

The whole CSV is never handed to the controller or the LLM analyst: only a
compact schema/profile is, which is the point of the architecture (the analyst
computes, the controller reads compressed analytical state).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_TIME_HINTS = ("time", "date", "day", "month", "year", "timestamp", "ts", "period")


@dataclass
class DatasetProfile:
    path: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    numeric_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    missing: dict[str, int] = field(default_factory=dict)
    head: str = ""
    describe: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "numeric_columns": self.numeric_columns,
            "datetime_columns": self.datetime_columns,
            "categorical_columns": self.categorical_columns,
            "missing": self.missing,
            "head": self.head,
            "describe": self.describe,
        }


def _looks_temporal(series: pd.Series, name: str) -> bool:
    sample = series.dropna()
    if sample.empty:
        return False
    sample = sample.head(200)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    ratio = float(parsed.notna().mean())
    name_hint = any(hint in name.lower() for hint in _TIME_HINTS)
    return ratio >= 0.9 and (name_hint or ratio >= 0.99)


def profile_csv(path: str, sample_rows: int = 5) -> DatasetProfile:
    """Read a CSV and describe it compactly."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    numeric_columns = [str(c) for c in df.select_dtypes(include="number").columns]
    datetime_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in df.columns:
        name = str(column)
        if name in numeric_columns:
            continue
        if _looks_temporal(df[column], name):
            datetime_columns.append(name)
        else:
            categorical_columns.append(name)

    missing = {
        str(c): int(n) for c, n in df.isna().sum().items() if int(n) > 0
    }

    describe = df[numeric_columns].describe().to_string() if numeric_columns else ""

    return DatasetProfile(
        path=str(path),
        rows=int(len(df)),
        columns=[str(c) for c in df.columns],
        dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        numeric_columns=numeric_columns,
        datetime_columns=datetime_columns,
        categorical_columns=categorical_columns,
        missing=missing,
        head=df.head(sample_rows).to_string(index=False),
        describe=describe,
    )
