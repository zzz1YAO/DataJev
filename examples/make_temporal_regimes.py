"""Generate `examples/temporal_regimes.csv`.

The demo dataset is synthetic on purpose: it reproduces the *structure* of the
DABench temporal-relationship example (strong global correlation, but locally
decoupled regimes) without redistributing any third-party data.

Structure it reproduces:
  * two continuous variables with a strong global Pearson correlation (~0.89)
  * four equal-count time segments, each with ~288 rows
  * one middle-late segment where the relationship nearly disappears (r ~ 0.01)
  * smooth, visually obvious co-movement at the large scale

(The original DABench example reports a global r ~ 0.92; the ceiling here is set
by how much of the record is decoupled, so this fallback lands a little lower
while keeping the regime structure unambiguous.)

Run:  python examples/make_temporal_regimes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260915
N = 1153
SEGMENTS = 4


def build_frame(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(N)

    # Shared large-scale trend: the reason a single global correlation is high.
    common = (
        3.2 * np.sin(2 * np.pi * t / 420.0)
        + 1.4 * np.sin(2 * np.pi * t / 137.0 + 0.7)
        + 0.0035 * t
    )

    meangam = 6.0 + 1.15 * common + rng.normal(0.0, 0.4, N)

    # Segment-specific coupling: segment 3 is nearly decoupled, so MEANGBZ goes
    # flat while MEANGAM keeps moving.
    coupling = np.array([1.8, 1.7, 0.0, 1.6])
    segment = np.minimum((t * SEGMENTS) // N, SEGMENTS - 1)
    slope = coupling[segment]
    offsets = np.array([0.0, 0.4, 1.6, -0.3])
    noise_scale = np.array([0.7, 0.7, 0.25, 0.7])
    meangbz = (
        4.0
        + slope * common
        + offsets[segment]
        + rng.normal(0.0, 1.0, N) * noise_scale[segment]
    )

    # Give the decoupled segment a slow internal wander so it still looks alive.
    meangbz = meangbz + 0.3 * np.sin(2 * np.pi * t / 90.0) * (segment == 2)

    start = pd.Timestamp("2000-01-01")
    true_time = start + pd.to_timedelta(t * 1.27, unit="D")

    frame = pd.DataFrame(
        {
            "TRUE_TIME": true_time.strftime("%Y-%m-%d"),
            "MEANGAM": np.round(meangam, 4),
            "MEANGBZ": np.round(meangbz, 4),
        }
    )
    return frame


def report(frame: pd.DataFrame) -> None:
    r = frame["MEANGAM"].corr(frame["MEANGBZ"])
    rho = frame["MEANGAM"].corr(frame["MEANGBZ"], method="spearman")
    print(f"rows={len(frame)}  global pearson={r:.4f}  spearman={rho:.4f}")
    part = frame.assign(_t=pd.to_datetime(frame["TRUE_TIME"]))
    part["_seg"] = pd.qcut(part["_t"].rank(method="first"), SEGMENTS, labels=["S1", "S2", "S3", "S4"])
    for seg, chunk in part.groupby("_seg", observed=True):
        print(
            f"  {seg}: n={len(chunk):>4}  pearson={chunk['MEANGAM'].corr(chunk['MEANGBZ']):.4f}"
        )


def main() -> None:
    out = Path(__file__).resolve().parent / "temporal_regimes.csv"
    frame = build_frame()
    frame.to_csv(out, index=False)
    print(f"wrote {out}")
    report(frame)


if __name__ == "__main__":
    main()
