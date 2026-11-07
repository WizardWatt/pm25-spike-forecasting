"""Step 3: tests that features never see the future and targets are correct.

Run:  python test_leakage.py
"""
import numpy as np
import pandas as pd

import config as cfg
from features import build_features, add_targets, add_period_labels, usable_extras


def check_no_future(clean, extras, make_features=build_features):
    """Features at row t must be identical whether or not the data after t exists.

    Rebuild features on truncated data. If any feature for a row before the cut changes,
    that feature was using future data.
    """
    drop = ["Datetime", "StationId"]
    for st, g in clean.groupby("StationId"):
        g = g.sort_values("Datetime").reset_index(drop=True)
        full = make_features(g, extras)
        for frac in (0.3, 0.5, 0.8):
            cut = int(len(g) * frac)
            part = make_features(g.iloc[:cut].copy(), extras)
            a = full.iloc[:cut].drop(columns=drop).reset_index(drop=True)
            b = part.drop(columns=drop).reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
    return True


def leaky_features(g, extras):
    """Deliberately wrong feature set, used to prove the test can catch leakage."""
    out = build_features(g, extras)
    g = g.sort_values("Datetime").reset_index(drop=True)
    out["leak"] = g["PM2.5"].shift(-1)
    return out


def check_targets(clean, extras, n=200, seed=0):
    """Recompute targets by hand for random rows and compare."""
    rng = np.random.default_rng(seed)
    H = cfg.HORIZON
    for st, g in clean.groupby("StationId"):
        g = g.sort_values("Datetime").reset_index(drop=True)
        t = add_targets(g, build_features(g, extras))
        for i in rng.choice(len(g) - H - 1, size=n, replace=False):
            window = g["PM2.5"].iloc[i + 1: i + H + 1].values
            exp_reg = g["PM2.5"].iloc[i + H]
            if np.isnan(exp_reg):
                assert np.isnan(t["y_reg"].iloc[i])
            else:
                assert t["y_reg"].iloc[i] == exp_reg, f"y_reg wrong at {st} row {i}"
            if np.isnan(window).any():
                assert np.isnan(t["y_spike"].iloc[i])
            else:
                assert t["y_spike"].iloc[i] == float(window.max() > cfg.THRESHOLD), \
                    f"y_spike wrong at {st} row {i}"
    return True


def check_hourly_grid(clean):
    for st, g in clean.groupby("StationId"):
        diffs = g.sort_values("Datetime")["Datetime"].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all(), f"{st}: timestamps are not a clean hourly grid"
    return True


def check_splits(clean):
    df = add_period_labels(clean[["Datetime"]].copy())
    ranges = df.groupby("split")["Datetime"].agg(["min", "max"])
    order = [s for s in ("train", "val", "test", "after_test") if s in ranges.index]
    for a, b in zip(order, order[1:]):
        assert ranges.loc[a, "max"] < ranges.loc[b, "min"], f"{a} overlaps {b}"
    print(ranges)
    return True


def main():
    clean = pd.read_csv(cfg.CLEAN_PATH, parse_dates=["Datetime"])
    extras = usable_extras(clean)

    check_hourly_grid(clean)
    print("PASS  hourly grid has no gaps or duplicates")

    check_no_future(clean, extras)
    print("PASS  features do not use future data")

    try:
        check_no_future(clean, extras, make_features=leaky_features)
        raise SystemExit("FAIL  the leakage test did not catch a deliberately leaky feature")
    except AssertionError:
        print("PASS  the leakage test catches a deliberately leaky feature")

    check_targets(clean, extras)
    print("PASS  targets match a hand calculation")

    check_splits(clean)
    print("PASS  splits are ordered and do not overlap")


if __name__ == "__main__":
    main()