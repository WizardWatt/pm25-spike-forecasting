"""Step 2: build features, targets and split labels.

Run:  python features.py
Reads data/processed/clean_hourly.csv, writes data/processed/features.csv

Timing rule: a row at time t represents the moment we make a forecast. Features may use
data up to and INCLUDING hour t (the current reading is known at that moment). They must
never use anything after t. The targets look HORIZON hours into the future.
"""
import numpy as np
import pandas as pd

import config as cfg


def usable_extras(clean):
    """Extra pollutants that are complete enough at every station."""
    out = []
    for c in cfg.EXTRA_POLLUTANTS:
        if c not in clean.columns:
            continue
        share = clean.groupby("StationId")[c].apply(lambda s: s.isna().mean())
        if (share < cfg.EXTRA_MAX_MISSING).all():
            out.append(c)
    return out


def build_features(g, extras):
    """Features for ONE station, using only data up to and including each row's time."""
    g = g.sort_values("Datetime").reset_index(drop=True)
    out = pd.DataFrame({"Datetime": g["Datetime"], "StationId": g["StationId"]})
    pm = g["PM2.5"]

    out["pm25_now"] = pm
    for lag in cfg.LAGS:
        out[f"pm25_lag{lag}"] = pm.shift(lag)
    for w in cfg.ROLL_WINDOWS:
        roll = pm.rolling(w, min_periods=w)
        out[f"pm25_mean{w}"] = roll.mean()
        out[f"pm25_std{w}"] = roll.std()
        out[f"pm25_max{w}"] = roll.max()
    out["pm25_trend3"] = pm - pm.shift(3)
    out["pm25_trend6"] = pm - pm.shift(6)

    for c in extras:
        out[f"{c}_now"] = g[c]
        for lag in cfg.EXTRA_LAGS:
            out[f"{c}_lag{lag}"] = g[c].shift(lag)

    hour = g["Datetime"].dt.hour
    month = g["Datetime"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out["dayofweek"] = g["Datetime"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def add_targets(g, feats):
    """Targets for ONE station. These look into the future and are never used as inputs."""
    g = g.sort_values("Datetime").reset_index(drop=True)
    pm = g["PM2.5"]
    H = cfg.HORIZON

    feats = feats.copy()
    feats["y_reg"] = pm.shift(-H)
    future = pd.concat([pm.shift(-k) for k in range(1, H + 1)], axis=1)
    fut_max = future.max(axis=1, skipna=False)   # NaN if any hour in the window is missing
    feats["y_spike"] = (fut_max > cfg.THRESHOLD).astype(float).where(fut_max.notna())
    feats["target_imputed"] = g["pm25_imputed"].shift(-H, fill_value=False).astype(bool)
    return feats


def add_period_labels(df):
    d = df["Datetime"]
    df["split"] = np.select(
        [d < cfg.TRAIN_END, d < cfg.VAL_END, d < cfg.TEST_END],
        ["train", "val", "test"],
        default="after_test",
    )
    df["is_lockdown"] = d >= pd.Timestamp(cfg.LOCKDOWN_START)
    window = pd.Timedelta(days=cfg.DIWALI_WINDOW_DAYS)
    diwali = pd.Series(False, index=df.index)
    for day in cfg.DIWALI_DATES:
        t = pd.Timestamp(day)
        diwali |= (d >= t - window) & (d <= t + window + pd.Timedelta(days=1))
    df["is_diwali"] = diwali
    return df


def main():
    clean = pd.read_csv(cfg.CLEAN_PATH, parse_dates=["Datetime"])
    extras = usable_extras(clean)
    print("Extra pollutants used:", extras or "none")

    parts = []
    for st, g in clean.groupby("StationId"):
        feats = build_features(g, extras)
        parts.append(add_targets(g, feats))
    df = pd.concat(parts, ignore_index=True)
    df = add_period_labels(df)

    feature_cols = [c for c in df.columns if c not in
                    ("Datetime", "StationId", "y_reg", "y_spike", "target_imputed",
                     "split", "is_lockdown", "is_diwali")]
    df["model_ready"] = df[feature_cols + ["y_reg", "y_spike"]].notna().all(axis=1)

    cfg.FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.FEATURES_PATH, index=False)

    ready = df[df["model_ready"]]
    print("\nUsable rows by station and split:")
    print(ready.groupby(["StationId", "split"]).size().unstack(fill_value=0))
    print("\nSpike rate (share of usable rows labelled spike):")
    print(ready.groupby(["StationId", "split"])["y_spike"].mean().unstack().round(3))
    print("\nSpike rows in shift windows:")
    print(ready.groupby("StationId").apply(
        lambda x: pd.Series({"diwali_rows": int(x["is_diwali"].sum()),
                             "diwali_spikes": int(x.loc[x["is_diwali"], "y_spike"].sum()),
                             "lockdown_rows": int(x["is_lockdown"].sum()),
                             "lockdown_spikes": int(x.loc[x["is_lockdown"], "y_spike"].sum())})))
    print(f"\nSaved {cfg.FEATURES_PATH}  ({len(feature_cols)} features)")


if __name__ == "__main__":
    main()