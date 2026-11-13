import numpy as np, pandas as pd

def make_clean(station="KA003", seed=0, shift_2018=40.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", "2020-07-31 23:00", freq="h")
    pm = 45 + 15 * np.sin(2 * np.pi * idx.hour / 24) + rng.gamma(2, 8, len(idx))
    pm = pm + np.where(idx.year == 2018, shift_2018, 0)
    pm = pd.Series(pm, index=idx)
    pm.iloc[rng.choice(len(pm), 8000, replace=False)] = np.nan          # random gaps
    pm.iloc[20000:20100] = np.nan                                       # one long gap
    y_reg = pm.shift(-6)
    fmax = pm[::-1].rolling(6, min_periods=1).max()[::-1].shift(-1)
    y_spike = (fmax > 90).astype(float).where(fmax.notna())
    return pd.DataFrame({"Datetime": idx, "StationId": station, "PM2.5": pm.values,
                         "y_reg": y_reg.values, "y_spike": y_spike.values})


def make_features(station="KA003", seed=0):
    """Shaped like A's features.csv: pm25_now, y_reg, y_spike, split, flags, model_ready, target_imputed."""
    import config as cfg
    d = make_clean(station, seed).rename(columns={"PM2.5": "pm25_now"})
    t = d["Datetime"]
    d["split"] = np.select([t < cfg.TRAIN_END, t < cfg.VAL_END, t < cfg.TEST_END], ["train", "val", "test"], "after_test")
    d["is_diwali"] = False
    d["is_lockdown"] = t >= cfg.LOCKDOWN_START
    d["target_imputed"] = False
    d["model_ready"] = d[["pm25_now", "y_reg", "y_spike"]].notna().all(axis=1)
    d["pm25_lag1"] = d["pm25_now"].shift(1)
    return d
