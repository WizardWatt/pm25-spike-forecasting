import numpy as np, pandas as pd

def make_clean(station="KA003", seed=0, shift_2018=40.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", "2019-12-31 23:00", freq="h")
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
