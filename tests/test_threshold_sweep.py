import sys; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from threshold_sweep import sweep

def test_all():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", "2018-12-31 23:00", freq="h")
    pm = pd.Series(50.0 + rng.normal(0, 3, len(idx)), index=idx)
    for s in rng.choice(np.arange(100, len(idx) - 100, 40), 40, replace=False):
        pm.iloc[s:s + 2] = 120
    fmax = pm[::-1].rolling(6, min_periods=1).max()[::-1].shift(-1)
    ys = (fmax > 90).astype(float).where(fmax.notna())
    truth = pd.DataFrame({"Datetime": idx, "StationId": "KA003", "PM2.5": pm.values,
                          "y_reg": pm.shift(-6).values, "y_spike": ys.values})
    good = np.where(ys.fillna(0) == 1, 0.7, 0.0) + rng.uniform(0, 0.3, len(idx))
    pred = pd.DataFrame({"Datetime": idx, "StationId": "KA003", "y_true": truth.y_reg, "y_pred": truth.y_reg,
                         "lower": np.nan, "upper": np.nan, "spike_prob": good})
    r = sweep(pred, truth, "KA003", thresholds=[0.5], n_null=30)
    assert r.episode_recall[0] > 0.9 and r.ep_recall_chance[0] < 0.4, r
    junk = pred.assign(spike_prob=rng.uniform(0, 1, len(idx)))
    r = sweep(junk, truth, "KA003", thresholds=[0.8], n_null=30)
    assert abs(r.episode_recall[0] - r.ep_recall_chance[0]) < 0.25, r   # uninformative alerts ~ chance
    print(sweep(pred, truth, "KA003", thresholds=[0.2, 0.5], n_null=30).to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS threshold_sweep")
