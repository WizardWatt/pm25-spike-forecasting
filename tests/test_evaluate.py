import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from evaluate import *

def test_regression():
    r = regression_metrics([1, 2, 3], [2, 2, 5])
    assert abs(r["MAE"] - 1.0) < 1e-9 and abs(r["RMSE"] - np.sqrt(5 / 3)) < 1e-9

def test_spike_counts():
    s = spike_metrics([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (s["TP"], s["FP"], s["FN"], s["TN"]) == (1, 1, 1, 2)
    assert s["recall"] == 0.5 and abs(s["false_alarm_rate"] - 1 / 3) < 1e-9

def test_interval():
    r = interval_metrics([1, 5, 9], [0, 0, 0], [2, 4, 10])
    assert abs(r["coverage"] - 2 / 3) < 1e-9

def test_episode_lead():
    idx = pd.date_range("2019-01-01", periods=30, freq="h")
    pm = pd.Series(50.0, index=idx); pm.iloc[10:12] = 120; pm.iloc[20] = 120   # onsets at 10 and 20
    al = pd.Series(False, index=idx); al.iloc[7] = True                          # alert 3h before onset 10 only
    r = episode_metrics(pm, al)
    assert r["n_episodes"] == 2 and r["episodes_detected"] == 1 and r["mean_lead_h"] == 3

def test_end_to_end_perfect_and_silent():
    idx = pd.date_range("2019-01-01", periods=200, freq="h")
    rng = np.random.default_rng(0)
    pm = pd.Series(rng.uniform(20, 60, 200), index=idx); pm.iloc[[50, 51, 120]] = 150
    ys = (pm[::-1].rolling(6, min_periods=1).max()[::-1].shift(-1) > 90).astype(float)  # any >90 in next 6h
    truth = pd.DataFrame({"Datetime": idx, "StationId": "KA003", "PM2.5": pm.values, "y_reg": pm.shift(-6).values, "y_spike": ys.values})
    perfect = pd.DataFrame({"Datetime": idx, "StationId": "KA003", "y_true": truth.y_reg, "y_pred": truth.y_reg,
                            "lower": truth.y_reg - 1, "upper": truth.y_reg + 1, "spike_prob": truth.y_spike})
    r = evaluate(perfect, truth)["KA003"]
    assert r["MAE"] == 0 and r["recall"] == 1 and r["precision"] == 1 and r["coverage"] == 1
    assert r["episode_recall"] == 1
    silent = perfect.assign(spike_prob=0.0)
    r = evaluate(silent, truth)["KA003"]
    assert r["recall"] == 0 and r["episode_recall"] == 0 and r["false_alarm_rate"] == 0
