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


def test_episodes_only_in_evaluated_period():
    idx = pd.date_range("2018-01-01", "2019-12-31 23:00", freq="h")
    pm = pd.Series(50.0, index=idx)
    pm.loc["2018-06-01 10:00"] = 120      # inside the evaluated period
    pm.loc["2019-06-01 10:00"] = 120      # in a year that was never forecast
    al = pd.Series(False, index=idx)
    al.loc["2018-06-01 07:00"] = True
    r = episode_metrics(pm, al, start="2018-01-01 00:00", end="2018-12-31 23:00")
    assert r["n_episodes"] == 1 and r["episodes_detected"] == 1 and r["mean_lead_h"] == 3
    assert episode_metrics(pm, al)["n_episodes"] == 2          # without the range both count


def test_real_world_extra_columns_and_imputed_rows():
    idx = pd.date_range("2018-01-01", periods=300, freq="h")
    rng = np.random.default_rng(0)
    pm = pd.Series(rng.uniform(20, 60, 300), index=idx); pm.iloc[[100, 101, 200]] = 150
    ys = (pm[::-1].rolling(6, min_periods=1).max()[::-1].shift(-1) > 90).astype(float)
    truth = pd.DataFrame({"Datetime": idx, "StationId": "KA003", "PM2.5": pm.values,
                          "y_reg": pm.shift(-6).values, "y_spike": ys.values})
    pred = pd.DataFrame({"Datetime": idx, "StationId": "KA003", "split": "val", "model": "x",
                         "y_true": truth.y_reg, "y_pred": truth.y_reg, "lower": np.nan, "upper": np.nan,
                         "spike_prob": truth.y_spike, "y_spike": truth.y_spike, "alert": truth.y_spike,
                         "pm25_now": pm.values, "target_imputed": False})
    r = evaluate(pred, truth)["KA003"]              # used to raise KeyError (y_spike collision)
    assert r["recall"] == 1 and r["episode_recall"] == 1
    bad = pred.copy(); bad.loc[bad.index[50:60], "target_imputed"] = True
    bad.loc[bad.index[50:60], "y_pred"] = 9999       # wrong forecasts on imputed targets must not count
    assert evaluate(bad, truth)["KA003"]["MAE"] == 0
    assert evaluate(bad, truth, exclude_imputed=False)["KA003"]["MAE"] > 0


def test_alert_runs_and_episode_lengths():
    idx = pd.date_range("2018-01-01", periods=24 * 14, freq="h")           # 2 weeks
    al = pd.Series(False, index=idx); al.iloc[10:14] = True; al.iloc[100:103] = True   # two runs
    ys = pd.Series(0.0, index=idx); ys.iloc[12] = 1                         # only the first run precedes a spike
    r = alert_run_metrics(al, ys)
    assert r["alert_runs"] == 2 and r["false_alert_runs"] == 1
    assert abs(r["false_alert_runs_per_week"] - 0.5) < 1e-9
    pm = pd.Series(50.0, index=idx); pm.iloc[50:52] = 120; pm.iloc[200:205] = 120   # a 2h and a 5h episode
    a2 = pd.Series(False, index=idx); a2.iloc[47] = True                     # alert only before the short one
    e = episode_metrics(pm, a2)
    assert (e["short_n"], e["short_recall"], e["long_n"], e["long_recall"]) == (1, 1.0, 1, 0.0)
