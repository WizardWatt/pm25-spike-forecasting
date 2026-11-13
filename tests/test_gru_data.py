import sys; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from gru_data import make_splits, predictions_frame
from synth import make_clean

def test_all():
    clean = make_clean()
    d = make_splits(clean, "KA003", "val", lookback=24)
    tr, ev, sc = d["train"], d["eval"], d["scaler"]
    assert tr["X"].shape[1:] == (24, 10) and len(tr["X"]) == len(tr["times"]) == len(tr["y_reg"])
    # no leakage: train targets stay inside train period, eval issue times inside 2018
    assert tr["times"].max() + pd.Timedelta(hours=6) <= pd.Timestamp(C.TRAIN_END)
    assert ev["times"].min() >= pd.Timestamp(C.VAL_START) and ev["times"].max() <= pd.Timestamp(C.VAL_END)
    # scaler fitted on train only (2018 is shifted +40, so full-data mean would be clearly higher)
    c = clean.set_index("Datetime")["PM2.5"]
    assert abs(sc.mean - c[C.TRAIN_START:C.TRAIN_END].mean()) < 1e-6
    assert sc.mean < c.mean() - 3
    # window content: last step is t, first is t-23h, target is t+6h
    s = clean.set_index("Datetime")
    for k in np.random.default_rng(1).choice(len(ev["times"]), 50, replace=False):
        t = ev["times"][k]
        x_last, x_first = ev["X"][k, -1, 0], ev["X"][k, 0, 0]
        obs_last, obs_first = ev["X"][k, -1, 1], ev["X"][k, 0, 1]
        if obs_last: assert abs(sc.inverse(x_last) - s.loc[t, "PM2.5"]) < 1e-3
        if obs_first: assert abs(sc.inverse(x_first) - s.loc[t - pd.Timedelta(hours=23), "PM2.5"]) < 1e-3
        assert abs(ev["y_reg"][k] - s.loc[t + pd.Timedelta(hours=6), "PM2.5"]) < 1e-3
    # gap handling
    assert tr["miss_frac"].max() <= 0.25 and ev["miss_frac"].max() <= 0.25
    gap_times = pd.date_range(clean.Datetime[20010], periods=60, freq="h")
    assert not set(gap_times) & set(tr["times"])
    # test mode: scaler fit on 2015-18, eval 2019
    t_ = make_splits(clean, "KA003", "test")
    assert t_["eval"]["times"].min() >= pd.Timestamp(C.TEST_START)
    assert t_["train"]["times"].max() + pd.Timedelta(hours=6) <= pd.Timestamp(C.REFIT_END)
    # output frame format
    f = predictions_frame(ev, "KA003", ev["y_reg"], np.zeros(len(ev["y_reg"])))
    assert list(f.columns) == C.PRED_COLUMNS

if __name__ == "__main__":
    test_all(); print("PASS gru_data")
