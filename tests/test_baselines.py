import sys, tempfile, os; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from baselines import run, summarize, persistence, write_all, period
from shared_format import SHARED_COLUMNS, load_preds
from synth import make_clean

def test_all():
    clean = pd.concat([make_clean("KA003", 0), make_clean("KA009", 1)], ignore_index=True)
    preds, thrs = run(clean)
    assert set(preds) == {"persistence", "hist_avg", "naive_alarm"}
    for p in preds.values():
        assert list(p.columns) == C.PRED_COLUMNS
        assert p["Datetime"].min() >= pd.Timestamp(C.VAL_START) and p["Datetime"].max() <= pd.Timestamp(C.VAL_END)
    assert set(thrs["naive_alarm"]) == set(C.STATIONS) and all(0 < f < 1 for f in thrs["naive_alarm"].values())
    # persistence = PM2.5 at the issue time
    ka = period(clean[clean.StationId == "KA003"], C.VAL_START, C.VAL_END)
    assert np.allclose(persistence(ka)["y_pred"].dropna(), ka["PM2.5"].dropna())
    # hist avg uses ONLY fit-period data: the 2018 shift (+40) must not leak into its predictions
    ha = preds["hist_avg"]
    assert ha["y_pred"].mean() < ha["y_true"].mean() - 20
    s = summarize(preds, thrs, clean)
    assert s.groupby("station")["n"].nunique().max() == 1            # identical rows per station
    # files in A's format: 2018 rows 'train_only', 2019+ rows 'refit'
    with tempfile.TemporaryDirectory() as td:
        table, _ = write_all(clean, td)
        f = load_preds("naive_alarm", "KA003", td)
        assert list(f.columns) == SHARED_COLUMNS
        assert set(f["model"]) == {"train_only", "refit"}
        assert (pd.to_datetime(f[f.model == "train_only"]["Datetime"]).dt.year == 2018).all()
        assert pd.to_datetime(f[f.model == "refit"]["Datetime"]).min() >= pd.Timestamp(C.TEST_START)
    print(s.to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS baselines")
