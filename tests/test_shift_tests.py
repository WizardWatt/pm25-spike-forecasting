import sys, tempfile; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from baselines import write_all
from intervals import apply_intervals
from shift_tests import run, summarize, windows, TEST_WINDOWS
from shared_format import _path
from data_io import load_clean
from synth import make_features

def test_all():
    feats = pd.concat([make_features("KA003", 0), make_features("KA009", 1)], ignore_index=True)
    t = feats["Datetime"]
    feats["is_diwali"] = ((t >= "2018-11-04") & (t < "2018-11-11")) | ((t >= "2019-10-24") & (t < "2019-10-31"))
    with tempfile.TemporaryDirectory() as td:
        feats.to_csv(td + "/features.csv", index=False)
        clean = load_clean(td + "/features.csv")
        write_all(clean, td)
        for st in C.STATIONS:
            apply_intervals(_path("hist_avg", st, td), "scaled", 0.1, model_dir=td)     # -> hist_avg_ci
        models = ["hist_avg_ci", "persistence", "naive_alarm"]
        # default run: 2018 only, no 2019/2020 window may appear
        a = run(models, pred_dir=td)
        assert set(a["window"]) == {"Diwali 2018", "rest of 2018 (no Diwali)"}
        assert a.groupby(["station", "window"])["n"].nunique().max() == 1            # identical rows per model
        assert a[a.model == "hist_avg_ci"]["coverage"].notna().all() and a[a.model == "persistence"]["coverage"].isna().all()
        # final run: all five windows + three comparisons
        f = run(models, pred_dir=td, final=True)
        assert set(f["window"]) >= TEST_WINDOWS | {"Diwali 2018"}
        s = summarize(f)
        assert set(s["window"]) == {"Diwali 2018", "Diwali 2019", "lockdown 2020"} and len(s) == 3 * 3 * 2
        # a model with no refit rows is skipped, not silently empty
        import pandas as pd2
        p = _path("naive_alarm", "KA003", td); d = pd2.read_csv(p); d[d.model == "train_only"].to_csv(p, index=False)
        f2 = run(models, pred_dir=td, final=True)
        assert "naive_alarm" not in set(f2[f2.station == "KA003"]["model"])
        print(summarize(f).round(2).head(6).to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS shift_tests")
