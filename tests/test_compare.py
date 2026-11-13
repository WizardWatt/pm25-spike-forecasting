import sys, tempfile; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from baselines import write_all
from compare_models import compare, threshold_for_far
from shared_format import load_preds, save_model_preds, _path
from data_io import load_clean
from synth import make_features

def test_all():
    feats = pd.concat([make_features("KA003", 0), make_features("KA009", 1)], ignore_index=True)
    with tempfile.TemporaryDirectory() as td:
        path = td + "/features.csv"; feats.to_csv(path, index=False)
        clean = load_clean(path)
        write_all(clean, td)
        # replacing rows of one model label keeps the other label
        f = load_preds("persistence", "KA003", td)
        save_model_preds(f[f.model == "train_only"].copy(), "persistence", "KA003", td)
        assert set(load_preds("persistence", "KA003", td)["model"]) == {"train_only", "refit"}
        t = compare(["persistence", "hist_avg", "naive_alarm", "missing_model"], clean, pred_dir=td)
        assert set(t.model) == {"persistence", "hist_avg", "naive_alarm"}
        assert t.groupby("station")["n"].nunique().max() == 1
        t2 = compare(["naive_alarm"], clean, far=0.05, pred_dir=td)
        assert (t2["false_alarm_rate"] <= 0.05 + 1e-9).all(), t2
        d = load_preds("naive_alarm", "KA003", td)
        assert threshold_for_far(d[d.model == "train_only"], 0.5) <= threshold_for_far(d[d.model == "train_only"], 0.01)
        # --final: 2019 rows only, thresholds fixed on 2018 (same as the validation run), 2020 rows excluded
        tf = compare(["persistence", "hist_avg", "naive_alarm", "missing_model"], clean, far=0.05, pred_dir=td, final=True)
        tv = compare(["persistence", "hist_avg", "naive_alarm"], clean, far=0.05, pred_dir=td)
        assert set(tf["rows"]) == {"2019 test"} and set(tv["rows"]) == {"2018 val"}
        key = ["model", "station"]
        m = tf.merge(tv, on=key, suffixes=("_t", "_v"))
        assert (m["threshold_t"] == m["threshold_v"]).all() and (m["n_t"] != m["n_v"]).all()
        assert tf.groupby("station")["n"].nunique().max() == 1
        # a model with no refit rows is skipped in --final, not scored on nothing
        p = _path("naive_alarm", "KA003", td); d = load_preds("naive_alarm", "KA003", td); d[d.model == "train_only"].to_csv(p, index=False)
        t3 = compare(["persistence", "naive_alarm"], clean, pred_dir=td, final=True)
        assert "naive_alarm" not in set(t3[t3.station == "KA003"]["model"])
        print(t[["model", "station", "n", "MAE", "recall", "false_alarm_rate", "n_episodes", "episode_recall"]].to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS compare")
