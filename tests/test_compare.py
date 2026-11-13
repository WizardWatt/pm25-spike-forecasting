import sys, tempfile; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from baselines import write_all
from compare_models import compare, threshold_for_far
from shared_format import load_preds, save_model_preds
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
        print(t[["model", "station", "n", "MAE", "recall", "false_alarm_rate", "n_episodes", "episode_recall"]].to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS compare")
