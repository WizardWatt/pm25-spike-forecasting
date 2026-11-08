import sys, os, tempfile; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from gru_data import make_splits, predictions_frame
from train_gru import pick_threshold, pos_weight, report
from baselines import load_clean, run
from synth import make_clean

def test_all():
    # threshold picking: well separated scores -> threshold lands between the classes
    y = np.r_[np.zeros(950), np.ones(50)]
    p = np.r_[np.random.default_rng(0).uniform(0, 0.3, 950), np.random.default_rng(1).uniform(0.6, 1, 50)]
    t = pick_threshold(y, p); assert 0.3 <= t <= 0.6
    assert pick_threshold(np.zeros(10), np.random.rand(10)) == 0.5
    assert abs(pos_weight(np.r_[np.zeros(900), np.ones(100)]) - 3.0) < 1e-9
    assert pos_weight(np.r_[np.zeros(9999), np.ones(1)]) == 10.0          # capped

    # glue: fake "GRU" (uses last observed PM2.5 channel) -> shared format -> report vs baselines
    clean = make_clean()
    d = make_splits(clean, "KA003", "val")
    ev, sc = d["eval"], d["scaler"]
    fake_pred = sc.inverse(ev["X"][:, -1, 0])
    fake_prob = np.clip(fake_pred / 180, 0, 1)
    frame = predictions_frame(ev, "KA003", fake_pred, fake_prob)
    assert list(frame.columns) == C.PRED_COLUMNS and len(frame) == len(ev["y_reg"])
    with tempfile.TemporaryDirectory() as td:
        for name, p in run(clean).items():
            p.to_csv(os.path.join(td, f"{name}_val.csv"), index=False)
        tab = report(clean, "KA003", "val", frame, 0.5, td)
    assert set(tab["model"]) == {"gru", "persistence", "hist_avg"} and tab["n"].nunique() == 1
    print(tab.to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS train_gru (torch-free parts)")
