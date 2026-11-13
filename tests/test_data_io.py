import sys, os, tempfile; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import config as cfg
import eval_config as C
from data_io import load_clean
from baselines import run
from gru_data import make_splits
from synth import make_clean

from synth import make_features as a_style_features

def test_all():
    feats = pd.concat([a_style_features("KA003"), a_style_features("KA009")], ignore_index=True)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "features.csv"); feats.to_csv(path, index=False)
        clean = load_clean(path)
        assert {"Datetime", "StationId", "PM2.5", "y_reg", "y_spike"} <= set(clean.columns)
        # my split bounds must select exactly A's split labels
        for name, (a, b) in {"train": (C.TRAIN_START, C.TRAIN_END), "val": (C.VAL_START, C.VAL_END),
                             "test": (C.TEST_START, C.TEST_END)}.items():
            mine = ((clean.Datetime >= a) & (clean.Datetime <= b)).sum()
            hers = (clean["split"] == name).sum()
            assert mine == hers, (name, mine, hers)
        preds, _ = run(clean)
        assert set(preds) == {"persistence", "hist_avg", "naive_alarm"}
        d = make_splits(clean, "KA003", "val")
        assert len(d["train"]["X"]) > 0 and len(d["eval"]["X"]) > 0
        # clean_hourly-style file (no targets) gives a clear error
        bad = os.path.join(td, "bad.csv"); feats[["Datetime", "StationId", "pm25_now"]].rename(columns={"pm25_now": "PM2.5"}).to_csv(bad, index=False)
        try:
            load_clean(bad); assert False
        except SystemExit as e:
            assert "features.csv" in str(e)

if __name__ == "__main__":
    test_all(); print("PASS data_io")
