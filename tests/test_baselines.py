import sys; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from baselines import run, summarize, persistence, HistoricalAverage, period
from synth import make_clean

def test_all():
    clean = pd.concat([make_clean("KA003", 0), make_clean("KA009", 1)], ignore_index=True)
    preds = run(clean)
    for p in preds.values():
        assert list(p.columns) == C.PRED_COLUMNS
        assert p["Datetime"].min() >= pd.Timestamp(C.VAL_START) and p["Datetime"].max() <= pd.Timestamp(C.VAL_END)
    # persistence = PM2.5 at the issue time
    ka = period(clean[clean.StationId == "KA003"], C.VAL_START, C.VAL_END)
    pp = persistence(ka)
    assert np.allclose(pp["y_pred"].dropna(), ka["PM2.5"].dropna())
    # hist avg uses ONLY fit-period data: 2018 shift (+40) must not leak into its predictions
    ha = preds["hist_avg"]
    assert ha["y_pred"].mean() < ha["y_true"].mean() - 20
    # summary runs end to end and both models share row counts after alignment
    s = summarize(preds, clean)
    assert s.groupby("station")["n"].nunique().max() == 1
    print(s.to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS baselines")
