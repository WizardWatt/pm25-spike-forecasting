"""Compare saved prediction files (outputs/predictions/{model}_{station}.csv) on IDENTICAL rows.

Scores the 2018 validation rows only (model == 'train_only'); 2019 stays untouched until the final run.
Each model is scored at its own saved alert threshold (the one implied by its `alert` column), or, with
--far 0.05, at the lowest threshold whose false-alarm rate on these rows is at most 5%, which puts every
model on the same footing.

  python compare_models.py
  python compare_models.py --far 0.05 --models xgb gru persistence
"""
import argparse, os
import numpy as np
import pandas as pd
import eval_config as C
from data_io import load_clean
from evaluate import evaluate, common_rows
from shared_format import load_preds, _path

DEFAULT_MODELS = ["persistence", "hist_avg", "naive_alarm", "xgb", "gru"]
COLS = ["model", "station", "threshold", "n", "base_rate", "MAE", "RMSE", "coverage", "mean_width", "recall", "precision", "false_alarm_rate",
        "false_alert_runs_per_week", "n_episodes", "episode_recall", "short_n", "short_recall", "long_n", "long_recall"]


def implied_threshold(df):
    a = df.loc[df["alert"] == 1, "spike_prob"]
    return float(a.min()) if len(a) else 1.0


def threshold_for_far(df, far):
    """Lowest probability cut-off whose row-level false-alarm rate is <= far (scored rows only)."""
    d = df[(df["y_spike"] == 0) & df["spike_prob"].notna() & ~df["target_imputed"].fillna(False).astype(bool)]
    negs = np.sort(d["spike_prob"].to_numpy())
    cand = np.unique(df["spike_prob"].dropna().to_numpy())
    fars = 1 - np.searchsorted(negs, cand, side="left") / max(len(negs), 1)
    ok = cand[fars <= far]
    return float(ok.min()) if len(ok) else float(cand.max()) + 1e-9


def compare(models, clean, stations=None, far=None, pred_dir=None, label="train_only"):
    rows = []
    for st in stations or C.STATIONS:
        frames = {}
        for m in models:
            if not os.path.exists(_path(m, st, pred_dir)):
                print(f"(skipping {m} / {st}: no file)")
                continue
            df = load_preds(m, st, pred_dir)
            frames[m] = df[df["model"] == label]
        for m, df in common_rows(frames).items():
            thr = threshold_for_far(df, far) if far else implied_threshold(df)
            r = evaluate(df, clean, prob_threshold=thr)[st]
            rows.append({"model": m, "station": st, "threshold": round(thr, 3),
                         "base_rate": (r["TP"] + r["FN"]) / max(r["TP"] + r["FN"] + r["FP"] + r["TN"], 1), **r})
    return pd.DataFrame(rows)[COLS].round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--far", type=float, default=None)
    ap.add_argument("--clean", default=str(C.FEATURES_PATH))
    a = ap.parse_args()
    print(compare(a.models, load_clean(a.clean), far=a.far).to_string(index=False))
