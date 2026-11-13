"""SHAP for individual alerts, plus a per-row SHAP file the dashboard can read without the model.

Complements shap_xgb.py (which gives the global picture). Needs the same inputs:
  outputs/models/xgb_clf_{station}.json, outputs/models/xgb_{station}_meta.json, data/processed/features.csv,
  outputs/predictions/xgb_{station}.csv.   pip install shap xgboost

What it does, per station (spike classifier only; the alert is what the dashboard has to explain):
  1. Integrity check: the saved model must reproduce the saved 'refit' spike_prob (inputs only, no truth used).
     Stops if not, because explaining a different model than the one that raised the alerts is meaningless.
  2. SHAP values for every 2019+ row (test + after_test). Those are exactly the rows scored by the refit model
     that is saved on disk, so the explanation matches the alert in the prediction file. (2018 rows were scored
     by the train-only model, which is not saved, so they are left out on purpose.)
  3. Additivity check: base + sum of SHAP == model output (log-odds).
  4. Writes outputs/shap/shap_rows_{station}.csv.gz : one row per hour with x_<feature>, shap_<feature>,
     alert_score_logit, base_logit, spike_prob and a short `top_reasons` text.
  5. Picks five single alerts from 2019 BY A FIXED RULE (no hand-picking) and explains each:
       confident_hit : alert, real spike, highest score          typical_hit : alert, real spike, median score
       false_alarm   : alert, no spike, highest score            near_miss   : no alert, real spike, highest score
       typical_miss  : no alert, real spike, median actual PM2.5 (the worst miss is skipped on purpose: extreme
                       readings are often sensor glitches)
     Writes outputs/reports/shap_{station}_cases.csv and a waterfall PNG per case.

How to read it:
  * SHAP is in log-odds of the model's score. Positive = pushes towards "spike". It explains the MODEL, not the
    atmosphere: "pm25_max24 pushed the score up" does not mean the 24h maximum caused the spike.
  * The classifier is trained with class weights, so its score is not a true probability. Say "alert score".
  * Lags and rolling statistics are correlated, so credit is shared between them. Read the top reasons loosely.
  * 2019 is used here only to EXPLAIN already-frozen models. Nothing is tuned on it.

  python shap_explain.py
  python shap_explain.py --station KA003 --no-plots
"""
import argparse, json, os
import numpy as np
import pandas as pd
import eval_config as C
from shared_format import load_preds
from shap_xgb import _load

SHAP_SPLITS = ("test", "after_test")
CASE_ORDER = ["confident_hit", "typical_hit", "false_alarm", "near_miss", "typical_miss"]
CASE_TEXT = {"confident_hit": "most confident hit", "typical_hit": "typical hit (median score)",
             "false_alarm": "most confident false alarm", "near_miss": "near miss (highest score among misses)",
             "typical_miss": "typical miss (median actual PM2.5 among misses)"}


def select_cases(pred):
    """Pick one row per case from the 2019 'refit' rows, by a fixed rule. Returns {case: row}."""
    d = pred.copy().reset_index(drop=True)
    ok = (d["split"] == "test") & d["spike_prob"].notna() & d["y_spike"].notna() \
        & ~d["target_imputed"].fillna(False).astype(bool)
    d = d[ok].reset_index(drop=True)
    hits = d[(d["alert"] == 1) & (d["y_spike"] == 1)]
    false_alarms = d[(d["alert"] == 1) & (d["y_spike"] == 0)]
    misses = d[(d["alert"] == 0) & (d["y_spike"] == 1)]
    out = {}
    if len(hits):
        out["confident_hit"] = hits.loc[hits["spike_prob"].idxmax()]
        med = hits["spike_prob"].median()
        out["typical_hit"] = hits.loc[(hits["spike_prob"] - med).abs().idxmin()]
    if len(false_alarms):
        out["false_alarm"] = false_alarms.loc[false_alarms["spike_prob"].idxmax()]
    if len(misses):
        # NOT the miss with the highest actual PM2.5: that picks the most extreme reading, which is often a sensor glitch
        out["near_miss"] = misses.loc[misses["spike_prob"].idxmax()]
        med = misses["y_true"].median()
        out["typical_miss"] = misses.loc[(misses["y_true"] - med).abs().idxmin()]
    return {k: out[k] for k in CASE_ORDER if k in out}


def reasons(sv_row, x_row, feats, k=3):
    """Short text: the k features with the largest |SHAP|, their value and push (+ towards spike)."""
    sv_row, x_row = np.asarray(sv_row, float), np.asarray(x_row, float)
    order = np.argsort(-np.abs(sv_row))[:k]
    return "; ".join(f"{feats[i]}={x_row[i]:.3g} ({sv_row[i]:+.2f})" for i in order)


def compute_shap(model, X, shap):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    return np.asarray(sv), float(np.ravel(explainer.expected_value)[0])


def explain_station(station, df, pred, model, feats, rows_dir, report_dir, shap, plots=True):
    d = df[(df["StationId"] == station) & df["model_ready"].astype(bool) & df["split"].isin(SHAP_SPLITS)].copy()
    pr = pred[pred["model"] == "refit"][["Datetime", "spike_prob"]].copy()
    pr["Datetime"] = pd.to_datetime(pr["Datetime"])
    pr = pr.rename(columns={"spike_prob": "saved_prob"})
    d = d.merge(pr, on="Datetime", how="inner").reset_index(drop=True)
    if d.empty:
        print(f"[{station}] no 2019+ rows to explain")
        return
    X = d[feats]
    prob = model.predict_proba(X)[:, 1]
    diff = float(np.max(np.abs(prob - d["saved_prob"].to_numpy())))
    print(f"[{station}] reproduces saved refit spike_prob on {len(d)} rows: max diff {diff:.2e}")
    if diff > 1e-4:
        raise SystemExit("MISMATCH: the model file is not the one that produced the prediction file "
                         "(rerun xgb.py, or check feature order). Not writing explanations.")

    sv, base = compute_shap(model, X, shap)
    margin = model.predict(X, output_margin=True)
    add_err = float(np.max(np.abs(base + sv.sum(axis=1) - margin)))
    print(f"[{station}] additivity max error {add_err:.2e}; base score {base:.3f} (log-odds)")

    # per-row file for the dashboard
    xv = X.to_numpy(float)
    meta = d[["Datetime", "StationId", "split", "target_imputed"]].copy()
    meta["spike_prob"] = d["saved_prob"].round(4)
    meta["alert_score_logit"] = np.round(margin, 4)
    meta["base_logit"] = round(base, 4)
    meta["top_reasons"] = [reasons(sv[i], xv[i], feats) for i in range(len(d))]
    xs = pd.DataFrame(np.round(xv, 3), columns=[f"x_{f}" for f in feats])
    ss = pd.DataFrame(np.round(sv, 4), columns=[f"shap_{f}" for f in feats])
    rows = pd.concat([meta.reset_index(drop=True), xs, ss], axis=1)
    os.makedirs(rows_dir, exist_ok=True)
    rows_path = os.path.join(rows_dir, f"shap_rows_{station}.csv.gz")
    rows.to_csv(rows_path, index=False)
    print(f"[{station}] wrote {rows_path} ({os.path.getsize(rows_path) / 1e6:.1f} MB, {len(rows)} rows)")

    # single-alert explanations
    pred2 = pred[pred["model"] == "refit"].copy()
    pred2["Datetime"] = pd.to_datetime(pred2["Datetime"])
    cases = select_cases(pred2)
    os.makedirs(report_dir, exist_ok=True)
    table = []
    for name, row in cases.items():
        idx = d.index[d["Datetime"] == pd.Timestamp(row["Datetime"])]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        order = np.argsort(-np.abs(sv[i]))[:8]
        for rank, j in enumerate(order, 1):
            table.append(dict(case=name, Datetime=row["Datetime"], rank=rank, feature=feats[j],
                              value=round(float(xv[i, j]), 3), shap_logodds=round(float(sv[i, j]), 3),
                              spike_prob=round(float(row["spike_prob"]), 3), alert=int(row["alert"]),
                              actual_spike=int(row["y_spike"]), actual_pm25_t_plus_6=row["y_true"]))
        if plots:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            exp = shap.Explanation(values=sv[i], base_values=base, data=xv[i], feature_names=feats)
            plt.figure()
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.title(f"{station}: {CASE_TEXT[name]} - {row['Datetime']}\n"
                      f"alert={int(row['alert'])}, actual spike={int(row['y_spike'])} (log-odds)", fontsize=9, pad=28)
            plt.savefig(os.path.join(report_dir, f"shap_{station}_case_{name}.png"), dpi=150, bbox_inches="tight")
            plt.close()
    if table:
        t = pd.DataFrame(table)
        t.to_csv(os.path.join(report_dir, f"shap_{station}_cases.csv"), index=False)
        print(f"\n{station}: single-alert explanations (top 3 drivers each)")
        for name in cases:
            s = t[(t["case"] == name) & (t["rank"] <= 3)]
            if len(s):
                head = s.iloc[0]
                print(f"  {CASE_TEXT[name]} @ {head['Datetime']}: score {head['spike_prob']}, "
                      f"actual spike {head['actual_spike']}")
                for _, r in s.iterrows():
                    print(f"      {r['feature']} = {r['value']}  ->  {r['shap_logodds']:+.2f}")
    missing = [c for c in CASE_ORDER if c not in cases]
    if missing:
        print(f"[{station}] no example found for: {', '.join(missing)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", nargs="+", default=C.STATIONS)
    ap.add_argument("--features", default=str(C.FEATURES_PATH))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    try:
        import xgboost as xgb, shap
    except ImportError:
        raise SystemExit("pip install shap xgboost")
    df = pd.read_csv(a.features, parse_dates=["Datetime"])
    for st in a.station:
        meta = json.load(open(os.path.join(str(C.MODEL_DIR), f"xgb_{st}_meta.json")))
        model = _load(xgb, "clf", os.path.join(str(C.MODEL_DIR), f"xgb_clf_{st}.json"))
        explain_station(st, df, load_preds("xgb", st), model, meta["features"],
                        os.path.join(C.ROOT, "outputs", "shap"), os.path.join(C.ROOT, "outputs", "reports"),
                        shap, plots=not a.no_plots)