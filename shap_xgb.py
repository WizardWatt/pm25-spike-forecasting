"""SHAP for Person A's XGBoost models: which inputs drive the forecast and the spike score?

Needs (from A's xgb.py):  outputs/models/xgb_reg_{station}.json, xgb_clf_{station}.json, xgb_{station}_meta.json
(meta holds the feature list), plus data/processed/features.csv. pip install shap xgboost.

What it does, per station and per model (reg = PM2.5 at t+6 in ug/m3; clf = spike score in log-odds):
  1. Integrity check: the loaded model must reproduce A's saved 2019 'refit' predictions from the features
     (inputs only, no 2019 truth is used). If not, the model file / feature order / units are not what we think.
  2. SHAP values on 2018 validation rows with real (not imputed) targets.
  3. Additivity check: base value + sum of SHAP == the model's raw output.
  4. Tables: importance by feature and by feature family, and Diwali-week vs other-week importance.
  5. Plots: bar chart and beeswarm.
Features are strongly correlated (lags, rolling stats), so credit is shared between them: read the FAMILY table
first, and don't over-interpret one lag against another.

  python shap_xgb.py                      # both stations, both models
  python shap_xgb.py --station KA003 --kind reg
"""
import argparse, json, os
import numpy as np
import pandas as pd
import eval_config as C
from shared_format import load_preds

FAMILIES = [("current level", ("pm25_now",)), ("lags", ("pm25_lag",)), ("rolling mean", ("pm25_mean",)),
            ("rolling max", ("pm25_max",)), ("rolling std", ("pm25_std",)), ("trend", ("pm25_trend",)),
            ("hour of day", ("hour_",)), ("season", ("month_",)), ("weekday", ("dayofweek", "is_weekend"))]


def family(name):
    for fam, prefixes in FAMILIES:
        if name.startswith(prefixes):
            return fam
    return "other"


def importance_table(sv, feats):
    m = np.abs(np.asarray(sv)).mean(axis=0)
    t = pd.DataFrame({"feature": feats, "family": [family(f) for f in feats], "mean_abs_shap": m})
    t["share"] = t["mean_abs_shap"] / t["mean_abs_shap"].sum()
    return t.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def family_table(imp):
    f = imp.groupby("family", as_index=False)[["mean_abs_shap", "share"]].sum()
    return f.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_compare(sv, feats, mask, label="Diwali week"):
    """Mean |SHAP| per feature inside the mask vs outside it."""
    a = np.abs(np.asarray(sv))
    mask = np.asarray(mask, bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return pd.DataFrame()
    t = pd.DataFrame({"feature": feats, f"{label}": a[mask].mean(axis=0), "other weeks": a[~mask].mean(axis=0)})
    t["ratio"] = t[label] / t["other weeks"].replace(0, np.nan)
    return t.sort_values(label, ascending=False).reset_index(drop=True)


def _load(xgb, kind, path):
    m = xgb.XGBRegressor() if kind == "reg" else xgb.XGBClassifier()
    m.load_model(path)
    return m


def run(station, kind, df, mdir, out_dir, xgb, shap, plots=True):
    meta = json.load(open(os.path.join(mdir, f"xgb_{station}_meta.json")))
    feats = meta["features"]
    path = os.path.join(mdir, f"xgb_{kind}_{station}.json")
    if not os.path.exists(path):
        print(f"(skipping {station} {kind}: {path} not found)")
        return
    model = _load(xgb, kind, path)
    d = df[(df["StationId"] == station) & df["model_ready"].astype(bool)]

    # 1. integrity check against A's saved refit predictions (inputs only)
    pred = load_preds("xgb", station)
    pred = pred[pred["model"] == "refit"].copy()
    pred["Datetime"] = pd.to_datetime(pred["Datetime"])
    chk = d.merge(pred[["Datetime", "y_pred", "spike_prob"]], on="Datetime", how="inner")
    if len(chk):
        mine = model.predict(chk[feats]) if kind == "reg" else model.predict_proba(chk[feats])[:, 1]
        theirs = chk["y_pred"] if kind == "reg" else chk["spike_prob"]
        diff = float(np.max(np.abs(mine - theirs.to_numpy())))
        tol = 1e-3 if kind == "reg" else 1e-4
        print(f"[{station} {kind}] reproduces A's saved refit predictions on {len(chk)} rows: max diff {diff:.2e}"
              + ("" if diff < tol else "   <-- MISMATCH: check feature order / target transform / which model file"))
    else:
        print(f"[{station} {kind}] no refit rows in A's file to check against")

    # 2. SHAP on 2018 validation rows with real targets
    ex = d[(d["split"] == "val") & ~d["target_imputed"].fillna(False).astype(bool)]
    X = ex[feats]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    # 3. additivity check
    base = float(np.ravel(explainer.expected_value)[0])
    margin = model.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    add_err = float(np.max(np.abs(base + sv.sum(axis=1) - margin)))
    print(f"[{station} {kind}] {len(X)} rows; additivity max error {add_err:.2e}; base value {base:.3f}")

    imp = importance_table(sv, feats)
    fam = family_table(imp)
    dw = group_compare(sv, feats, ex["is_diwali"].fillna(False).astype(bool).to_numpy())
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"shap_{station}_{kind}")
    imp.round(4).to_csv(stem + "_importance.csv", index=False)
    fam.round(4).to_csv(stem + "_families.csv", index=False)
    if len(dw):
        dw.round(4).to_csv(stem + "_diwali.csv", index=False)
    unit = "ug/m3" if kind == "reg" else "log-odds"
    print(f"\n{station} {kind}: importance by feature family (mean |SHAP|, {unit})")
    print(fam.round(3).to_string(index=False))
    print(f"\nTop 8 features:")
    print(imp.head(8).round(3).to_string(index=False))
    if len(dw):
        print("\nDiwali week vs other weeks (mean |SHAP|), top 6:")
        print(dw.head(6).round(3).to_string(index=False))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for ptype, suffix in (("bar", "bar"), ("dot", "beeswarm")):
            plt.figure()
            shap.summary_plot(sv, X, plot_type=ptype, show=False, max_display=15)
            plt.title(f"{station} XGBoost {kind} ({unit}), 2018")
            plt.tight_layout(); plt.savefig(f"{stem}_{suffix}.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", nargs="+", default=C.STATIONS)
    ap.add_argument("--kind", nargs="+", default=["reg", "clf"], choices=["reg", "clf"])
    ap.add_argument("--features", default=str(C.FEATURES_PATH))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    try:
        import xgboost as xgb, shap
    except ImportError:
        raise SystemExit("pip install shap xgboost")
    df = pd.read_csv(a.features, parse_dates=["Datetime"])
    out = os.path.join(C.ROOT, "outputs", "reports")
    for st in a.station:
        for k in a.kind:
            run(st, k, df, str(C.MODEL_DIR), out, xgb, shap, plots=not a.no_plots)
