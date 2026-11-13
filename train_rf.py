"""Step 4b: Random Forest regression (PM2.5 six hours ahead) and spike classifier.

Run:  python train_rf.py                      (both stations, 20 search trials each)
      python train_rf.py --stations KA003 --search 10

Same protocol as the XGBoost script, so the comparison is fair:
  1. Train on 2015-2017, tune on 2018 (random search, chosen by 2018 score).
  2. Pick the alert threshold on 2018 only (F2, favours recall).
  3. Refit on 2015-2018 with the chosen settings, then predict 2019 and 2020 once.
  4. 2018 predictions come from the train-only model, so Diwali 2018 stays held out.

Outputs (in outputs/):
  predictions/rf_<station>.csv   same columns as the XGBoost files
  models/                        saved models (compressed) and chosen settings
  reports/rf_<station>_*.csv     metric tables and feature importance
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve

import config as cfg

OUT = cfg.ROOT / "outputs"
SEED = 42
N_TREES = 300
NON_FEATURES = {"Datetime", "StationId", "y_reg", "y_spike", "target_imputed",
                "split", "is_lockdown", "is_diwali", "model_ready"}
DEFAULT_PARAMS = dict(max_depth=12, min_samples_leaf=10, max_features=0.5)


def sample_params(rng):
    return dict(
        max_depth=int(rng.choice([6, 8, 12, 16, 24])),
        min_samples_leaf=int(rng.choice([3, 5, 10, 20, 50])),
        max_features=float(rng.choice([0.33, 0.5, 0.8, 1.0])),
    )


def make_model(kind, params):
    common = dict(n_estimators=N_TREES, random_state=SEED, n_jobs=-1, **params)
    if kind == "reg":
        return RandomForestRegressor(**common)
    # balanced_subsample plays the role of scale_pos_weight in the XGBoost script
    return RandomForestClassifier(class_weight="balanced_subsample", **common)


def score_model(kind, m, Xva, yva):
    """Lower is better for 'reg' (val MAE). Higher is better for 'clf' (val PR-AUC)."""
    if kind == "reg":
        return float(np.abs(m.predict(Xva) - yva).mean())
    return float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))


def search(kind, Xtr, ytr, Xva, yva, n_iter, rng):
    """Random search. Trial 0 is the default setting. Returns the best settings."""
    best = None
    for i in range(n_iter):
        params = dict(DEFAULT_PARAMS) if i == 0 else sample_params(rng)
        m = make_model(kind, params)
        m.fit(Xtr, ytr)
        score = score_model(kind, m, Xva, yva)
        better = best is None or (score < best["score"] if kind == "reg" else score > best["score"])
        if better:
            best = dict(params=params, score=score)
            print(f"  [{kind}] trial {i + 1}/{n_iter}: new best val score {score:.4f} "
                  f"(depth {params['max_depth']}, leaf {params['min_samples_leaf']}, "
                  f"max_features {params['max_features']})")
    return best


def pick_threshold(y, p, beta=2.0):
    """Alert threshold that maximises F-beta on validation (beta=2 favours recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta ** 2) * prec * rec / np.maximum(beta ** 2 * prec + rec, 1e-12)
    return float(thr[int(np.nanargmax(f))])


def spike_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return dict(rows=len(y), positives=int(y.sum()), base_rate=y.mean(),
                recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                false_alarm_rate=fp / max(fp + tn, 1), alarms=int(pred.sum()),
                pr_auc=average_precision_score(y, p) if y.sum() > 0 else np.nan)


def evaluate(preds, thr):
    """Metric tables by period. Rows whose target was interpolated are excluded."""
    p = preds[~preds["target_imputed"].astype(bool)]
    sp, dw, lk = p["split"], p["is_diwali"].astype(bool), p["is_lockdown"].astype(bool)
    subsets = {
        "val 2018 (train-only model)": sp == "val",
        "test 2019 (refit model)": sp == "test",
        "Diwali 2018 (train-only model)": (sp == "val") & dw,
        "Diwali 2019 (refit model)": (sp == "test") & dw,
        "lockdown 2020 (refit model)": lk,
    }
    reg_rows, spike_rows = {}, {}
    for name, mask in subsets.items():
        s = p[mask]
        if len(s) == 0:
            continue
        err = s["y_pred"] - s["y_true"]
        reg_rows[name] = dict(rows=len(s), mae=err.abs().mean(),
                              rmse=float(np.sqrt((err ** 2).mean())), bias=err.mean(),
                              persistence_mae=(s["pm25_now"] - s["y_true"]).abs().mean())
        spike_rows[name] = spike_metrics(s["y_spike"].values, s["spike_prob"].values, thr)
    return pd.DataFrame(reg_rows).T, pd.DataFrame(spike_rows).T


def run_station(df, st, n_iter, rng):
    print(f"\n=== {st} ===")
    d = df[(df["StationId"] == st) & df["model_ready"]].copy()
    feats = [c for c in d.columns if c not in NON_FEATURES]
    train = d[d["split"] == "train"]
    val = d[d["split"] == "val"]
    later = d[d["split"].isin(["test", "after_test"])]
    trval = pd.concat([train, val])
    print(f"rows: train {len(train)}, val {len(val)}, later {len(later)}; {len(feats)} features")

    n_pos_val = int(val["y_spike"].sum())
    print(f"spike rows: train {int(train['y_spike'].sum())}, val {n_pos_val}")
    if n_pos_val < 20:
        print("WARNING: very few validation spikes, the alert threshold will be unreliable")

    print("Regression search")
    reg_best = search("reg", train[feats], train["y_reg"], val[feats], val["y_reg"], n_iter, rng)
    print("Classifier search")
    clf_best = search("clf", train[feats], train["y_spike"], val[feats], val["y_spike"],
                      n_iter, rng)

    # Train-only models: validation-year predictions and the alert threshold.
    reg_tr = make_model("reg", reg_best["params"]).fit(train[feats], train["y_reg"])
    clf_tr = make_model("clf", clf_best["params"]).fit(train[feats], train["y_spike"])
    val_pred = reg_tr.predict(val[feats])
    val_prob = clf_tr.predict_proba(val[feats])[:, 1]
    thr = pick_threshold(val["y_spike"].values, val_prob)
    print(f"alert threshold chosen on validation: {thr:.3f}")

    # Refit on train + validation (no early stopping in RF, so no tree-count scaling needed).
    reg_fin = make_model("reg", reg_best["params"]).fit(trval[feats], trval["y_reg"])
    clf_fin = make_model("clf", clf_best["params"]).fit(trval[feats], trval["y_spike"])
    later_pred = reg_fin.predict(later[feats])
    later_prob = clf_fin.predict_proba(later[feats])[:, 1]

    def frame(rows, model_name, y_pred, prob):
        return pd.DataFrame({
            "Datetime": rows["Datetime"].values, "StationId": st,
            "split": rows["split"].values, "model": model_name,
            "y_true": rows["y_reg"].values, "y_pred": y_pred,
            "lower": np.nan, "upper": np.nan,
            "spike_prob": prob, "y_spike": rows["y_spike"].values,
            "alert": (prob >= thr).astype(int), "pm25_now": rows["pm25_now"].values,
            "is_diwali": rows["is_diwali"].values, "is_lockdown": rows["is_lockdown"].values,
            "target_imputed": rows["target_imputed"].values})

    preds = pd.concat([frame(val, "train_only", val_pred, val_prob),
                       frame(later, "refit", later_pred, later_prob)], ignore_index=True)

    reg_tab, spike_tab = evaluate(preds, thr)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = None
    print("\nForecast error (PM2.5, ug/m3; bias = mean prediction minus truth):")
    print(reg_tab)
    print(f"\nSpike alerts at threshold {thr:.3f} (row level):")
    print(spike_tab)

    imp = pd.DataFrame({"regression_importance": reg_fin.feature_importances_,
                        "classifier_importance": clf_fin.feature_importances_},
                       index=feats).sort_values("regression_importance", ascending=False)
    print("\nTop features (impurity importance):")
    print(imp.head(8))

    for sub in ("predictions", "models", "reports"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / "predictions" / f"rf_{st}.csv", index=False)
    reg_tab.to_csv(OUT / "reports" / f"rf_{st}_forecast_metrics.csv")
    spike_tab.to_csv(OUT / "reports" / f"rf_{st}_spike_metrics.csv")
    imp.to_csv(OUT / "reports" / f"rf_{st}_feature_importance.csv")
    joblib.dump(reg_fin, OUT / "models" / f"rf_reg_{st}.joblib", compress=3)
    joblib.dump(clf_fin, OUT / "models" / f"rf_clf_{st}.joblib", compress=3)
    meta = dict(station=st, features=feats, alert_threshold=thr,
                regression=reg_best, classifier=clf_best, n_estimators=N_TREES,
                horizon=cfg.HORIZON, spike_threshold=cfg.THRESHOLD, seed=SEED)
    (OUT / "models" / f"rf_{st}_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=cfg.STATIONS)
    ap.add_argument("--search", type=int, default=20, help="random-search trials per model")
    args = ap.parse_args()

    df = pd.read_csv(cfg.FEATURES_PATH, parse_dates=["Datetime"])
    rng = np.random.default_rng(SEED)
    for st in args.stations:
        run_station(df, st, args.search, rng)


if __name__ == "__main__":
    main()