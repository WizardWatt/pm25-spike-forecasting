"""Step 4: XGBoost regression (PM2.5 six hours ahead) and spike classifier.

Run:  python train_xgb.py                      (both stations, 20 search trials each)
      python train_xgb.py --stations KA003 --search 40

Protocol (one pair of models per station):
  1. Train on 2015-2017, early-stop and tune on 2018.
  2. Pick the alert threshold on 2018 only.
  3. Refit on 2015-2018 with the settings found, then predict 2019 and 2020 once.
  4. 2018 predictions come from the train-only model, so Diwali 2018 stays held out.

Outputs (in outputs/):
  predictions/xgb_<station>.csv   shared format, read by the evaluation harness
  models/                         saved models, chosen settings and threshold
  reports/xgb_<station>_*.csv     metric tables and feature importance
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve

import config as cfg

OUT = cfg.ROOT / "outputs"
SEED = 42
NON_FEATURES = {"Datetime", "StationId", "y_reg", "y_spike", "target_imputed",
                "split", "is_lockdown", "is_diwali", "model_ready"}
DEFAULT_PARAMS = dict(max_depth=5, min_child_weight=5, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, learning_rate=0.03)


def sample_params(rng):
    return dict(
        max_depth=int(rng.choice([3, 4, 5, 6, 8])),
        min_child_weight=int(rng.choice([1, 5, 10, 20])),
        subsample=float(rng.choice([0.6, 0.8, 1.0])),
        colsample_bytree=float(rng.choice([0.6, 0.8, 1.0])),
        reg_lambda=float(rng.choice([1.0, 5.0, 10.0])),
        learning_rate=0.03,
    )


def make_model(kind, params, n_estimators, early_stopping=None, spw=1.0):
    common = dict(n_estimators=n_estimators, random_state=SEED, n_jobs=-1,
                  importance_type="gain", **params)
    if early_stopping:
        common["early_stopping_rounds"] = early_stopping
    if kind == "reg":
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="mae", **common)
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                             scale_pos_weight=spw, **common)


def search(kind, Xtr, ytr, Xva, yva, n_iter, rng, spw=1.0):
    """Random search. Trial 0 is the default setting. Returns the best settings."""
    best = None
    for i in range(n_iter):
        params = dict(DEFAULT_PARAMS) if i == 0 else sample_params(rng)
        m = make_model(kind, params, 3000, early_stopping=100, spw=spw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        score = float(m.best_score)
        better = best is None or (score < best["score"] if kind == "reg" else score > best["score"])
        if better:
            best = dict(params=params, score=score, n_trees=int(m.best_iteration) + 1)
            print(f"  [{kind}] trial {i + 1}/{n_iter}: new best val score {score:.4f} "
                  f"(depth {params['max_depth']}, {best['n_trees']} trees)")
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

    n_pos_tr, n_pos_val = int(train["y_spike"].sum()), int(val["y_spike"].sum())
    print(f"spike rows: train {n_pos_tr}, val {n_pos_val}")
    if n_pos_val < 20:
        print("WARNING: very few validation spikes, the alert threshold will be unreliable")
    spw = (len(train) - n_pos_tr) / max(n_pos_tr, 1)

    print("Regression search")
    reg_best = search("reg", train[feats], train["y_reg"], val[feats], val["y_reg"], n_iter, rng)
    print("Classifier search")
    clf_best = search("clf", train[feats], train["y_spike"], val[feats], val["y_spike"],
                      n_iter, rng, spw=spw)

    # Train-only models: used for validation-year predictions and for the threshold.
    reg_tr = make_model("reg", reg_best["params"], reg_best["n_trees"])
    reg_tr.fit(train[feats], train["y_reg"])
    clf_tr = make_model("clf", clf_best["params"], clf_best["n_trees"], spw=spw)
    clf_tr.fit(train[feats], train["y_spike"])
    val_pred = reg_tr.predict(val[feats])
    val_prob = clf_tr.predict_proba(val[feats])[:, 1]
    thr = pick_threshold(val["y_spike"].values, val_prob)
    print(f"alert threshold chosen on validation: {thr:.3f}")

    # Refit on train + validation with trees scaled to the larger data, then predict later years.
    scale = len(trval) / len(train)
    spw_all = (len(trval) - trval["y_spike"].sum()) / max(trval["y_spike"].sum(), 1)
    reg_fin = make_model("reg", reg_best["params"], int(reg_best["n_trees"] * scale))
    reg_fin.fit(trval[feats], trval["y_reg"])
    clf_fin = make_model("clf", clf_best["params"], int(clf_best["n_trees"] * scale), spw=spw_all)
    clf_fin.fit(trval[feats], trval["y_spike"])
    later_pred = reg_fin.predict(later[feats])
    later_prob = clf_fin.predict_proba(later[feats])[:, 1]

    def frame(rows, model_name, y_pred, prob):
        return pd.DataFrame({
            "Datetime": rows["Datetime"].values, "StationId": st,
            "split": rows["split"].values, "model": model_name,
            "y_true": rows["y_reg"].values, "y_pred": y_pred,
            "lower": np.nan, "upper": np.nan,          # filled in by the interval step
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
    print(f"\nSpike alerts at threshold {thr:.3f} (row level; base_rate is the share of spike rows):")
    print(spike_tab)

    imp = pd.DataFrame({"regression_gain": reg_fin.feature_importances_,
                        "classifier_gain": clf_fin.feature_importances_},
                       index=feats).sort_values("regression_gain", ascending=False)
    print("\nTop features (gain):")
    print(imp.head(8))

    for sub in ("predictions", "models", "reports"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / "predictions" / f"xgb_{st}.csv", index=False)
    reg_tab.to_csv(OUT / "reports" / f"xgb_{st}_forecast_metrics.csv")
    spike_tab.to_csv(OUT / "reports" / f"xgb_{st}_spike_metrics.csv")
    imp.to_csv(OUT / "reports" / f"xgb_{st}_feature_importance.csv")
    reg_fin.save_model(str(OUT / "models" / f"xgb_reg_{st}.json"))
    clf_fin.save_model(str(OUT / "models" / f"xgb_clf_{st}.json"))
    meta = dict(station=st, features=feats, alert_threshold=thr,
                regression=reg_best, classifier=clf_best,
                horizon=cfg.HORIZON, spike_threshold=cfg.THRESHOLD, seed=SEED)
    (OUT / "models" / f"xgb_{st}_meta.json").write_text(json.dumps(meta, indent=2))


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