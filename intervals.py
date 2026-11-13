"""Conformal prediction intervals for any model's prediction file (outputs/predictions/{model}_{station}.csv).

Idea (split conformal): look at how wrong the model was on 2018, using only rows where the target was a real
reading. Take the 5th and 95th percentile of the signed errors (truth - forecast), with the standard
finite-sample correction, and add them to every forecast: lower = forecast + q_low, upper = forecast + q_high.
That gives about 90% coverage (alpha = 0.1) if future errors look like 2018 errors. The two sides are
separate, so the upper side can be wider than the lower one (spikes are under-forecast more than over-forecast).

Three variants, because PM2.5 errors grow with the level:
  split    : one pair of quantiles for everything (constant width)
  scaled   : errors divided by the forecast level (width grows with the forecast)
  mondrian : separate quantiles for low / mid / high forecast levels (3 bins by default)

Honest checking:
  --dev          calibrate on Jan-Jun 2018 and check on Jul-Dec 2018 (and the reverse). Uses no 2019 data.
                 Seasons differ, so this is a hard test. Use it to choose a method.
  --score-test   after the final refit, also print coverage on the refit rows (2019 and later). Do this once.
Intervals go into the lower/upper columns of a NEW file, {model}_ci_{station}.csv (A's files stay untouched;
add --inplace to overwrite the original). Settings go to outputs/models/. Use e.g. compare_models --models xgb_ci gru_ci.

  python intervals.py --dev
  python intervals.py --method mondrian --alpha 0.1 --models xgb gru
"""
import argparse, json, os
import numpy as np
import pandas as pd
import eval_config as C
from shared_format import _path

METHODS = ("split", "scaled", "mondrian")


def _q(r, alpha):
    """(low, high) conformal quantiles of signed residuals r = truth - forecast."""
    r = np.sort(np.asarray(r, float))
    n = len(r)
    k_hi = int(np.ceil((n + 1) * (1 - alpha / 2)))
    k_lo = int(np.floor((n + 1) * (alpha / 2)))
    return float(r[max(k_lo, 1) - 1]), float(r[min(k_hi, n) - 1])


class ConformalInterval:
    def __init__(self, method="mondrian", alpha=0.1, n_bins=3, floor=10.0, min_bin=100):
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.alpha, self.n_bins, self.floor, self.min_bin = method, alpha, n_bins, floor, min_bin

    def fit(self, y_true, y_pred):
        y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
        ok = ~(np.isnan(y) | np.isnan(p))
        y, p = y[ok], p[ok]
        if len(y) < 50:
            raise ValueError("need at least 50 calibration rows")
        r = y - p
        self.n_cal = int(len(y))
        self.pooled = _q(r, self.alpha)
        if self.method == "scaled":
            self.scaled = _q(r / np.maximum(p, self.floor), self.alpha)
        if self.method == "mondrian":
            self.edges = np.quantile(p, np.linspace(0, 1, self.n_bins + 1)[1:-1])
            b = np.digitize(p, self.edges)
            self.bin_q = [_q(r[b == i], self.alpha) if (b == i).sum() >= self.min_bin else self.pooled
                          for i in range(self.n_bins)]
        return self

    def predict(self, y_pred):
        p = np.asarray(y_pred, float)
        if self.method == "split":
            lo, hi = self.pooled
            lower, upper = p + lo, p + hi
        elif self.method == "scaled":
            s, (lo, hi) = np.maximum(p, self.floor), self.scaled
            lower, upper = p + s * lo, p + s * hi
        else:
            b = np.digitize(p, self.edges)                       # NaN lands in the last bin; result stays NaN
            lo = np.array([q[0] for q in self.bin_q])[b]
            hi = np.array([q[1] for q in self.bin_q])[b]
            lower, upper = p + lo, p + hi
        return np.maximum(lower, 0.0), upper                     # PM2.5 cannot be negative

    def to_dict(self):
        d = {"method": self.method, "alpha": self.alpha, "n_bins": self.n_bins, "floor": self.floor,
             "min_bin": self.min_bin, "n_cal": self.n_cal, "pooled": list(self.pooled)}
        if self.method == "scaled":
            d["scaled"] = list(self.scaled)
        if self.method == "mondrian":
            d["edges"], d["bin_q"] = [float(e) for e in self.edges], [list(q) for q in self.bin_q]
        return d

    @classmethod
    def from_dict(cls, d):
        o = cls(d["method"], d["alpha"], d["n_bins"], d["floor"], d["min_bin"])
        o.n_cal, o.pooled = d["n_cal"], tuple(d["pooled"])
        if "scaled" in d:
            o.scaled = tuple(d["scaled"])
        if "edges" in d:
            o.edges, o.bin_q = np.array(d["edges"]), [tuple(q) for q in d["bin_q"]]
        return o


# ---------------------------------------------------------------- reading files / diagnostics
def _imputed(df):
    return df["target_imputed"].fillna(False).astype(bool) if "target_imputed" in df else pd.Series(False, index=df.index)


def calibration_rows(df):
    """2018 rows from the train-only fit, real (not imputed) targets, forecast and truth present."""
    return df[(df["model"] == "train_only") & ~_imputed(df) & df["y_true"].notna() & df["y_pred"].notna()]


def interval_report(df, label=""):
    """Coverage and width overall and by group. Skips imputed targets, like the rest of the scoring."""
    d = df[~_imputed(df) & df[["y_true", "y_pred", "lower", "upper"]].notna().all(axis=1)].copy()
    d["covered"] = (d["y_true"] >= d["lower"]) & (d["y_true"] <= d["upper"])
    d["width"] = d["upper"] - d["lower"]
    groups = {"all": pd.Series(True, index=d.index),
              "truth <= 90": d["y_true"] <= C.SPIKE_THRESHOLD, "truth > 90": d["y_true"] > C.SPIKE_THRESHOLD}
    for lo, hi, name in [(0, 30, "forecast <30"), (30, 60, "forecast 30-60"), (60, 90, "forecast 60-90"),
                         (90, np.inf, "forecast >=90")]:
        groups[name] = (d["y_pred"] >= lo) & (d["y_pred"] < hi)
    for col in ("is_diwali", "is_lockdown"):
        if col in d:
            groups[col] = d[col].fillna(False).astype(bool)
    if "split" in d:
        for sp in d["split"].dropna().unique():
            groups[f"split={sp}"] = d["split"] == sp
    rows = []
    for name, m in groups.items():
        g = d[m]
        if len(g):
            rows.append({"set": label, "group": name, "n": len(g), "coverage": g["covered"].mean(),
                         "mean_width": g["width"].mean()})
    return pd.DataFrame(rows).round(3)


def dev_check(df, methods=METHODS, alpha=0.1):
    """Calibrate on one half of 2018 (Jan-Jun or Jul-Dec), test on the other. No 2019 data."""
    cal = calibration_rows(df).copy()
    cal["month"] = pd.to_datetime(cal["Datetime"]).dt.month
    rows = []
    for name, a, b in [("Jan-Jun -> Jul-Dec", cal["month"] <= 6, cal["month"] > 6),
                       ("Jul-Dec -> Jan-Jun", cal["month"] > 6, cal["month"] <= 6)]:
        for m in methods:
            ci = ConformalInterval(m, alpha).fit(cal[a]["y_true"], cal[a]["y_pred"])
            t = cal[b].copy()
            t["lower"], t["upper"] = ci.predict(t["y_pred"].to_numpy())
            cov = (t["y_true"] >= t["lower"]) & (t["y_true"] <= t["upper"])
            hi_f, hi_t = t["y_pred"] >= 60, t["y_true"] > C.SPIKE_THRESHOLD
            rows.append({"direction": name, "method": m, "n_cal": int(a.sum()), "n_test": int(b.sum()),
                         "coverage": cov.mean(), "mean_width": (t["upper"] - t["lower"]).mean(),
                         "cov_forecast>=60": cov[hi_f].mean() if hi_f.any() else np.nan,
                         "cov_truth>90": cov[hi_t].mean() if hi_t.any() else np.nan})
    return pd.DataFrame(rows).round(3)


def ci_path(path):
    """xgb_KA003.csv -> xgb_ci_KA003.csv (so Person A's own files are never modified)."""
    stem, st = path[:-4].rsplit("_", 1)
    return f"{stem}_ci_{st}.csv"


def apply_intervals(path, method="mondrian", alpha=0.1, model_dir=None, inplace=False):
    """Calibrate on the 2018 rows of a prediction file and write lower/upper for ALL its rows.
    Writes to {model}_ci_{station}.csv unless inplace=True."""
    df = pd.read_csv(path)
    ci = ConformalInterval(method, alpha).fit(calibration_rows(df)["y_true"], calibration_rows(df)["y_pred"])
    df["lower"], df["upper"] = ci.predict(df["y_pred"].to_numpy())
    out = path if inplace else ci_path(path)
    df.to_csv(out, index=False)
    mdir = model_dir or C.MODEL_DIR
    os.makedirs(mdir, exist_ok=True)
    side = os.path.join(mdir, os.path.basename(out).replace(".csv", "_conformal.json"))
    json.dump(ci.to_dict(), open(side, "w"), indent=2)
    return df, ci, side


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xgb", "gru"])
    ap.add_argument("--method", default="mondrian", choices=METHODS)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--inplace", action="store_true", help="overwrite the original file instead of writing *_ci_*.csv")
    ap.add_argument("--dev", action="store_true", help="2018 half-year check only; writes nothing")
    ap.add_argument("--score-test", action="store_true", help="also print coverage on refit rows (2019+); once only")
    a = ap.parse_args()
    for m in a.models:
        for st in C.STATIONS:
            path = _path(m, st)
            if not os.path.exists(path):
                print(f"(skipping {m} / {st}: no file)")
                continue
            if a.dev:
                print(f"\n=== {m} {st}: 2018 half-year check (nominal coverage {1 - a.alpha:.0%}) ===")
                print(dev_check(pd.read_csv(path), alpha=a.alpha).to_string(index=False))
                continue
            df, ci, side = apply_intervals(path, a.method, a.alpha, inplace=a.inplace)
            print(f"\n=== {m} {st}: {a.method}, alpha={a.alpha}, calibrated on {ci.n_cal} rows of 2018 -> {side}")
            print(interval_report(df[df["model"] == "train_only"], "2018 (in-sample)").to_string(index=False))
            if a.score_test:
                print(interval_report(df[df["model"] == "refit"], "refit rows").to_string(index=False))
