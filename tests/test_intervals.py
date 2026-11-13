import sys, os, tempfile, json; sys.path.insert(0, "."); sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import eval_config as C
from intervals import ci_path, ConformalInterval, interval_report, dev_check, apply_intervals, calibration_rows, _q
from baselines import write_all
from shared_format import load_preds, _path
from data_io import load_clean
from synth import make_features

def hetero(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.gamma(3, 15, n)                                           # forecast level
    y = np.maximum(p + rng.standard_normal(n) * (3 + 0.4 * p) + 0.1 * p, 0)   # errors grow with level, skewed up
    return y, p

def test_all():
    # finite-sample quantile helper: ~5% below / ~5% above
    r = np.arange(1, 1001) - 500.5
    lo, hi = _q(r, 0.1)
    assert (r < lo).mean() <= 0.05 and (r > hi).mean() <= 0.05 + 1e-9
    # marginal coverage near 90% on exchangeable held-out data, for every method
    yc, pc = hetero(6000, 0); yt, pt = hetero(6000, 1)
    cov = {}
    for m in ("split", "scaled", "mondrian"):
        ci = ConformalInterval(m, 0.1).fit(yc, pc)
        lo, hi = ci.predict(pt)
        assert (lo >= 0).all() and np.isnan(ci.predict([np.nan])[0]).all()
        cov[m] = (yt >= lo) & (yt <= hi)
        assert abs(cov[m].mean() - 0.9) < 0.02, (m, cov[m].mean())
        ci2 = ConformalInterval.from_dict(json.loads(json.dumps(ci.to_dict())))   # save / load round trip
        assert np.allclose(ci2.predict(pt)[1], hi)
    # heteroscedastic errors: constant width under-covers the high forecasts; level-aware methods do better
    high = pt > np.quantile(pt, 0.8)
    assert cov["split"][high].mean() < 0.86 and cov["mondrian"][high].mean() > 0.87 and cov["scaled"][high].mean() > 0.87

    # end to end on a prediction file in A's format; imputed targets must not be used for calibration
    feats = pd.concat([make_features("KA003", 0), make_features("KA009", 1)], ignore_index=True)
    with tempfile.TemporaryDirectory() as td:
        feats.to_csv(td + "/features.csv", index=False)
        clean = load_clean(td + "/features.csv")
        write_all(clean, td)
        path = _path("hist_avg", "KA003", td)
        raw = pd.read_csv(path)
        raw.loc[(raw.model == "train_only") & (raw.index % 7 == 0), "target_imputed"] = True
        raw.to_csv(path, index=False)
        n_cal = len(calibration_rows(pd.read_csv(path)))
        df, ci, side = apply_intervals(path, "mondrian", 0.1, model_dir=td)
        assert ci.n_cal == n_cal and os.path.exists(side) and os.path.exists(ci_path(path))
        assert load_preds("hist_avg", "KA003", td)["lower"].isna().all()          # original file untouched
        assert load_preds("hist_avg_ci", "KA003", td)["lower"].notna().any()
        assert df.loc[df.y_pred.notna(), ["lower", "upper"]].notna().all().all()
        assert (df.lower <= df.upper).all()
        rep = interval_report(df[df.model == "train_only"], "val")
        assert abs(rep.loc[rep.group == "all", "coverage"].iloc[0] - 0.9) < 0.03        # in-sample ~ nominal
        d = dev_check(pd.read_csv(path))
        assert set(d.method) == {"split", "scaled", "mondrian"} and len(d) == 6
        print(rep.to_string(index=False)); print(d.to_string(index=False))

if __name__ == "__main__":
    test_all(); print("PASS intervals")
