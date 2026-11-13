import sys; sys.path.insert(0, ".")
import numpy as np
from shap_xgb import family, importance_table, family_table, group_compare

def test_all():
    assert [family(f) for f in ["pm25_now", "pm25_lag6", "pm25_mean24", "pm25_max6", "pm25_std24", "pm25_trend3",
                                "hour_sin", "month_cos", "dayofweek", "is_weekend", "other_thing"]] == \
        ["current level", "lags", "rolling mean", "rolling max", "rolling std", "trend",
         "hour of day", "season", "weekday", "weekday", "other"]
    feats = ["pm25_now", "pm25_lag1", "hour_sin"]
    rng = np.random.default_rng(0)
    sv = rng.standard_normal((200, 3)) * np.array([5.0, 2.0, 0.5])
    imp = importance_table(sv, feats)
    assert imp.feature.iloc[0] == "pm25_now" and abs(imp.share.sum() - 1) < 1e-9
    fam = family_table(imp)
    assert set(fam.family) == {"current level", "lags", "hour of day"} and abs(fam.share.sum() - 1) < 1e-9
    mask = np.arange(200) < 20
    sv2 = sv.copy(); sv2[mask, 2] *= 10                              # hour_sin matters 10x more inside the window
    g = group_compare(sv2, feats, mask)
    assert g.loc[g.feature == "hour_sin", "ratio"].iloc[0] > 5
    assert group_compare(sv, feats, np.zeros(200, bool)).empty

if __name__ == "__main__":
    test_all(); print("PASS shap_agg")
