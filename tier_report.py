"""Do the advisory tiers mean something? For each tier, how often did a spike really follow?

Reads outputs/predictions/xgb_ci_<station>.csv. Tier thresholds come from advisory.py and were
fixed from 2018 validation, so 2019 is a fair test. Nothing here changes any saved file.

  python tier_report.py
"""
import pandas as pd

import advisory as A
import eval_config as C

SETS = [("2018 val", "val", "train_only"), ("2019 test", "test", "refit")]


def main():
    rows = []
    for st in C.STATIONS:
        d = pd.read_csv(C.PRED_DIR / f"xgb_ci_{st}.csv", parse_dates=["Datetime"])
        thr = A.ALERT_THRESHOLDS[("xgb_ci", st)]
        for label, split, model in SETS:
            m = d[(d["split"] == split) & (d["model"] == model) & ~d["target_imputed"]].dropna(
                subset=["spike_prob", "y_spike", "pm25_now"])
            m = m.assign(tier=A.tier_array(m["spike_prob"], m["pm25_now"], thr))
            for t in A.TIERS:
                g = m[m["tier"] == t]
                rows.append({"station": st, "set": label, "tier": t, "hours": len(g),
                             "share_of_hours": round(len(g) / len(m), 3),
                             "spike_followed": round(g["y_spike"].mean(), 3) if len(g) else None})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(C.ROOT / "outputs" / "reports" / "tier_report.csv", index=False)


if __name__ == "__main__":
    main()