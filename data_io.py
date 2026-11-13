"""Load A's features.csv in the shape Person B's code expects.

A's features.csv has pm25_now (PM2.5 at the issue time t), y_reg (PM2.5 at t+6), y_spike
(any hour in t+1..t+6 above THRESHOLD), plus split / model_ready / target_imputed.
We rename pm25_now -> PM2.5 and keep only the columns B's code uses.
"""
import pandas as pd
import eval_config as C

NEEDED = ["Datetime", "StationId", "PM2.5", "y_reg", "y_spike"]
OPTIONAL = ["split", "model_ready", "target_imputed", "is_diwali", "is_lockdown"]


def load_clean(path=None):
    df = pd.read_csv(path or C.FEATURES_PATH, parse_dates=["Datetime"])
    if "PM2.5" not in df.columns and "pm25_now" in df.columns:
        df = df.rename(columns={"pm25_now": "PM2.5"})
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path or C.FEATURES_PATH} is missing {missing}. "
                         "Point --clean at data/processed/features.csv (not clean_hourly.csv).")
    keep = NEEDED + [c for c in OPTIONAL if c in df.columns]
    return df[keep].sort_values(["StationId", "Datetime"]).reset_index(drop=True)
