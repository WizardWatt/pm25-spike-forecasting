"""Step 1: clean the raw CPCB hourly data.

Run:  python clean.py
Reads data/raw/station_hour.csv, writes data/processed/clean_hourly.csv
plus a cleaning report and plots in reports/.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg


def nan_run_length(s):
    """For every NaN, the length of the NaN run it belongs to (0 for non-NaN)."""
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    lengths = isna.groupby(grp).transform("sum")
    return lengths.where(isna, 0)


def fill_short_gaps(s, max_gap):
    """Linearly interpolate only gaps of at most max_gap hours. Long gaps stay NaN."""
    interp = s.interpolate(method="linear", limit_area="inside")
    runs = nan_run_length(s)
    mask = s.isna() & (runs <= max_gap) & interp.notna()
    out = s.copy()
    out[mask] = interp[mask]
    return out, mask


def stuck_mask(s, min_len):
    """True where the same value repeats for at least min_len consecutive hours."""
    grp = s.ne(s.shift()).cumsum()
    run_len = s.groupby(grp).transform("size")
    return s.notna() & (run_len >= min_len)


def clean_station(g, pollutants):
    """Clean one station's data. Returns (cleaned frame, stats dict)."""
    stats = {}
    g = g.copy()
    g["Datetime"] = g["Datetime"].dt.floor("h")
    g = g.sort_values("Datetime")
    stats["rows_raw"] = len(g)
    stats["duplicate_timestamps"] = int(g.duplicated("Datetime").sum())
    g = g.drop_duplicates("Datetime", keep="first")

    full = pd.date_range(g["Datetime"].min(), g["Datetime"].max(), freq="h")
    stats["hours_inserted"] = len(full) - len(g)
    g = g.set_index("Datetime").reindex(full)
    g.index.name = "Datetime"

    # Invalid values -> NaN
    stats["negative_values"] = 0
    for c in pollutants:
        neg = g[c] < 0
        stats["negative_values"] += int(neg.sum())
        g.loc[neg, c] = np.nan
    too_high = g["PM2.5"] > cfg.MAX_PM25
    stats["pm25_above_max"] = int(too_high.sum())
    g.loc[too_high, "PM2.5"] = np.nan
    stuck = stuck_mask(g["PM2.5"], cfg.STUCK_HOURS)
    stats["pm25_stuck_hours"] = int(stuck.sum())
    g.loc[stuck, "PM2.5"] = np.nan

    g["pm25_missing_raw"] = g["PM2.5"].isna()

    # Fill only short gaps
    g["PM2.5"], g["pm25_imputed"] = fill_short_gaps(g["PM2.5"], cfg.MAX_FILL_GAP)
    for c in pollutants:
        if c != "PM2.5":
            g[c], _ = fill_short_gaps(g[c], cfg.MAX_FILL_GAP)

    stats["pm25_missing_after_pct"] = round(100 * g["PM2.5"].isna().mean(), 1)
    stats["pm25_imputed_hours"] = int(g["pm25_imputed"].sum())
    stats["longest_pm25_gap_hours"] = int(nan_run_length(g["PM2.5"]).max())
    g = g.reset_index()
    return g, stats


def investigate(df):
    """Spike counts per year and monthly plots, to sanity-check suspicious stretches."""
    d = df.copy()
    d["spike_raw"] = (d["PM2.5"] > cfg.THRESHOLD) & ~d["pm25_imputed"]
    d["year"] = d["Datetime"].dt.year
    by_year = d.groupby(["StationId", "year"]).agg(
        spike_hours=("spike_raw", "sum"),
        mean_pm25=("PM2.5", "mean"),
        missing_pct=("PM2.5", lambda s: 100 * s.isna().mean()),
    ).round(1)
    by_year.to_csv(cfg.REPORT_DIR / "yearly_summary.csv")
    print("\nPer-year summary (check unusual years, e.g. KA009 2016):")
    print(by_year)

    stations = sorted(d["StationId"].unique())
    fig, axes = plt.subplots(len(stations), 2, figsize=(13, 3.2 * len(stations)), squeeze=False)
    for i, st in enumerate(stations):
        g = d[d["StationId"] == st].set_index("Datetime")
        monthly_mean = g["PM2.5"].resample("MS").mean()
        monthly_missing = 100 * g["PM2.5"].isna().resample("MS").mean()
        axes[i, 0].plot(monthly_mean.index, monthly_mean.values)
        axes[i, 0].set_title(f"{st}: monthly mean PM2.5")
        axes[i, 1].bar(monthly_missing.index, monthly_missing.values, width=25)
        axes[i, 1].set_title(f"{st}: % of hours missing per month")
        axes[i, 1].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(cfg.REPORT_DIR / "monthly_overview.png", dpi=130)
    plt.close(fig)


def main():
    cfg.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(cfg.RAW_PATH, parse_dates=["Datetime"], low_memory=False)
    raw = raw[raw["StationId"].isin(cfg.STATIONS)]
    pollutants = ["PM2.5"] + [c for c in cfg.EXTRA_POLLUTANTS if c in raw.columns]
    raw = raw[["StationId", "Datetime"] + pollutants]

    frames, all_stats = [], {}
    for st, g in raw.groupby("StationId"):
        cleaned, stats = clean_station(g.drop(columns="StationId"), pollutants)
        cleaned.insert(0, "StationId", st)
        frames.append(cleaned)
        all_stats[st] = stats

    clean = pd.concat(frames, ignore_index=True)
    clean.to_csv(cfg.CLEAN_PATH, index=False)

    report = pd.DataFrame(all_stats).T
    report.to_csv(cfg.REPORT_DIR / "cleaning_report.csv")
    print("Cleaning report:")
    print(report.T)
    investigate(clean)
    print(f"\nSaved {cfg.CLEAN_PATH}")


if __name__ == "__main__":
    main()