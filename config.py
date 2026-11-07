"""Shared settings. Both people import from here, so nothing is hardcoded elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "station_hour.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "clean_hourly.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
REPORT_DIR = ROOT / "reports"

# Task definition
STATIONS = ["KA003", "KA009"]
HORIZON = 6            # forecast this many hours ahead
THRESHOLD = 90.0       # spike = PM2.5 above this (CPCB Poor band)

# Cleaning rules
MAX_PM25 = 1000.0      # readings above this are treated as sensor errors
STUCK_HOURS = 12       # identical readings for this many hours in a row = stuck sensor
MAX_FILL_GAP = 3       # only interpolate gaps up to this many hours
EXTRA_POLLUTANTS = []
EXTRA_MAX_MISSING = 0.40   # use an extra pollutant only if under 40% missing at both stations

# Chronological split (boundaries are start-inclusive, end-exclusive)
TRAIN_END = "2018-01-01"
VAL_END = "2019-01-01"
TEST_END = "2020-01-01"
LOCKDOWN_START = "2020-03-25"
DIWALI_DATES = ["2018-11-07", "2019-10-27"]
DIWALI_WINDOW_DAYS = 3

# Features
LAGS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 24]
EXTRA_LAGS = [6, 24]