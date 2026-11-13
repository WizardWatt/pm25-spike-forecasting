"""CPCB PM2.5 sub-index and category, mirroring the official AQI-Calculator.xls (24-hr avg breakpoints).

PM2.5 breakpoints (ug/m3): 0-30 Good, 31-60 Satisfactory, 61-90 Moderate, 91-120 Poor,
121-250 Very Poor, >250 Severe. Our spike threshold (90) is the Moderate/Poor boundary (sub-index 200).
NOTE: CPCB defines these on 24-hour averages; applying them to hourly values is our own choice.
"""
import numpy as np

_CONC = [0, 30, 60, 90, 120, 250]
_IDX = [0, 50, 100, 200, 300, 400]
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
UPPER = [30, 60, 90, 120, 250, np.inf]       # inclusive upper bounds (ug/m3)
HEALTH = {
    "Good": "Minimal impact",
    "Satisfactory": "Minor breathing discomfort to sensitive people",
    "Moderate": "Breathing discomfort to people with lung/heart disease, children and older adults",
    "Poor": "Breathing discomfort to people on prolonged exposure",
    "Very Poor": "Respiratory illness to people on prolonged exposure",
    "Severe": "Respiratory effects even on healthy people",
}


def pm25_subindex(c):
    """Same as the sheet; above 250 it continues linearly (slope 100/130) and is not capped."""
    c = np.asarray(c, float)
    sub = np.interp(c, _CONC, _IDX)
    return np.where(c > 250, 400 + (c - 250) * 100 / 130, sub)


def pm25_category(c):
    c = np.asarray(c, float)
    i = np.searchsorted(UPPER, c, side="left")
    out = np.array(CATEGORIES, dtype=object)[np.minimum(i, 5)]
    res = np.where(np.isnan(c), None, out)
    return res.item() if res.ndim == 0 else res
