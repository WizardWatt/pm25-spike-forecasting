"""Step 4: rule-based health advisory.

Turns one model output (spike probability, current PM2.5, point forecast, interval) into
a tier, plain-language advice and honest caveats. No machine learning here: every rule is
readable and can be explained.

Design choices (all fixed BEFORE looking at 2019 test results):
  * The alert threshold comes from the 2018 validation year (see ALERT_THRESHOLDS).
  * Three tiers only. A fourth "WATCH" tier (half the threshold, or current PM2.5 > 60) was
    tried and removed: spikes followed those hours about as rarely as "low risk" hours, so it
    carried no information.
  * spike_prob is a RANKING SCORE, not a true probability (class-weighted XGBoost; at KA003
    it never goes below ~0.36). Never show it to users as a percentage.
  * The interval is shown as a "typical range" and never drives the advice, because on
    real spike hours it covered only 27% (KA003) / 49% (KA009) of values.
  * Hourly PM2.5 is mapped to CPCB bands, but CPCB defines them on 24-hour averages, so
    the wording says "short-term" and never "air quality is Poor today".

This is general information, not medical advice.
"""
from dataclasses import dataclass, field

import numpy as np

import aqi

# Alert thresholds chosen on 2018 validation (from outputs/reports/compare_test.csv).
ALERT_THRESHOLDS = {
    ("xgb_ci", "KA003"): 0.533,
    ("xgb_ci", "KA009"): 0.51,
}
SPIKE_LEVEL = 90.0  # ug/m3, CPCB Moderate/Poor boundary

TIERS = ["LOW_RISK", "WARNING", "ELEVATED_NOW"]

HEADLINES = {
    "LOW_RISK": "No spike expected in the next 6 hours",
    "WARNING": "Pollution spike likely in the next 6 hours",
    "ELEVATED_NOW": "PM2.5 is high right now",
}

ACTIONS = {
    "LOW_RISK": [
        "Normal outdoor activity is fine for most people.",
    ],
    "WARNING": [
        "Consider moving strenuous outdoor activity (running, cycling, sport) to later or indoors.",
        "Children, older adults, pregnant people and people with asthma or heart/lung "
        "conditions should limit time outdoors.",
        "Keep windows facing busy roads closed while levels are high.",
        "If you must be outside, a well-fitted N95/FFP2 mask helps; a loose cloth mask does not.",
    ],
    "ELEVATED_NOW": [
        "Limit strenuous outdoor activity until readings come down.",
        "Children, older adults, pregnant people and people with asthma or heart/lung "
        "conditions should stay indoors where possible.",
        "If you have breathing difficulty, chest pain or severe symptoms, seek medical care.",
    ],
}

# Our own plain-language wording for hourly bands (NOT CPCB's official text).
BAND_NOTE = {
    "Good": "very low",
    "Satisfactory": "low",
    "Moderate": "moderate",
    "Poor": "high",
    "Very Poor": "very high",
    "Severe": "extremely high",
}

CAVEATS_ALWAYS = [
    "This is a forecast tool, not an official health advisory.",
    "Short spikes can be missed: 'no spike expected' does not guarantee clean air.",
]


@dataclass
class Advisory:
    tier: str
    headline: str
    summary: str
    actions: list
    caveats: list = field(default_factory=list)
    forecast_band: str = ""
    typical_range: tuple = (None, None)


def tier_array(spike_prob, pm25_now, alert_threshold):
    """Vectorised tier assignment (used by the dashboard and by tier_report.py)."""
    p = np.asarray(spike_prob, float)
    now = np.asarray(pm25_now, float)
    out = np.full(p.shape, "LOW_RISK", dtype=object)
    out[p >= alert_threshold] = "WARNING"
    out[now > SPIKE_LEVEL] = "ELEVATED_NOW"
    return out


def make_advisory(spike_prob, pm25_now, y_pred, lower, upper, alert_threshold):
    """Build the advice shown to the user for one forecast."""
    tier = str(tier_array(np.array([spike_prob]), np.array([pm25_now]), alert_threshold)[0])
    band = aqi.pm25_category(max(float(y_pred), 0.0))
    lo, hi = max(float(lower), 0.0), max(float(upper), 0.0)

    if tier == "ELEVATED_NOW":
        summary = (f"The latest reading is {pm25_now:.0f} µg/m³, above the {SPIKE_LEVEL:.0f} "
                   "level used for spike alerts.")
    else:
        summary = (f"Expected PM2.5 in 6 hours: about {y_pred:.0f} µg/m³ "
                   f"({BAND_NOTE[band]} for a short-term reading).")

    caveats = list(CAVEATS_ALWAYS)
    caveats.append(f"Typical range {lo:.0f}–{hi:.0f} µg/m³. This range can be too narrow "
                   "when a spike is developing.")
    if tier == "WARNING":
        caveats.append("Most warnings are not followed by a spike (about 2 in 3 at our test station "
                       "in 2019), so treat this as a heads-up, not a certainty.")

    return Advisory(tier, HEADLINES[tier], summary, ACTIONS[tier], caveats, band, (lo, hi))


def render_text(a):
    """Plain-text version, handy for tests and for the dashboard."""
    lines = [a.headline, a.summary, ""] + [f"- {x}" for x in a.actions] + [""]
    lines += [f"Note: {c}" for c in a.caveats]
    return "\n".join(lines)


if __name__ == "__main__":
    demo = make_advisory(spike_prob=0.62, pm25_now=55, y_pred=70, lower=35, upper=110,
                         alert_threshold=ALERT_THRESHOLDS[("xgb_ci", "KA003")])
    print(render_text(demo))