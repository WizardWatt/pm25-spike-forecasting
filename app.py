"""Aeris dashboard: replay of the 6-hour PM2.5 spike early warning.

Run from the project root:   streamlit run app.py
Reads only files already in outputs/ (no retraining). Tiers and advice come from advisory.py,
so the wording here is exactly what the evaluation used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import advisory as A
import eval_config as C
from alert_replay import find_episodes, lead_time, load_test_series

st.set_page_config(page_title="Aeris", page_icon="🌫️", layout="wide")

ROOT = Path(__file__).resolve().parent
PRED, REPORTS, SHAP = ROOT / "outputs" / "predictions", ROOT / "outputs" / "reports", ROOT / "outputs" / "shap"
MODEL = "xgb_ci"
H = pd.Timedelta(hours=1)
SPIKE = A.SPIKE_LEVEL
INK, MUTED, LINE, BLUE, GREY = "#12212B", "#5B6B76", "#D9E1E5", "#1F5FA8", "#9AA8B2"
TIER = {  # accent, tint, display name
    "LOW_RISK": ("#2F7F6F", "#E6F2EF", "Low risk"),
    "WARNING": ("#B97700", "#FFF3D6", "Warning"),
    "ELEVATED_NOW": ("#B3342A", "#FBE7E4", "High right now"),
}
FEATURE = {
    "pm25_now": "Reading right now", "pm25_max24": "Highest reading, last 24 h",
    "pm25_mean24": "Average, last 24 h", "pm25_std24": "Swings, last 24 h",
    "pm25_max6": "Highest reading, last 6 h", "pm25_mean6": "Average, last 6 h",
    "pm25_std6": "Swings, last 6 h", "pm25_trend3": "Change over last 3 h",
    "pm25_trend6": "Change over last 6 h", "pm25_lag1": "Reading 1 h ago",
    "pm25_lag2": "Reading 2 h ago", "pm25_lag3": "Reading 3 h ago", "pm25_lag6": "Reading 6 h ago",
    "pm25_lag12": "Reading 12 h ago", "pm25_lag24": "Reading 24 h ago",
    "hour": "Time of day", "month": "Time of year", "dayofweek": "Day of week",
}
GROUPS = {"hour": ["hour_sin", "hour_cos"], "month": ["month_sin", "month_cos"],
          "dayofweek": ["dayofweek", "is_weekend"]}
MODEL_NAME = {"persistence": "Persistence (next = now)", "hist_avg": "Historical average",
              "naive_alarm": "Rule: alert if already high", "xgb": "XGBoost", "rf": "Random forest",
              "gru": "GRU"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;650;700&display=swap');
/* Consistent, slightly larger typography across Streamlit widgets. */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMetric"] *, .stApp button, .stApp [data-baseweb="tab"],
.stApp [data-testid="stCaptionContainer"] {
  font-family: 'Schibsted Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-variant-numeric: tabular-nums;
}
.stApp p, .stApp li, .stApp label, .stApp [data-baseweb="tab"] { font-size: 1.08rem; }
.stApp h1 { font-size: 2.25rem; }
.stApp h2 { font-size: 1.9rem; }
.stApp h3, .stApp h4 { font-size: 1.45rem; }
.block-container { padding-top: 3rem !important; max-width: 1200px; }
.aeris-banner { border-left: 6px solid; border-radius: 4px; padding: 1rem 1.3rem; margin: .6rem 0 1.2rem; }
.aeris-banner .head { font-size: 1.9rem; font-weight: 700; line-height: 1.3; color: #12212B; }
.aeris-banner .sub { margin-top: .45rem; font-size: 1.2rem; line-height: 1.5; color: #12212B; }
.aeris-title { display: block; font-size: 2.5rem !important; font-weight: 700; line-height: 1.3 !important; padding: .15rem 0 .25rem; margin: 0 !important; overflow: visible; }
.aeris-sub { display: block; color: #5B6B76; font-size: 1.25rem !important; line-height: 1.55 !important; margin: .35rem 0 1.25rem !important; max-width: 100%; width: 100%; white-space: normal; overflow-wrap: anywhere; }
</style>
""", unsafe_allow_html=True)


# ---------- data ----------
@st.cache_data(show_spinner=False)
def load_forecasts(sid):
    d = pd.read_csv(PRED / f"{MODEL}_{sid}.csv", parse_dates=["Datetime"])
    return d[d["model"] == "refit"].drop_duplicates("Datetime").set_index("Datetime").sort_index()


@st.cache_data(show_spinner=False)
def load_shap(sid):
    d = pd.read_csv(SHAP / f"shap_rows_{sid}.csv.gz", parse_dates=["Datetime"])
    return d.drop_duplicates("Datetime").set_index("Datetime").sort_index()


@st.cache_data(show_spinner=False)
def load_replay(sid):
    pm, alert, _ = load_test_series(MODEL, sid)
    rows = []
    for start, end, peak_time, peak in find_episodes(pm):
        rows.append({"onset": start, "peak_time": peak_time, "peak": peak,
                     "hours": int((end - start) / H) + 1, "lead_h": lead_time(start, alert)})
    eps = pd.DataFrame(rows, columns=["onset", "peak_time", "peak", "hours", "lead_h"])
    return pm, alert, eps.sort_values("peak", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_report(name):
    return pd.read_csv(REPORTS / name)


def style(fig, height):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#FFFFFF", font=dict(family="Schibsted Grotesk, system-ui, sans-serif", color=INK),
                      legend=dict(orientation="h", y=1.14, x=0))
    fig.update_xaxes(gridcolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, linecolor=LINE)
    return fig


def show(fig):
    st.plotly_chart(fig)


# ---------- charts ----------
def forecast_chart(df, t, r, reveal):
    idx = pd.date_range(t - 24 * H, t, freq="h")
    past = df["pm25_now"].reindex(idx)
    fig = go.Figure()
    fig.add_vrect(x0=t, x1=t + 6 * H, fillcolor="#E4EBEE", opacity=0.6, line_width=0)
    fig.add_hline(y=SPIKE, line_dash="dash", line_color=GREY, annotation_text=f"Spike level ({SPIKE:.0f})",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=idx, y=past, mode="lines", name="Recorded PM2.5", line=dict(color=INK, width=2)))
    lo, hi = max(r.lower, 0), max(r.upper, 0)
    fig.add_trace(go.Scatter(x=[t + 6 * H], y=[r.y_pred], mode="markers", name="Forecast for +6 h",
                             marker=dict(color=BLUE, size=11),
                             error_y=dict(type="data", symmetric=False, array=[hi - r.y_pred],
                                          arrayminus=[r.y_pred - lo], color=BLUE, thickness=2, width=7)))
    if reveal:
        fut = df["pm25_now"].reindex(pd.date_range(t, t + 6 * H, freq="h"))
        fig.add_trace(go.Scatter(x=fut.index, y=fut, mode="lines", name="What actually happened",
                                 line=dict(color=INK, width=2, dash="dot")))
    fig.update_yaxes(title="PM2.5 (µg/m³)", rangemode="tozero")
    return style(fig, 340)


def why_chart(srow, t):
    contrib = {c[5:]: float(srow[c]) for c in srow.index if c.startswith("shap_")}
    members = {m for g in GROUPS.values() for m in g}
    items = []
    for f, v in contrib.items():
        if f in members:
            continue
        items.append((f, v, f"{srow['x_' + f]:.0f}" + (" µg/m³" if f != "pm25_trend3" and f != "pm25_trend6" else "")
                      if "x_" + f in srow.index else ""))
    for g, parts in GROUPS.items():
        v = sum(contrib.get(p, 0.0) for p in parts)
        shown = {"hour": f"{t:%H}:00", "month": f"{t:%B}", "dayofweek": f"{t:%A}"}[g]
        items.append((g, v, shown))
    items = sorted(items, key=lambda x: -abs(x[1]))[:7][::-1]
    labels = [f"{FEATURE.get(f, f)} ({val})" if val else FEATURE.get(f, f) for f, _, val in items]
    vals = [v for _, v, _ in items]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=["#B3342A" if v > 0 else "#2F7F6F" for v in vals],
                           text=[f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Pushed toward a warning (right) or away from one (left)", zeroline=True, zerolinecolor=INK)
    return style(fig, 330)


def episode_chart(pm, alert, ep):
    onset, peak_time = ep["onset"], ep["peak_time"]
    w = pm.loc[onset - 18 * H: peak_time + 18 * H]
    al = alert.loc[w.index]
    fig = go.Figure()
    fig.add_hline(y=SPIKE, line_dash="dash", line_color=GREY, annotation_text=f"Spike level ({SPIKE:.0f})",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=w.index, y=w, mode="lines", name="Recorded PM2.5", line=dict(color=INK, width=2)))
    hot = w[al]
    fig.add_trace(go.Scatter(x=hot.index, y=hot, mode="markers", name="Hour with an active warning",
                             marker=dict(color="#D98A00", size=8)))
    top = float(np.nanmax(w.values)) * 1.05
    fig.add_trace(go.Scatter(x=[onset, onset], y=[0, top], mode="lines", name="Spike begins",
                             line=dict(color="#B3342A", dash="dot", width=2)))
    fig.update_yaxes(title="PM2.5 (µg/m³)", rangemode="tozero")
    return style(fig, 340)


# ---------- page header and sidebar ----------
st.markdown('<p class="aeris-title">Aeris</p>', unsafe_allow_html=True)
st.markdown('<p class="aeris-sub">Early warning of PM2.5 spikes, six hours ahead. This dashboard replays recorded '
            'station data from 2019 and early 2020, so nothing here is live.</p>', unsafe_allow_html=True)

if not (PRED / f"{MODEL}_{C.STATIONS[0]}.csv").exists():
    st.error("Prediction files not found. Run this from the project root so outputs/predictions/ is next to app.py.")
    st.stop()

with st.sidebar:
    st.markdown("### Pick a moment")
    sid = st.radio("Station", C.STATIONS, format_func=lambda s: f"{s} (main)" if s == C.PRIMARY_STATION else f"{s} (second)")
    df = load_forecasts(sid)
    pm_series, alert_series, eps = load_replay(sid)
    thr = A.ALERT_THRESHOLDS[(MODEL, sid)]

    moments = {}
    for _, e in eps.head(3).iterrows():
        moments[f"Big spike, {e.peak:.0f} µg/m³ on {e.onset:%d %b %Y}"] = e.onset - 3 * H
    moments["Diwali evening, 27 Oct 2019"] = pd.Timestamp("2019-10-27 18:00")
    moments["Lockdown begins, 25 Mar 2020"] = pd.Timestamp("2020-03-25 09:00")
    moments = {k: df.index[df.index.get_indexer([v], method="nearest")[0]] for k, v in moments.items()}

    lo_d, hi_d = df.index.min().date(), df.index.max().date()
    if "day" not in st.session_state:
        first = next(iter(moments.values()))
        st.session_state["day"], st.session_state["hr"] = first.date(), first.hour
    st.session_state["day"] = min(max(st.session_state["day"], lo_d), hi_d)

    def _jump(options, key):
        t_ = options[st.session_state[key]]
        st.session_state["day"], st.session_state["hr"] = t_.date(), t_.hour

    jkey = f"jump_{sid}"
    st.selectbox("Jump to a notable moment", list(moments), index=None, placeholder="Choose one",
                 key=jkey, on_change=_jump, args=(moments, jkey))
    st.date_input("Day", key="day", min_value=lo_d, max_value=hi_d)
    st.slider("Hour of day", 0, 23, key="hr", format="%d:00")
    st.caption("The forecast is made at the hour you pick and covers the six hours after it.")

t = pd.Timestamp(st.session_state["day"]) + st.session_state["hr"] * H
tab_fc, tab_rp, tab_cmp, tab_trust = st.tabs(["Forecast", "Past alerts", "Model comparison", "How far to trust it"])

# ---------- Forecast ----------
with tab_fc:
    st.markdown(f"#### {t:%A, %d %B %Y, %H:%M} at {sid}")
    if t not in df.index:
        st.info("No forecast exists for this hour, usually because the sensor had a gap. Try the hour before or after.")
    else:
        r = df.loc[t]
        adv = A.make_advisory(r.spike_prob, r.pm25_now, r.y_pred, r.lower, r.upper, thr)
        accent, tint, _ = TIER[adv.tier]
        st.markdown(f'<div class="aeris-banner" style="border-color:{accent};background:{tint}">'
                    f'<div class="head">{adv.headline}</div><div class="sub">{adv.summary}</div></div>',
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Reading now", f"{r.pm25_now:.0f} µg/m³",
                  help="Latest hourly PM2.5. CPCB bands are defined on 24-hour averages, so an hourly band is only indicative.")
        m2.metric("Expected in 6 hours", f"{r.y_pred:.0f} µg/m³")
        m3.metric("Typical range", f"{adv.typical_range[0]:.0f} to {adv.typical_range[1]:.0f} µg/m³",
                  help="Meant to hold about 90% of outcomes, but it is too narrow when a spike is building (see the last tab).")

        left, right = st.columns([3, 2], gap="large")
        with left:
            reveal = st.checkbox("Show what actually happened next", key="reveal")
            show(forecast_chart(df, t, r, reveal))
            if reveal:
                warned, spike = r.spike_prob >= thr, r.y_spike == 1
                st.markdown({(True, True): "**Right call.** PM2.5 went above 90 within 6 hours.",
                             (True, False): "**False alarm.** PM2.5 stayed at or below 90.",
                             (False, True): "**Missed.** PM2.5 went above 90 and no warning was raised.",
                             (False, False): "**Correctly quiet.** PM2.5 stayed at or below 90."}[(bool(warned), bool(spike))])
        with right:
            st.markdown("**What to do**")
            for a in adv.actions:
                st.markdown(f"- {a}")
            with st.expander("What this forecast cannot tell you"):
                for c in adv.caveats:
                    st.markdown(f"- {c}")
            with st.expander("Technical details"):
                st.markdown(f"Alert score **{r.spike_prob:.2f}**, warning cut-off **{thr:.3f}** for {sid}, chosen on 2018 data. "
                            "The score ranks hours from calmer to riskier. It is not a probability, so it is never shown as a percentage.")

        st.markdown("#### What drove this forecast")
        shap_df = load_shap(sid)
        if t in shap_df.index:
            show(why_chart(shap_df.loc[t], t))
            st.caption("SHAP explanation of the XGBoost spike classifier. Bar sizes are in the model's internal units, "
                       "so compare bars with each other, not with a percentage.")
        else:
            st.caption("No stored explanation for this hour. Explanations cover 2019 onward.")
        st.caption("General information, not medical advice.")

# ---------- Past alerts ----------
with tab_rp:
    comp = read_report("compare_test.csv")
    cr = comp[(comp["model"] == MODEL) & (comp["station"] == sid)].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes warned in time, 2019", f"{cr.episode_recall:.0%}", help=f"Out of {int(cr.n_episodes)} episodes in the evaluation report.")
    c2.metric("False-alarm runs per week", f"{cr.false_alert_runs_per_week:.1f}")
    c3.metric("Episodes in the report", f"{int(cr.n_episodes)}")
    st.caption("Headline numbers come from the evaluation report. The list below counts each unbroken run above 90 as one "
               "episode, so its count differs from the report's. Use it to browse individual cases.")
    if eps.empty:
        st.info("No episodes above the spike level in this window.")
    else:
        pick = st.selectbox("Episode", list(eps.index), format_func=lambda i: (
            f"{eps.onset[i]:%d %b %Y %H:%M}, peak {eps.peak[i]:.0f} µg/m³, "
            + ("not warned in time" if pd.isna(eps.lead_h[i]) else f"warned {eps.lead_h[i]:.0f} h ahead")))
        show(episode_chart(pm_series, alert_series, eps.loc[pick]))
        tbl = pd.DataFrame({"Spike begins": eps.onset.dt.strftime("%d %b %Y %H:%M"),
                            "Peak (µg/m³)": eps.peak.round(0).astype(int), "Hours above 90": eps.hours,
                            "Warning ahead": eps.lead_h.map(lambda x: "no warning" if pd.isna(x) else f"{x:.0f} h")})
        st.dataframe(tbl, hide_index=True)

# ---------- Model comparison ----------
with tab_cmp:
    cmp_ = read_report("compare_test_far0.05.csv")
    m = cmp_[cmp_["station"] == sid].copy()
    m["name"] = m["model"].map(MODEL_NAME)
    colors = [BLUE if x == "xgb" else GREY for x in m["model"]]
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Average forecast error** (lower is better)")
        show(style(go.Figure(go.Bar(x=m["name"], y=m["MAE"], marker_color=colors, text=m["MAE"].round(1), textposition="outside")), 300)
             .update_yaxes(title="µg/m³"))
    with cb:
        st.markdown("**Spike episodes warned in time** (higher is better)")
        show(style(go.Figure(go.Bar(x=m["name"], y=m["episode_recall"], marker_color=colors,
                                    text=m["episode_recall"].map("{:.0%}".format), textposition="outside")), 300)
             .update_yaxes(tickformat=".0%", range=[0, 1]))
    base = m.set_index("model")
    if {"xgb", "persistence"} <= set(base.index):
        gain = 1 - base.loc["xgb", "MAE"] / base.loc["persistence", "MAE"]
        st.markdown(f"At {sid} in 2019, XGBoost's average error was **{gain:.0%} lower** than simply assuming the next reading equals the current one.")
    out = pd.DataFrame({"Model": m["name"], "Avg error (µg/m³)": m["MAE"].round(1),
                        "Spike hours caught": m["recall"].map("{:.0%}".format),
                        "Warnings that were right": m["precision"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}"),
                        "False-alarm rate": m["false_alarm_rate"].map("{:.1%}".format),
                        "False-alarm runs per week": m["false_alert_runs_per_week"].round(1),
                        "Episodes warned in time": m["episode_recall"].map("{:.0%}".format)})
    st.dataframe(out, hide_index=True)
    st.caption("Every model gets the same rule: its alert cut-off is set on 2018 so about 5% of calm hours raise a false alarm, "
               "then used unchanged on 2019. The Forecast tab uses XGBoost with intervals and a cut-off chosen on 2018 by F2 score.")

# ---------- Trust ----------
with tab_trust:
    tr = read_report("tier_report.csv")
    tr = tr[tr["station"] == sid]
    fig = go.Figure()
    for label, color in [("2018 val", GREY), ("2019 test", BLUE)]:
        g = tr[tr["set"] == label].set_index("tier").reindex(A.TIERS)
        fig.add_trace(go.Bar(x=[TIER[x][2] for x in A.TIERS], y=g["spike_followed"], name=label.replace("val", "(used to set the cut-off)").replace("test", "(unseen)"),
                             marker_color=color, text=g["spike_followed"].map("{:.0%}".format), textposition="outside"))
    fig.update_layout(barmode="group").update_yaxes(tickformat=".0%", range=[0, 1], title="Share followed by a spike")
    st.markdown("**When the dashboard shows each level, how often did PM2.5 go above 90 in the next 6 hours?**")
    show(style(fig, 320))
    w = tr[(tr["set"] == "2019 test") & (tr["tier"] == "WARNING")].iloc[0]
    st.markdown(f"A warning at {sid} was followed by a spike in **{w.spike_followed:.0%}** of 2019 hours, so treat it as a heads-up "
                "that raises your odds, not as a prediction that a spike will happen.")

    test = df[(df["split"] == "test") & ~df["target_imputed"]]
    hit = test[test["y_true"] > SPIKE]
    cov_all = ((test["lower"] <= test["y_true"]) & (test["y_true"] <= test["upper"])).mean()
    cov_spk = ((hit["lower"] <= hit["y_true"]) & (hit["y_true"] <= hit["upper"])).mean()
    st.markdown(f"**The typical range** is built to hold 90% of outcomes. In 2019 it held {cov_all:.0%} overall, "
                f"but only {cov_spk:.0%} of the time when PM2.5 really went above 90. Never read its upper end as a ceiling.")

    st.markdown("**When conditions change: Diwali and the 2020 lockdown**")
    sh = read_report("shift_tests.csv")
    sh = sh[(sh["station"] == sid) & (sh["model"] == MODEL)]
    order = ["rest of 2018 (no Diwali)", "Diwali 2018", "rest of 2019 (no Diwali)", "Diwali 2019",
             "2019, same dates as lockdown", "lockdown 2020"]
    sh = sh.set_index("window").reindex(order).reset_index()
    st.dataframe(pd.DataFrame({"Window": sh["window"], "Hours": sh["n"], "Average PM2.5 (µg/m³)": sh["mean_truth"].round(0),
                               "Avg error (µg/m³)": sh["MAE"].round(1), "Range coverage": sh["coverage"].map("{:.0%}".format),
                               "False-alarm rate": sh["false_alarm_rate"].map("{:.1%}".format)}), hide_index=True)
    lk, ref = sh.set_index("window").loc["lockdown 2020"], sh.set_index("window").loc["2019, same dates as lockdown"]
    st.caption(f"Lockdown hours averaged {lk.mean_truth:.0f} µg/m³ against {ref.mean_truth:.0f} on the same dates in 2019. "
               f"The two Diwali windows are only {int(sh.set_index('window').loc['Diwali 2018', 'n'])} and "
               f"{int(sh.set_index('window').loc['Diwali 2019', 'n'])} hours, so read them as hints, not conclusions.")
    st.caption("Hourly PM2.5 is mapped to CPCB bands even though CPCB defines them on 24-hour averages. "
               "This is a forecast tool, not an official health advisory.")