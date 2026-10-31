"""
Solar PV Performance Monitor  -  Streamlit dashboard
====================================================
Features
 1. Expected-power prediction (irradiation, temperature, time -> AC power)
 2. Multiple-model comparison (LR, Decision Tree, Random Forest, HistGradientBoosting)
    with MAE, RMSE, R2 and MAPE
 3. Combined anomaly detection (residual based + Isolation Forest)
 4. Sensor-quality monitoring (missing, impossible, stuck readings)
 5. Anomaly events with duration, estimated energy loss and revenue loss
 6. Explanation & reporting (feature importance, event vs normal, optional SHAP,
    what-if simulator, CSV downloads)
 7. SQLite maintenance log and plant health score (with solar-panel graphic)

Run:   pip install -r requirements.txt
       streamlit run app.py

Data:  Put the Kaggle CSVs (Plant_1_Generation_Data.csv, Plant_1_Weather_Sensor_Data.csv)
       inside  data/  to use real data.  If data/ is empty, 2026 demo data is generated
       automatically in sample_data/.
"""
import os
from pathlib import Path
import sqlite3
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingRegressor, IsolationForest,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("SOLAR_DATA_DIR", BASE / "data"))
SAMPLE_DIR = Path(os.environ.get("SOLAR_SAMPLE_DIR", BASE / "sample_data"))
ASSET_PHOTO = BASE / "assets" / "solar_panel.jpg"   # optional: your own panel photo
DB_PATH = Path(os.environ.get("SOLAR_MAINT_DB", BASE / "maintenance.db"))

FEATURES = ["irr", "amb", "mod", "hour"]
FEATURE_LABELS = {"irr": "Irradiation", "amb": "Ambient temp",
                  "mod": "Module temp", "hour": "Time of day"}
STATUS_OPTIONS = ["Open", "Acknowledged", "In progress", "Resolved", "False alarm"]

st.set_page_config(page_title="Solar PV Performance Monitor", page_icon="☀️",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
.stApp {background:#f4f6fb;}
.block-container {padding-top:1.2rem; max-width:1400px;}
h1,h2,h3,h4,p,label,span,div {color:#1f2937;}
[data-testid="stSidebar"] {background:#0f172a;}
[data-testid="stSidebar"] * {color:#e2e8f0 !important;}
.hero {background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#f59e0b 130%);
  border-radius:18px; padding:26px 32px; margin-bottom:18px;
  box-shadow:0 10px 30px rgba(15,23,42,.25);}
.hero h1 {color:#fff !important; margin:0; font-size:2rem; font-weight:800;}
.hero p {color:#dbeafe !important; margin:4px 0 0 0; font-size:.98rem;}
.kpi {background:#fff; border-radius:14px; padding:16px 18px; height:100%;
  box-shadow:0 2px 10px rgba(15,23,42,.07); border-left:5px solid var(--c,#3b82f6);}
.kpi-l {font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:#64748b !important; font-weight:600;}
.kpi-v {font-size:1.7rem; font-weight:800; color:#0f172a !important; line-height:1.25;}
.kpi-s {font-size:.8rem; color:#64748b !important;}
.card {background:#fff; border-radius:16px; padding:18px 20px;
  box-shadow:0 2px 12px rgba(15,23,42,.07); margin-bottom:12px;}
.card h3 {margin:0 0 6px 0; font-size:1.05rem;}
.pill {display:inline-block; padding:3px 12px; border-radius:999px; color:#fff !important;
  font-weight:700; font-size:.8rem;}
.chk {display:flex; justify-content:space-between; align-items:center; padding:9px 0;
  border-bottom:1px solid #eef2f7; font-size:.92rem;}
.chk:last-child {border-bottom:none;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px;}
.panelwrap {display:flex; justify-content:center;}
button[data-baseweb="tab"] {font-weight:600;}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div {color:#000 !important; -webkit-text-fill-color:#000 !important; caret-color:#000 !important; font-weight:600;}
[data-testid="stSidebar"] [data-baseweb="input"], [data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#ffffff !important;}
[data-testid="stSidebar"] [data-baseweb="input"] svg, [data-testid="stSidebar"] [data-baseweb="select"] svg {fill:#000 !important; color:#000 !important;}
[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"], [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"],
[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] *, [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] * {color:#000 !important; fill:#000 !important;}
.pbar {background:#e5e7eb; border-radius:999px; height:18px; overflow:hidden; margin:10px 0 6px 0;}
.pfill {height:100%; border-radius:999px;}
.rec {background:#fff; border-radius:12px; padding:12px 16px; margin-bottom:10px; border-left:6px solid var(--c,#3b82f6); box-shadow:0 2px 8px rgba(15,23,42,.06);}
.rec b {color:#0f172a !important;} .rec small {color:#64748b !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def show_plot(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.18))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def kpi(label, value, sub="", color="#3b82f6"):
    return (f'<div class="kpi" style="--c:{color}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


# ----------------------------------------------------------------------------
# Demo data (2026) - generated in the same format as the Kaggle files
# ----------------------------------------------------------------------------
SAMPLE_CFG = [
    dict(id=4135001, name="Plant_1", seed=2026, wkey="HmiyD2TTLFNqkNe", gen_fmt="%d-%m-%Y %H:%M",
         keys=["1BY6WEcLGh8j5v7", "1IF53ai7Xc0U56Y", "3PZuoBAID5Wc2HD", "7JYdWkrLSPkdwr4"],
         eff=[1.0, 0.99, 1.01, 0.985], rated=1250),
    dict(id=4136001, name="Plant_2", seed=4136, wkey="iq8k7ZNt4Mwm3w0", gen_fmt="%Y-%m-%d %H:%M:%S",
         keys=["4UPUqMRk7TRMgml", "81aHJ1q11NBPMrL", "9kRcWv60rDACzjR", "Et9kgGMDl729KT4", "IQ2d7wF4YD8zU1Q"],
         eff=[1.0, 1.01, 0.99, 1.0, 0.995], rated=1100),
]


def _make_plant(cfg):
    rng = np.random.default_rng(cfg["seed"])
    ts = pd.date_range("2026-05-01 00:00", "2026-05-21 23:45", freq="15min")
    n = len(ts)
    h = np.asarray(ts.hour + ts.minute / 60)
    x = (h - 6.0) / 12.5
    sun = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 1.2, 0.0)
    cf = {d: rng.uniform(0.7, 1.0) for d in sorted(set(ts.normalize()))}
    cloud = np.array([cf[d] for d in ts.normalize()])
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.03)
    irr_true = np.clip(1.08 * sun * cloud * (1 + noise), 0, 1.3)
    irr_true[sun == 0] = 0
    amb = 26 + 6 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.4, n)
    mod = amb + 24 * irr_true + rng.normal(0, 0.5, n)
    t = pd.Series(ts)

    def win(a, b):
        return ((t >= a) & (t <= b)).values

    p1 = cfg["name"] == "Plant_1"
    frames = []
    for i, key in enumerate(cfg["keys"]):
        ac = cfg["rated"] * irr_true * (1 - 0.004 * (mod - 25)) * cfg["eff"][i] * (1 + rng.normal(0, 0.012, n))
        ac = np.clip(ac, 0, cfg["rated"] * 1.04)
        ac[irr_true == 0] = 0
        if p1:
            if i == 1:    # inverter trip / outage
                ac[win("2026-05-09 10:30", "2026-05-09 13:00")] = 0
            if i == 2:    # derating on 3 afternoons + one impossible spike
                m = ((t >= "2026-05-14") & (t < "2026-05-17") & (t.dt.hour >= 11) & (t.dt.hour < 15)).values
                ac[m] *= 0.62
                ac[(t == "2026-05-11 12:15").values] = 4200
            if i == 0:    # gradual loss (soiling) + impossible negative
                ac[((t >= "2026-05-18") & (t < "2026-05-22")).values] *= 0.84
                ac[(t == "2026-05-06 02:15").values] = -35
            if i == 3:    # stuck reading
                m = win("2026-05-19 12:00", "2026-05-19 14:00")
                ac[m] = ac[m][0]
        else:
            if i == 0:
                ac[win("2026-05-17 09:00", "2026-05-17 11:30")] = 0
            if i == 2:
                m = win("2026-05-11 12:00", "2026-05-11 15:00")
                ac[m] *= 0.7
            if i == 4:
                ac[((t >= "2026-05-15") & (t < "2026-05-22")).values] *= 0.80
            if i == 1:
                m = win("2026-05-19 12:00", "2026-05-19 14:00")
                ac[m] = ac[m][0]
            if i == 3:
                ac[(t == "2026-05-13 12:15").values] = -20
        ac = np.round(ac, 3)
        g = pd.DataFrame({"DATE_TIME": ts, "PLANT_ID": cfg["id"], "SOURCE_KEY": key,
                          "DC_POWER": np.round(ac / 0.97, 3), "AC_POWER": ac})
        g["DAILY_YIELD"] = (g.groupby(g.DATE_TIME.dt.date)["AC_POWER"].cumsum() * 0.25).round(3)
        g["TOTAL_YIELD"] = (6.0e6 + g["AC_POWER"].cumsum() * 0.25).round(3)
        frames.append(g)
    gen = pd.concat(frames)
    gen["DATE_TIME"] = gen["DATE_TIME"].dt.strftime(cfg["gen_fmt"])

    irr, modr = irr_true.copy(), mod.copy()
    if p1:
        irr[win("2026-05-07 11:00", "2026-05-07 13:00")] = 0.55      # stuck irradiation sensor
        irr[win("2026-05-12 09:00", "2026-05-12 09:45")] = np.nan    # missing readings
    else:
        modr[win("2026-05-10 10:00", "2026-05-11 16:00")] = amb[win("2026-05-10 10:00", "2026-05-11 16:00")] - 6
        modr[win("2026-05-13 08:00", "2026-05-13 08:30")] = np.nan
    wea = pd.DataFrame({"DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"), "PLANT_ID": cfg["id"],
                        "SOURCE_KEY": cfg["wkey"], "AMBIENT_TEMPERATURE": np.round(amb, 3),
                        "MODULE_TEMPERATURE": np.round(modr, 3), "IRRADIATION": np.round(irr, 4)})
    return gen, wea


def ensure_sample():
    files = [SAMPLE_DIR / f"{c['name']}_{kind}.csv" for c in SAMPLE_CFG
             for kind in ("Generation_Data", "Weather_Sensor_Data")]
    if all(f.exists() for f in files):
        return
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in SAMPLE_CFG:
        gen, wea = _make_plant(cfg)
        gen.to_csv(SAMPLE_DIR / f"{cfg['name']}_Generation_Data.csv", index=False)
        wea.to_csv(SAMPLE_DIR / f"{cfg['name']}_Weather_Sensor_Data.csv", index=False)


def _parse_dt(s):
    out = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s, errors="coerce")
    return out


def _read_all(files):
    """Each Kaggle file is parsed on its own (Plant 1 generation is day-first, Plant 2 is ISO)."""
    frames = []
    for f in files:
        x = pd.read_csv(f)
        x["ts"] = _parse_dt(x["DATE_TIME"])
        frames.append(x)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Loading data...")
def load_data():
    gen_files = sorted(DATA_DIR.glob("*Generation*.csv")) if DATA_DIR.exists() else []
    wea_files = sorted(DATA_DIR.glob("*Weather*.csv")) if DATA_DIR.exists() else []
    if not (gen_files and wea_files):
        ensure_sample()
        gen_files = sorted(SAMPLE_DIR.glob("*Generation*.csv"))
        wea_files = sorted(SAMPLE_DIR.glob("*Weather*.csv"))
    gen = _read_all(gen_files)
    wea = _read_all(wea_files)
    wea = (wea.groupby(["PLANT_ID", "ts"], as_index=False)
              [["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].mean())
    df = gen.merge(wea, on=["PLANT_ID", "ts"], how="left")
    df = df.rename(columns={"AC_POWER": "ac", "DC_POWER": "dc", "AMBIENT_TEMPERATURE": "amb",
                            "MODULE_TEMPERATURE": "mod", "IRRADIATION": "irr",
                            "PLANT_ID": "plant"})
    df = df.dropna(subset=["ts"]).sort_values(["plant", "SOURCE_KEY", "ts"])
    df["inverter"] = ("INV-" + (df.groupby("plant")["SOURCE_KEY"]
                                  .transform(lambda s: pd.factorize(s)[0] + 1)
                                  .astype(int).astype(str).str.zfill(2)))
    df["hour"] = df.ts.dt.hour + df.ts.dt.minute / 60
    return df[["ts", "plant", "inverter", "SOURCE_KEY", "ac", "dc", "irr", "amb", "mod", "hour"]] \
        .reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Sensor-quality monitoring
# ----------------------------------------------------------------------------
def _stuck(s, min_run, floor):
    rid = s.ne(s.shift()).cumsum()
    size = s.groupby(rid).transform("size")
    return (size >= min_run) & (s > floor)


@st.cache_data(show_spinner=False)
def add_quality(df):
    d = df.sort_values(["inverter", "ts"]).reset_index(drop=True).copy()
    d["daytime"] = d["irr"] >= 0.05
    rated = d.groupby("inverter")["ac"].quantile(0.99).median() * 1.5
    d["q_missing"] = d[["ac", "irr", "amb", "mod"]].isna().any(axis=1)
    d["q_impossible"] = ((d.ac < 0) | (d.ac > rated) | (d.irr < 0) | (d.irr > 1.6) |
                         (d["mod"] < -20) | (d["mod"] > 110) | (d.amb < -20) | (d.amb > 65))
    stuck_ac = d.groupby("inverter", group_keys=False)["ac"].apply(lambda s: _stuck(s, 6, 1.0))
    stuck_irr = d.groupby("inverter", group_keys=False)["irr"].apply(lambda s: _stuck(s, 6, 0.02))
    d["q_stuck"] = stuck_ac | stuck_irr
    d["q_any"] = d.q_missing | d.q_impossible | d.q_stuck
    reason = np.where(d.q_missing, "Missing", np.where(d.q_impossible, "Impossible",
                      np.where(d.q_stuck, "Stuck", "")))
    d["q_reason"] = reason
    return d


# ----------------------------------------------------------------------------
# 1 & 2. Expected-power prediction + model comparison
# ----------------------------------------------------------------------------
def _mape(y, p):
    m = y > 0.05 * y.max()
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


@st.cache_data(show_spinner="Training and comparing models...")
def fit_models(d):
    day = d[d.daytime & ~d.q_any & d[FEATURES + ["ac"]].notna().all(axis=1)].sort_values("ts")
    cut = day.ts.quantile(0.7)
    tr, te = day[day.ts <= cut], day[day.ts > cut]
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=0),
        "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                                               n_jobs=-1, random_state=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=0),
    }
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(tr[FEATURES], tr["ac"])
        p = np.clip(m.predict(te[FEATURES]), 0, None)
        preds[name] = p
        rows.append({"Model": name,
                     "MAE (kW)": mean_absolute_error(te.ac, p),
                     "RMSE (kW)": float(np.sqrt(mean_squared_error(te.ac, p))),
                     "R²": r2_score(te.ac, p),
                     "MAPE (%)": _mape(te.ac.to_numpy(), p)})
    metrics = pd.DataFrame(rows).sort_values("RMSE (kW)").reset_index(drop=True)
    best = metrics.loc[0, "Model"]

    # final model: fit on all clean data, drop clear under-performers, refit (robust)
    final = clone(models[best]).fit(day[FEATURES], day["ac"])
    p0 = np.clip(final.predict(day[FEATURES]), 0, None)
    rel = (day.ac.to_numpy() - p0) / np.maximum(p0, 0.1 * np.percentile(p0, 99))
    keep = rel > -0.25
    final = clone(models[best]).fit(day.loc[keep, FEATURES], day.loc[keep, "ac"])

    samp = day.sample(min(len(day), 1500), random_state=0)
    pi = permutation_importance(final, samp[FEATURES], samp["ac"], n_repeats=5,
                                random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in FEATURES],
                        "Importance": pi.importances_mean}).sort_values("Importance")
    test = pd.DataFrame({"ts": te.ts.to_numpy(), "actual": te.ac.to_numpy(), "pred": preds[best]})
    return {"metrics": metrics, "best": best, "model": final, "importance": imp,
            "test": test, "train_rows": len(tr), "test_rows": len(te)}


# ----------------------------------------------------------------------------
# 3. Combined anomaly detection
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Detecting anomalies...")
def detect(d, _model, k, min_dev, contamination, mode, min_len):
    d = d.copy()
    ok = d.daytime & d[FEATURES].notna().all(axis=1)
    d["pred"] = np.nan
    d.loc[ok, "pred"] = np.clip(_model.predict(d.loc[ok, FEATURES]), 0, None)
    cap = np.nanpercentile(d["pred"].dropna(), 99)
    denom = np.maximum(d["pred"], 0.1 * cap)
    d["res"] = d.ac - d.pred
    d["res_pct"] = d.res / denom
    base = ok & d.ac.notna() & ~d.q_any & (d.pred >= 0.15 * cap)   # ignore dawn/dusk trickle power
    ref = d.loc[base, "res_pct"]
    med = ref.median()
    mad = max(1.4826 * np.median(np.abs(ref - med)), 0.01)
    d["z"] = (d.res_pct - med) / mad
    d["flag_res"] = base & (d.z.abs() > k) & (d.res_pct.abs() > min_dev)

    d["flag_iso"] = False
    d["iso_score"] = np.nan
    if base.sum() > 50:
        X = d.loc[base, ["irr", "amb", "mod", "ac", "res_pct", "hour"]]
        Xs = StandardScaler().fit_transform(X)
        iso = IsolationForest(n_estimators=150, contamination=contamination,
                              random_state=0).fit(Xs)
        d.loc[base, "flag_iso"] = iso.predict(Xs) == -1
        d.loc[base, "iso_score"] = -iso.score_samples(Xs)

    if mode == "Residual only":
        a = d.flag_res
    elif mode == "Isolation Forest only":
        a = d.flag_iso
    else:   # combined
        a = d.flag_res | (d.flag_iso & (d.z.abs() > k / 2))
    a = (a & base).astype(bool)
    key = (a.ne(a.shift()) | d.inverter.ne(d.inverter.shift())).cumsum()
    d["anomaly"] = a & (a.groupby(key).transform("size") >= min_len)   # drop 1-reading blips
    d["status"] = np.select([d.q_any, d.anomaly, d.daytime], ["Sensor issue", "Anomaly", "Normal"],
                            default="Night")
    return d


# ----------------------------------------------------------------------------
# 5. Events and estimated loss
# ----------------------------------------------------------------------------
def build_events(d, step_min, tariff, ptag="P1"):
    step_h = step_min / 60
    out = []
    for inv, g in d.groupby("inverter"):
        g = g.sort_values("ts").reset_index(drop=True)
        a = g["anomaly"]
        filled = a | (a.shift(1, fill_value=False) & a.shift(-1, fill_value=False))
        gid = (filled != filled.shift()).cumsum()
        for _, sub in g[filled].groupby(gid[filled]):
            an = sub[sub.anomaly]
            lost = float(np.clip(sub.pred - sub.ac, 0, None).sum() * step_h)
            short = float(-an.res_pct.mean() * 100)
            ratio = float(an.ac.sum() / max(an.pred.sum(), 1e-6))
            if short < 0:
                typ = "Over-reading (check sensor)"
            elif ratio < 0.1:
                typ = "Outage / trip"
            elif short > 30:
                typ = "Severe derating"
            else:
                typ = "Under-performance"
            sev = "High" if short > 50 else ("Medium" if short > 25 else "Low")
            out.append({"Event ID": f"{ptag}-{inv}_{sub.ts.min():%Y%m%d%H%M}", "Inverter": inv,
                        "Start": sub.ts.min(), "End": sub.ts.max() + pd.Timedelta(minutes=step_min),
                        "Duration (min)": len(sub) * step_min, "Type": typ, "Severity": sev,
                        "Avg shortfall (%)": round(short, 1),
                        "Lost energy (kWh)": round(lost, 1),
                        "Lost revenue (₹)": round(lost * tariff, 0)})
    cols = ["Event ID", "Inverter", "Start", "End", "Duration (min)", "Type", "Severity",
            "Avg shortfall (%)", "Lost energy (kWh)", "Lost revenue (₹)"]
    ev = pd.DataFrame(out, columns=cols)
    return ev.sort_values("Start").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 7. Health score
# ----------------------------------------------------------------------------
def health_score(d, step_h):
    day = d[d.daytime]
    n = max(len(day), 1)
    anom_pct = day.anomaly.sum() / n * 100
    expected = day.pred.sum() * step_h
    lost = np.clip(day.pred - day.ac, 0, None)[day.anomaly].sum() * step_h
    loss_pct = lost / expected * 100 if expected > 0 else 0.0
    sens_pct = d.q_any.sum() / max(len(d), 1) * 100
    p_loss = min(45.0, loss_pct * 4)
    p_anom = min(35.0, anom_pct * 3)
    p_sens = min(20.0, sens_pct * 2)
    score = float(np.clip(100 - p_loss - p_anom - p_sens, 0, 100))
    return {"score": score, "anom_pct": anom_pct, "loss_pct": loss_pct, "sens_pct": sens_pct,
            "lost_kwh": lost, "expected_kwh": expected,
            "p_loss": p_loss, "p_anom": p_anom, "p_sens": p_sens}


def band(score):
    if score >= 85:
        return "Excellent", "#16a34a"
    if score >= 70:
        return "Good", "#65a30d"
    if score >= 50:
        return "Needs attention", "#f59e0b"
    return "Critical", "#dc2626"


def panel_svg(score):
    """Solar-panel illustration; the number of greyed/damaged cells follows the health score."""
    label, color = band(score)
    rows, cols = 6, 10
    n = rows * cols
    bad = int(round((100 - score) / 100 * n))
    rng = np.random.default_rng(int(score * 10))
    bad_idx = set(rng.choice(n, size=min(bad, n), replace=False).tolist())
    x0, y0, gap = 56, 58, 3.4
    cw = (310 - gap * (cols - 1)) / cols
    ch = (186 - gap * (rows - 1)) / rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            x, y = x0 + c * (cw + gap), y0 + r * (ch + gap)
            fill = "url(#cellBad)" if i in bad_idx else "url(#cellOK)"
            stroke = "#f59e0b" if i in bad_idx else "#60a5fa"
            cells.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" rx="2" '
                         f'fill="{fill}" stroke="{stroke}" stroke-width=".7"/>')
            cells.append(f'<line x1="{x + cw / 2:.1f}" y1="{y:.1f}" x2="{x + cw / 2:.1f}" '
                         f'y2="{y + ch:.1f}" stroke="#ffffff" stroke-opacity=".18" stroke-width=".6"/>')
            if i in bad_idx:
                cells.append(f'<path d="M{x + 3:.1f} {y + 3:.1f} L{x + cw / 2:.1f} {y + ch / 2:.1f} '
                             f'L{x + cw - 4:.1f} {y + ch - 3:.1f}" stroke="#fde68a" stroke-width=".9" '
                             f'fill="none" stroke-opacity=".8"/>')
    rays = "".join(
        f'<line x1="{372 + 27 * np.cos(a):.1f}" y1="{34 + 27 * np.sin(a):.1f}" '
        f'x2="{372 + 36 * np.cos(a):.1f}" y2="{34 + 36 * np.sin(a):.1f}" stroke="{color}" '
        f'stroke-width="3" stroke-linecap="round"/>' for a in np.linspace(0, 2 * np.pi, 13)[:-1])
    svg = (
        '<div class="panelwrap"><svg viewBox="0 0 430 340" width="100%" style="max-width:520px" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<defs>'
        '<linearGradient id="cellOK" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#2563eb"/><stop offset="1" stop-color="#1e3a8a"/></linearGradient>'
        '<linearGradient id="cellBad" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#94a3b8"/><stop offset="1" stop-color="#475569"/></linearGradient>'
        f'<radialGradient id="sunG"><stop offset="0" stop-color="#fff7cc"/><stop offset="1" stop-color="{color}"/></radialGradient>'
        '<linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#fff" stop-opacity=".28"/><stop offset=".45" stop-color="#fff" stop-opacity="0"/></linearGradient>'
        '</defs>'
        '<rect x="0" y="0" width="430" height="340" rx="18" fill="#eff6ff"/>'
        f'{rays}<circle cx="372" cy="34" r="20" fill="url(#sunG)"/>'
        '<g transform="translate(26,0) skewX(-6)">'
        '<rect x="46" y="296" width="9" height="30" fill="#64748b"/>'
        '<rect x="345" y="296" width="9" height="30" fill="#64748b"/>'
        '<rect x="40" y="46" width="322" height="210" rx="7" fill="#e2e8f0" stroke="#94a3b8" stroke-width="3"/>'
        + "".join(cells) +
        '<rect x="40" y="46" width="322" height="210" rx="7" fill="url(#sheen)"/>'
        '<rect x="40" y="262" width="322" height="26" rx="5" fill="#cbd5e1" stroke="#94a3b8"/>'
        '<rect x="46" y="296" width="308" height="5" fill="#94a3b8"/>'
        '</g>'
        f'<text x="215" y="283" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14" '
        f'font-weight="700" fill="#1f2937">Panel condition: {label}  ·  {n - len(bad_idx)}/{n} cells healthy</text>'
        '</svg></div>')
    return svg


# ----------------------------------------------------------------------------
# 7. Maintenance Centre: SQLite storage, indicators, score, recommendations
# ----------------------------------------------------------------------------
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
MSTATUS = ["Open", "In Progress", "Completed", "Follow-up Required"]
CATEGORIES = ["Power anomaly", "Underperformance", "Temperature abnormality", "Sensor fault / calibration",
              "Inverter fault", "Panel soiling / shading", "Wiring / connections", "Routine inspection", "Other"]
SUGGESTED_ACTION = {
    "Power anomaly": "Inspect panels, shading, connections and inverter performance.",
    "Underperformance": "Schedule a detailed plant inspection (strings, soiling, inverter logs).",
    "Temperature abnormality": "Check temperature sensors and panel operating conditions.",
    "Sensor fault / calibration": "Inspect sensor connections and recalibrate the sensor.",
    "Inverter fault": "Check inverter error codes, cooling and AC/DC terminations.",
    "Panel soiling / shading": "Clean the panels and remove or assess the shading source.",
    "Wiring / connections": "Inspect connectors, junction boxes and cable insulation.",
    "Routine inspection": "Carry out the scheduled visual and electrical inspection.",
    "Other": "",
}
PRI_W = {"Low": 1.5, "Medium": 4.0, "High": 8.0, "Urgent": 12.0}        # unresolved issue weight
STAT_F = {"Open": 1.0, "Follow-up Required": 0.8, "In Progress": 0.6}   # status factor
OVERDUE_W = {"Low": 1.5, "Medium": 3.0, "High": 6.0, "Urgent": 8.0}
BONUS_W = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Urgent": 4.0}      # completed + verified
EVENT_W = {"High": 2.0, "Medium": 1.0, "Low": 0.3}
STATUS_COLOR = {"Open": "#ef4444", "In Progress": "#3b82f6", "Completed": "#16a34a",
                "Follow-up Required": "#f59e0b"}


def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS maintenance_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, event_key TEXT, inverter TEXT,
        status TEXT, author TEXT, note TEXT)""")
    if "plant" not in [r[1] for r in con.execute("PRAGMA table_info(maintenance_log)")]:
        con.execute("ALTER TABLE maintenance_log ADD COLUMN plant TEXT")
    con.execute("""CREATE TABLE IF NOT EXISTS maint_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT, plant TEXT NOT NULL, inspection_date TEXT, asset_id TEXT,
        category TEXT, priority TEXT, description TEXT, action TEXT, technician TEXT, status TEXT,
        next_due TEXT, notes TEXT, acknowledged INTEGER DEFAULT 0, ack_by TEXT, ack_at TEXT,
        created_at TEXT, updated_at TEXT)""")
    con.commit()
    return con


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def add_record(rec):
    rec = dict(rec)
    rec["created_at"] = rec["updated_at"] = _now()
    if rec.get("acknowledged"):
        rec["ack_at"] = _now()
    cols = ",".join(rec)
    con = db()
    con.execute(f"INSERT INTO maint_records({cols}) VALUES({','.join('?' * len(rec))})", list(rec.values()))
    con.commit()
    con.close()


def update_record(rid, fields):
    fields = dict(fields)
    fields["updated_at"] = _now()
    sets = ",".join(f"{c}=?" for c in fields)
    con = db()
    con.execute(f"UPDATE maint_records SET {sets} WHERE id=?", list(fields.values()) + [int(rid)])
    con.commit()
    con.close()


def read_records():
    con = db()
    df = pd.read_sql_query("SELECT * FROM maint_records ORDER BY id DESC", con)
    con.close()
    for c in ["inspection_date", "next_due"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["acknowledged"] = df["acknowledged"].fillna(0).astype(int)
    return df


def add_log(event_key, plant, inverter, status, author, note):
    con = db()
    con.execute("INSERT INTO maintenance_log(created_at,event_key,plant,inverter,status,author,note) "
                "VALUES(?,?,?,?,?,?,?)", (_now(), event_key, plant, inverter, status, author, note))
    con.commit()
    con.close()


def read_log():
    con = db()
    out = pd.read_sql_query("SELECT * FROM maintenance_log ORDER BY id DESC", con)
    con.close()
    return out


def attach_status(ev):
    log = read_log()
    ev = ev.copy()
    if log.empty or ev.empty:
        ev["Status"] = "Open"
        return ev
    last = (log[log.event_key.notna() & (log.event_key != "")]
            .sort_values("id").groupby("event_key").tail(1).set_index("event_key")["status"])
    ev["Status"] = ev["Event ID"].map(last).fillna("Open")
    return ev


def flag_overdue(rec, ref):
    """Adds `overdue` and `days_overdue`. A record is overdue when its next-due date has passed and
    it is either still unresolved, or completed with no newer inspection of the same asset."""
    r = rec.copy()
    ref = pd.Timestamp(ref)
    if r.empty:
        r["overdue"], r["days_overdue"] = pd.Series(dtype=bool), pd.Series(dtype=int)
        return r
    latest = r.groupby(["plant", "asset_id"])["inspection_date"].transform("max")
    superseded = (r.status == "Completed") & (r.inspection_date < latest)
    r["overdue"] = (r.next_due < ref) & ~superseded
    days = (ref - r.next_due).dt.days
    r["days_overdue"] = days.where(r.overdue, 0).fillna(0).clip(lower=0).astype(int)
    return r


def maint_stats(r):
    total = len(r)
    done = int((r.status == "Completed").sum()) if total else 0
    unres = r[r.status != "Completed"] if total else r
    return {"total": total, "open": int((r.status == "Open").sum()) if total else 0, "completed": done,
            "overdue": int(r.overdue.sum()) if total else 0,
            "high_unres": int(unres.priority.isin(["High", "Urgent"]).sum()) if total else 0,
            "unresolved": len(unres), "completion": (done / total * 100) if total else 0.0}


def plant_indicators(d, events):
    day = d[d.daytime]
    n = max(len(day), 1)
    tdiff = d["mod"] - d.amb
    temp_bad = (((tdiff < -3) & (d.irr > 0.2)) | (tdiff > 45) | (d.amb < -20) | (d.amb > 65) |
                (d["mod"] < -20) | (d["mod"] > 110))
    under = events[events.Type.isin(["Under-performance", "Severe derating"])] if len(events) else events
    top = d[d.anomaly].inverter.value_counts().head(3)
    hs_ = health_score(d, 0.25)
    return {"anom_pct": day.anomaly.sum() / n * 100,
            "sens_pct": d.q_any.sum() / max(len(d), 1) * 100,
            "missing_pct": d.q_missing.sum() / max(len(d), 1) * 100,
            "impl_pct": d.q_impossible.sum() / max(len(d), 1) * 100,
            "stuck_pct": d.q_stuck.sum() / max(len(d), 1) * 100,
            "temp_pct": float(temp_bad[d.daytime].sum()) / n * 100,
            "loss_pct": hs_["loss_pct"], "n_events": len(events),
            "under_events": len(under),
            "under_days": int(pd.to_datetime(under.Start).dt.date.nunique()) if len(under) else 0,
            "top_inv": ", ".join(f"{i} ({c})" for i, c in top.items())}


def maintenance_score(rec, ind, events, ref):
    """Indicative maintenance-management score (0-100) built from the maintenance records of the plant
    and its monitoring indicators. Starts at 100, subtracts weighted penalties, adds a capped bonus."""
    r = flag_overdue(rec, ref)
    unres = r[r.status != "Completed"] if len(r) else r
    ov = r[r.overdue] if len(r) else r
    comp = r[r.status == "Completed"] if len(r) else r
    cnt = unres.priority.value_counts() if len(unres) else pd.Series(dtype=int)
    mix = ", ".join(f"{cnt[k]} {k}" for k in ["Urgent", "High", "Medium", "Low"] if k in cnt)

    p_unres = min(40.0, sum(PRI_W.get(p, 4) * STAT_F.get(s, 1) for p, s in zip(unres.priority, unres.status))) \
        if len(unres) else 0.0
    p_over = min(25.0, sum(OVERDUE_W.get(p, 3) * (1 + min(dd, 30) / 30) for p, dd in
                           zip(ov.priority, ov.days_overdue))) if len(ov) else 0.0
    p_anom = min(10.0, ind["anom_pct"] * 1.5)
    p_sens = min(5.0, ind["sens_pct"] * 2)
    p_loss = min(5.0, ind["loss_pct"] * 1.5)
    pending_ev = events[events.Status.isin(["Open", "In progress"])] if len(events) else events
    p_ev = min(10.0, sum(EVENT_W.get(sv, 0.5) * (1.0 if stt == "Open" else 0.4)
                         for sv, stt in zip(pending_ev.Severity, pending_ev.Status))) if len(pending_ev) else 0.0
    ver = comp[comp.acknowledged == 1] if len(comp) else comp
    unv = comp[comp.acknowledged != 1] if len(comp) else comp
    bonus = min(15.0, sum(BONUS_W.get(p, 2) for p in ver.priority) +
                sum(BONUS_W.get(p, 2) for p in unv.priority) / 3)
    score = float(np.clip(100 - p_unres - p_over - p_anom - p_sens - p_loss - p_ev + bonus, 0, 100))
    factors = [
        {"Factor": "Unresolved issues (status × priority)", "Points": -p_unres,
         "Why": f"{len(unres)} unresolved record(s)" + (f": {mix}" if mix else "")},
        {"Factor": "Overdue inspections / maintenance", "Points": -p_over,
         "Why": f"{len(ov)} overdue; larger deduction for high priority and days overdue"},
        {"Factor": "Power-anomaly frequency", "Points": -p_anom,
         "Why": f"{ind['anom_pct']:.2f}% of daytime readings flagged"},
        {"Factor": "Sensor-quality problems", "Points": -p_sens,
         "Why": f"{ind['sens_pct']:.2f}% of readings missing, implausible or stuck"},
        {"Factor": "Estimated performance loss", "Points": -p_loss,
         "Why": f"{ind['loss_pct']:.2f}% of expected energy lost"},
        {"Factor": "Unacknowledged detected events", "Points": -p_ev,
         "Why": f"{len(pending_ev)} event(s) still Open / In progress in the Detected-events log"},
        {"Factor": "Completed & verified tasks (bonus)", "Points": bonus,
         "Why": f"{len(ver)} verified, {len(unv)} completed but not yet acknowledged (counted at ⅓)"},
    ]
    return {"score": score, "factors": pd.DataFrame(factors), "flagged": r, "stats": maint_stats(r)}


def m_band(score):
    if score >= 90:
        return "Excellent", "#16a34a"
    if score >= 75:
        return "Good", "#2563eb"
    if score >= 50:
        return "Needs Attention", "#f59e0b"
    return "Critical", "#dc2626"


def score_card(title, score, sub=""):
    lab, col = m_band(score)
    return (f'<div class="card"><h3>{title}</h3>'
            f'<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">'
            f'<span style="font-size:2.8rem;font-weight:800;color:{col}">{score:.0f}</span>'
            f'<span style="color:#64748b">/ 100</span><span class="pill" style="background:{col}">{lab}</span></div>'
            f'<div class="pbar"><div class="pfill" style="width:{score:.0f}%;background:{col}"></div></div>'
            f'<div style="color:#64748b;font-size:.82rem">{sub}</div></div>')


def build_recommendations(ind, flagged):
    out = []
    if ind["anom_pct"] >= 3 or ind["n_events"] >= 5:
        out.append(("#dc2626" if ind["anom_pct"] >= 8 else "#f59e0b", "Frequent power anomalies",
                    f"{ind['anom_pct']:.1f}% of daytime readings flagged in {ind['n_events']} events"
                    + (f"; most affected: {ind['top_inv']}" if ind["top_inv"] else ""),
                    "Inspect panels, shading, connections and inverter performance."))
    if ind["temp_pct"] >= 0.5:
        out.append(("#f59e0b", "Abnormal temperature readings",
                    f"{ind['temp_pct']:.1f}% of daytime readings show implausible module/ambient temperatures",
                    "Check temperature sensors and panel operating conditions."))
    if ind["missing_pct"] + ind["impl_pct"] + ind["stuck_pct"] >= 0.5:
        out.append(("#f59e0b", "Missing or implausible sensor readings",
                    f"missing {ind['missing_pct']:.2f}% · implausible {ind['impl_pct']:.2f}% · stuck {ind['stuck_pct']:.2f}%",
                    "Inspect sensor connections and calibration."))
    if ind["under_events"] >= 3:
        out.append(("#dc2626", "Repeated underperformance",
                    f"{ind['under_events']} underperformance / derating events on {ind['under_days']} different day(s)",
                    "Schedule a detailed plant inspection."))
    if len(flagged) and flagged.overdue.any():
        for _, o in flagged[flagged.overdue].sort_values("days_overdue", ascending=False).head(6).iterrows():
            out.append(("#dc2626", f"Overdue maintenance: {o.asset_id} · {o.category}",
                        f"{o.priority} priority, was due {o.next_due:%d %b %Y} ({o.days_overdue} day(s) overdue), "
                        f"status {o.status}",
                        "Highlight this task and follow up with " + (o.technician or "the assigned technician") + "."))
    if not out:
        out.append(("#16a34a", "No automatic actions needed",
                    "Monitoring indicators and maintenance records are within limits", "Continue routine inspections."))
    return out


def fmt_dates(s):
    return s.dt.strftime("%d %b %Y").fillna("")


def history_view(r):
    out = pd.DataFrame({
        "ID": r.id, "Plant": r.plant, "Inspection date": fmt_dates(r.inspection_date), "Asset ID": r.asset_id,
        "Category": r.category, "Priority": r.priority, "Status": r.status, "Technician": r.technician,
        "Next due": fmt_dates(r.next_due),
        "Overdue": [f"Yes ({d_}d)" if o else "No" for o, d_ in zip(r.overdue, r.days_overdue)],
        "Acknowledged": [f"✔ {b or ''} {a or ''}".strip() if k else "—" for k, b, a in
                         zip(r.acknowledged, r.ack_by, r.ack_at)],
        "Description": r.description, "Recommended action": r.action, "Notes": r.notes})
    return out.reset_index(drop=True)


def style_history(df):
    pc = {"Urgent": "#fecaca", "High": "#fed7aa", "Medium": "#fef08a", "Low": "#dcfce7"}
    sc = {"Open": "#fecaca", "In Progress": "#bfdbfe", "Completed": "#bbf7d0", "Follow-up Required": "#fde68a"}
    sty = df.style
    mp = sty.map if hasattr(sty, "map") else sty.applymap
    sty = mp(lambda v: f"background-color:{pc.get(v, '')};color:#111", subset=["Priority"])
    sty = sty.map(lambda v: f"background-color:{sc.get(v, '')};color:#111", subset=["Status"]) \
        if hasattr(sty, "map") else sty.applymap(lambda v: f"background-color:{sc.get(v, '')};color:#111", subset=["Status"])
    fn = sty.map if hasattr(sty, "map") else sty.applymap
    return fn(lambda v: "background-color:#fecaca;color:#7f1d1d;font-weight:600" if str(v).startswith("Yes") else "",
              subset=["Overdue"])


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)


def fig_status(r):
    counts = [int((r.status == s).sum()) for s in MSTATUS]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    bars = ax.bar(MSTATUS, counts, color=[STATUS_COLOR[s] for s in MSTATUS])
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("Tasks")
    ax.set_title("Maintenance tasks by status", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _clean(ax)
    fig.tight_layout()
    return fig


def fig_category(r):
    vc = r.category.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    bars = ax.barh(vc.index, vc.values, color="#6366f1")
    ax.bar_label(bars, padding=2)
    ax.set_xlabel("Issues")
    ax.set_title("Maintenance issues by category", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_factors(fac):
    f = fac[fac.Points.abs() > 0.01].iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    if f.empty:
        ax.text(.5, .5, "No deductions", ha="center", va="center")
        ax.axis("off")
    else:
        bars = ax.barh(f.Factor.str.replace(r" \(.*\)", "", regex=True), f.Points,
                       ["#16a34a" if p > 0 else "#dc2626" for p in f.Points])
        ax.bar_label(bars, fmt="%+.1f", padding=2)
        ax.axvline(0, color="#475569", lw=.8)
    ax.set_title("Points added / removed", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
raw = load_data()
plant_ids = sorted(raw.plant.unique())
PNAME = {pid: f"Plant {i + 1}" for i, pid in enumerate(plant_ids)}
PTAG = {pid: f"P{i + 1}" for i, pid in enumerate(plant_ids)}
NAME2PID = {v: k for k, v in PNAME.items()}

with st.sidebar:
    st.markdown("## ☀️ Control panel")
    sel_pid = (st.selectbox("Plant", plant_ids, format_func=lambda p: PNAME[p])
               if len(plant_ids) > 1 else plant_ids[0])
    rp = raw[raw.plant == sel_pid]
    dmin, dmax = rp.ts.min().date(), rp.ts.max().date()
    rng_sel = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(rng_sel, (tuple, list)) and len(rng_sel) == 2:
        d_from, d_to = rng_sel
    else:
        d_from, d_to = dmin, dmax
    st.markdown("### Detection")
    mode = st.radio("Method", ["Combined", "Residual only", "Isolation Forest only"],
                    help="Residual = measured vs expected power. Isolation Forest = unusual combinations of "
                         "readings. Combined uses both.")
    k = st.slider("Residual sensitivity (z-score)", 2.0, 6.0, 3.5, 0.5,
                  help="How many 'standard deviations' away from expected power a reading must be to count as "
                       "unusual. Higher = fewer alarms.")
    min_dev = st.slider("Minimum deviation (%)", 5, 40, 12, 1,
                        help="A reading must also differ from expected power by at least this percentage.") / 100
    contamination = st.slider("Isolation Forest contamination", 0.005, 0.05, 0.01, 0.005,
                              help="Share of readings the Isolation Forest is allowed to flag as unusual "
                                   "(0.01 = 1%).")
    min_len = st.slider("Minimum event length (readings)", 1, 6, 2,
                        help="Ignore blips shorter than this many consecutive readings (each reading is 15 min).")
    st.markdown("### Business & maintenance")
    tariff = st.number_input("Tariff (₹ per kWh)", 1.0, 30.0, 6.0, 0.5)
    ref_date = st.date_input("Maintenance as-of date", value=dt.date.today(),
                             help="Used to decide which maintenance tasks are overdue.")
    if isinstance(ref_date, (tuple, list)):
        ref_date = ref_date[0] if ref_date else dt.date.today()


def run_plant(pid, d_from, d_to, k, min_dev, contamination, mode, min_len, tariff):
    """Full monitoring pipeline for one plant (cached building blocks)."""
    rp_ = raw[raw.plant == pid]
    sub = rp_[(rp_.ts.dt.date >= d_from) & (rp_.ts.dt.date <= d_to)]
    if sub.empty:
        sub = rp_
    qc_ = add_quality(sub.reset_index(drop=True))
    fit_ = fit_models(qc_)
    d_ = detect(qc_, fit_["model"], k, min_dev, contamination, mode, min_len)
    sm = int(d_.ts.drop_duplicates().sort_values().diff().dt.total_seconds().median() / 60) or 15
    ev_ = attach_status(build_events(d_, sm, tariff, PTAG[pid]))
    hs_ = health_score(d_, sm / 60)
    return {"d": d_, "fit": fit_, "events": ev_, "hs": hs_, "ind": plant_indicators(d_, ev_),
            "step_min": sm, "step_h": sm / 60}


R = run_plant(sel_pid, d_from, d_to, k, min_dev, contamination, mode, min_len, tariff)
d, fit, events, hs, ind = R["d"], R["fit"], R["events"], R["hs"], R["ind"]
step_min, step_h = R["step_min"], R["step_h"]
label, color = band(hs["score"])
sel_label = PNAME[sel_pid]

rec_all = read_records()
rec_sel = rec_all[rec_all.plant == sel_label]
ms = maintenance_score(rec_sel, ind, events, ref_date)
m_label, m_color = m_band(ms["score"])

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.markdown(f"""<div class="hero"><h1>☀️ Solar PV Performance Monitor</h1>
<p>Expected-power modelling · anomaly detection · sensor quality · loss estimation &nbsp;|&nbsp;
{d.ts.min():%d %b %Y} → {d.ts.max():%d %b %Y} &nbsp;|&nbsp; {d.inverter.nunique()} inverters</p></div>""",
            unsafe_allow_html=True)

valid = d[d.pred.notna() & d.ac.notna()]
actual_mwh = valid.ac.sum() * step_h / 1000
expected_mwh = valid.pred.sum() * step_h / 1000
lost_kwh = events["Lost energy (kWh)"].sum()
lost_rs = events["Lost revenue (₹)"].sum()
c = st.columns(6)
c[0].markdown(kpi("Plant health", f"{hs['score']:.0f} / 100", label, color), unsafe_allow_html=True)
c[1].markdown(kpi("Energy generated", f"{actual_mwh:,.1f} MWh", f"Expected {expected_mwh:,.1f} MWh", "#3b82f6"),
              unsafe_allow_html=True)
c[2].markdown(kpi("Estimated energy loss", f"{lost_kwh:,.0f} kWh", f"{hs['loss_pct']:.2f}% of expected", "#f97316"),
              unsafe_allow_html=True)
c[3].markdown(kpi("Revenue loss", f"₹{lost_rs:,.0f}", f"@ ₹{tariff:g}/kWh", "#ef4444"), unsafe_allow_html=True)
c[4].markdown(kpi("Anomaly events", f"{len(events)}", f"{(events.Status == 'Open').sum()} open · "
                  f"{int(d.q_any.sum())} sensor issues", "#8b5cf6"), unsafe_allow_html=True)
c[5].markdown(kpi("Maintenance score", f"{ms['score']:.0f} / 100", f"{m_label} · {sel_label}", m_color),
              unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["🏠 Overview", "🔎 Anomalies", "📋 Events", "🛰️ Sensor quality", "🤖 Models",
                "💡 Explain & What-if", "🛠️ Maintenance Centre", "📥 Reports"])

# ---------------------------- Overview + health -----------------------------
with tabs[0]:
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="card"><h3>Panel health</h3>', unsafe_allow_html=True)
        st.markdown(panel_svg(hs["score"]), unsafe_allow_html=True)
        if ASSET_PHOTO.exists():
            st.image(str(ASSET_PHOTO), caption="Installed array")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=hs["score"], number={"suffix": " / 100", "font": {"size": 34}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.3},
                   "steps": [{"range": [0, 50], "color": "#fee2e2"}, {"range": [50, 70], "color": "#fef3c7"},
                             {"range": [70, 85], "color": "#ecfccb"}, {"range": [85, 100], "color": "#dcfce7"}]}))
        gauge.update_layout(height=250)
        st.markdown(f'<div class="card"><h3>Plant health score &nbsp;<span class="pill" '
                    f'style="background:{color}">{label}</span></h3>', unsafe_allow_html=True)
        show_plot(gauge)

        def chk(name, val, unit, good, warn):
            col = "#16a34a" if val <= good else ("#f59e0b" if val <= warn else "#dc2626")
            txt = "Good" if val <= good else ("Watch" if val <= warn else "High")
            return (f'<div class="chk"><span><span class="dot" style="background:{col}"></span>{name}</span>'
                    f'<span><b>{val:.2f}{unit}</b> &nbsp;<span style="color:{col}">{txt}</span></span></div>')
        st.markdown(
            chk("Anomalous daytime readings", hs["anom_pct"], "%", 2, 6) +
            chk("Estimated performance loss", hs["loss_pct"], "%", 1, 3) +
            chk("Questionable sensor readings", hs["sens_pct"], "%", 0.5, 2) +
            f'<div class="chk"><span>Score deductions</span><span>loss −{hs["p_loss"]:.1f} · '
            f'frequency −{hs["p_anom"]:.1f} · sensors −{hs["p_sens"]:.1f}</span></div></div>',
            unsafe_allow_html=True)

    st.markdown("#### Health by inverter")
    inv_rows = []
    for inv, g in d.groupby("inverter"):
        h = health_score(g, step_h)
        inv_rows.append({"Inverter": inv, "Health score": round(h["score"], 1), "Condition": band(h["score"])[0],
                         "Anomalous readings (%)": round(h["anom_pct"], 2),
                         "Loss (%)": round(h["loss_pct"], 2), "Lost energy (kWh)": round(h["lost_kwh"], 1)})
    inv_df = pd.DataFrame(inv_rows)
    show_df(inv_df, column_config={"Health score": st.column_config.ProgressColumn(
        "Health score", min_value=0, max_value=100, format="%.0f")})

    a, b = st.columns(2)
    daily = valid.assign(date=valid.ts.dt.date).groupby("date")[["ac", "pred"]].sum() * step_h / 1000
    f1 = go.Figure()
    f1.add_bar(x=daily.index, y=daily.ac, name="Actual (MWh)", marker_color="#3b82f6")
    f1.add_scatter(x=daily.index, y=daily.pred, name="Expected (MWh)", mode="lines+markers",
                   line=dict(color="#f59e0b", width=3))
    f1.update_layout(title="Daily energy: actual vs expected", height=340)
    with a:
        show_plot(f1)
    f2 = px.bar(inv_df, x="Inverter", y="Lost energy (kWh)", color="Health score",
                color_continuous_scale=["#dc2626", "#f59e0b", "#16a34a"], range_color=[0, 100],
                title="Estimated energy loss by inverter")
    f2.update_layout(height=340)
    with b:
        show_plot(f2)

# ------------------------------- Anomalies ---------------------------------
with tabs[1]:
    c1, c2 = st.columns([1, 2])
    inv_sel = c1.selectbox("Inverter", sorted(d.inverter.unique()))
    days = sorted(d.ts.dt.date.unique())
    win = c2.select_slider("Date window", options=days, value=(days[0], days[-1]))
    g = d[(d.inverter == inv_sel) & (d.ts.dt.date >= win[0]) & (d.ts.dt.date <= win[1])]
    fig = go.Figure()
    fig.add_scatter(x=g.ts, y=g.ac, name="Measured AC", line=dict(color="#2563eb", width=1.6))
    fig.add_scatter(x=g.ts, y=g.pred, name="Expected AC", line=dict(color="#f59e0b", width=1.6, dash="dot"))
    an = g[g.status == "Anomaly"]
    se = g[g.status == "Sensor issue"]
    fig.add_scatter(x=an.ts, y=an.ac, mode="markers", name="Anomaly",
                    marker=dict(color="#dc2626", size=7, symbol="x"))
    fig.add_scatter(x=se.ts, y=se.ac, mode="markers", name="Sensor issue",
                    marker=dict(color="#8b5cf6", size=7, symbol="diamond"))
    fig.update_layout(title=f"{inv_sel}: measured vs expected AC power (kW)", height=420)
    show_plot(fig)
    fr = go.Figure()
    fr.add_scatter(x=g.ts, y=g.res_pct * 100, line=dict(color="#64748b", width=1), name="Residual %")
    fr.add_hline(y=-min_dev * 100, line_dash="dash", line_color="#dc2626")
    fr.add_hline(y=min_dev * 100, line_dash="dash", line_color="#dc2626")
    fr.update_layout(title="Residual (% of expected)", height=260)
    show_plot(fr)
    m = st.columns(4)
    m[0].metric("Residual flags", int(d.flag_res.sum()))
    m[1].metric("Isolation Forest flags", int(d.flag_iso.sum()))
    m[2].metric("Final anomalies", int(d.anomaly.sum()))
    m[3].metric("Both methods agree", int((d.flag_res & d.flag_iso).sum()))

# --------------------------------- Events ----------------------------------
with tabs[2]:
    if events.empty:
        st.success("No anomaly events detected with the current settings.")
    else:
        e1, e2 = st.columns([2, 1])
        show_df(events)
        tl = events.copy()
        tl["End"] = pd.to_datetime(tl["End"])
        ft = px.timeline(tl, x_start="Start", x_end="End", y="Inverter", color="Severity",
                         color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#3b82f6"},
                         hover_data=["Type", "Lost energy (kWh)"], title="Event timeline")
        ft.update_layout(height=300)
        show_plot(ft)
        t = events.groupby("Type")[["Lost energy (kWh)", "Lost revenue (₹)"]].sum().reset_index()
        show_plot(px.bar(t, x="Type", y="Lost revenue (₹)", color="Type", title="Revenue loss by event type"))

# ------------------------------ Sensor quality -----------------------------
with tabs[3]:
    sq = d.groupby("inverter").agg(Missing=("q_missing", "sum"), Impossible=("q_impossible", "sum"),
                                   Stuck=("q_stuck", "sum")).reset_index()
    c1, c2 = st.columns([1, 1])
    with c1:
        show_plot(px.bar(sq.melt("inverter", var_name="Issue", value_name="Readings"), x="inverter",
                         y="Readings", color="Issue", title="Questionable readings by inverter",
                         color_discrete_map={"Missing": "#f59e0b", "Impossible": "#dc2626", "Stuck": "#8b5cf6"}))
    with c2:
        st.markdown("""<div class="card"><h3>How readings are checked</h3>
        <div class="chk"><span><b>Missing</b></span><span>blank power, irradiation or temperature</span></div>
        <div class="chk"><span><b>Impossible</b></span><span>negative power, power far above rating, out-of-range sensors</span></div>
        <div class="chk"><span><b>Stuck</b></span><span>identical non-zero value for 6+ consecutive readings</span></div>
        <div class="chk"><span><b>Handling</b></span><span>flagged rows are excluded from performance anomalies</span></div></div>""",
                    unsafe_allow_html=True)
    bad = d[d.q_any][["ts", "inverter", "q_reason", "ac", "irr", "amb", "mod"]].rename(
        columns={"ts": "Time", "inverter": "Inverter", "q_reason": "Issue", "ac": "AC (kW)",
                 "irr": "Irradiation", "amb": "Ambient °C", "mod": "Module °C"})
    show_df(bad.head(500))

# --------------------------------- Models ----------------------------------
with tabs[4]:
    mt = fit["metrics"]
    st.markdown(f"**Best model: {fit['best']}** (lowest RMSE) · trained on {fit['train_rows']:,} rows, "
                f"tested on the latest {fit['test_rows']:,} rows")
    show_df(mt.style.format({"MAE (kW)": "{:.2f}", "RMSE (kW)": "{:.2f}", "R²": "{:.4f}", "MAPE (%)": "{:.2f}"})
            .highlight_min(subset=["MAE (kW)", "RMSE (kW)", "MAPE (%)"], color="#dcfce7")
            .highlight_max(subset=["R²"], color="#dcfce7"))
    m1, m2 = st.columns(2)
    with m1:
        show_plot(px.bar(mt, x="Model", y="RMSE (kW)", color="Model", title="RMSE by model (lower is better)"))
    with m2:
        tt = fit["test"].sample(min(len(fit["test"]), 1500), random_state=1)
        f = px.scatter(tt, x="actual", y="pred", opacity=.5, title="Test set: actual vs predicted (kW)")
        mx = float(max(tt.actual.max(), tt.pred.max()))
        f.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#dc2626", dash="dash"), name="ideal")
        show_plot(f)

# --------------------------- Explain & what-if -----------------------------
with tabs[5]:
    x1, x2 = st.columns(2)
    with x1:
        imp = fit["importance"]
        show_plot(px.bar(imp, x="Importance", y="Feature", orientation="h",
                         title="Feature importance (permutation, MAE increase)"))
    with x2:
        ev_rows = d[d.status == "Anomaly"]
        nm_rows = d[d.status == "Normal"]
        if len(ev_rows):
            cmp_ = pd.DataFrame({"Measure": ["Irradiation", "Ambient °C", "Module °C", "Hour", "Measured AC (kW)",
                                             "Expected AC (kW)"],
                                 "Normal": [nm_rows[v].mean() for v in ["irr", "amb", "mod", "hour", "ac", "pred"]],
                                 "During anomalies": [ev_rows[v].mean() for v in
                                                      ["irr", "amb", "mod", "hour", "ac", "pred"]]})
            cmp_["Difference (%)"] = (cmp_["During anomalies"] / cmp_["Normal"] - 1) * 100
            st.markdown("**Events vs normal operation**")
            show_df(cmp_.style.format({"Normal": "{:.2f}", "During anomalies": "{:.2f}", "Difference (%)": "{:+.1f}"}))
        else:
            st.info("No anomalies to compare.")

    if st.checkbox("Compute SHAP explanations (optional, needs `pip install shap`)"):
        try:
            import shap
            import matplotlib.pyplot as plt
            samp = d[d.pred.notna()].sample(min(300, d.pred.notna().sum()), random_state=0)[FEATURES]
            bg = samp.iloc[:100]
            expl = shap.Explainer(fit["model"].predict, bg)
            sv = expl(samp)
            plt.figure()
            shap.plots.beeswarm(sv, show=False)
            st.pyplot(plt.gcf())
        except ImportError:
            st.caption("SHAP is not installed. Run `pip install shap` and reload.")
        except Exception as e:   # pragma: no cover
            st.caption(f"SHAP could not be computed: {e}")

    st.markdown("### 🎛️ What-if simulator")
    w1, w2 = st.columns([1, 1.4])
    with w1:
        wi = st.slider("Irradiation (kW/m²)", 0.0, 1.3, 0.8, 0.05)
        wa = st.slider("Ambient temperature (°C)", 10.0, 50.0, 30.0, 0.5)
        wm = st.slider("Module temperature (°C)", 10.0, 85.0, float(min(85, 30 + 24 * 0.8)), 0.5)
        wh = st.slider("Time of day (hour)", 5.0, 19.0, 12.0, 0.25)
        meas = st.number_input("Measured AC power to compare (kW, optional)", 0.0, 5000.0, 0.0, 10.0)
    exp_kw = float(max(fit["model"].predict(pd.DataFrame([[wi, wa, wm, wh]], columns=FEATURES))[0], 0))
    with w2:
        s1, s2 = st.columns(2)
        s1.metric("Expected AC power", f"{exp_kw:,.0f} kW")
        if meas > 0:
            dev = (meas - exp_kw) / max(exp_kw, 1) * 100
            s2.metric("Measured vs expected", f"{dev:+.1f}%", delta_color="normal")
        grid = np.linspace(0, 1.3, 40)
        curve = fit["model"].predict(pd.DataFrame({"irr": grid, "amb": wa, "mod": wm, "hour": wh}))
        fw = go.Figure()
        fw.add_scatter(x=grid, y=np.clip(curve, 0, None), name="Expected curve", line=dict(color="#f59e0b", width=3))
        fw.add_scatter(x=[wi], y=[exp_kw], mode="markers", name="Selected", marker=dict(size=13, color="#2563eb"))
        fw.update_layout(title="Expected power vs irradiation", xaxis_title="Irradiation (kW/m²)",
                         yaxis_title="AC power (kW)", height=320)
        show_plot(fw)

# ----------------------------- Maintenance Centre ---------------------------
with tabs[6]:
    st.markdown(f"### 🛠️ Maintenance Centre · {sel_label}")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    mt = st.tabs(["📊 Maintenance score", "📝 Records (add / update)", "📈 Analytics & history",
                  "🔔 Detected events"])

    # ---------------------------- Maintenance score ----------------------------
    with mt[0]:
        st.caption("Indicative maintenance-management score. It is recalculated automatically from the saved "
                   "maintenance records and this plant's monitoring indicators whenever a record is added or its "
                   "status changes. It is not a certified engineering or safety rating.")
        s1, s2 = st.columns([1, 1.15])
        with s1:
            st.markdown(score_card(f"Maintenance score · {sel_label}", ms["score"],
                                   "90–100 Excellent · 75–89 Good · 50–74 Needs Attention · 0–49 Critical"),
                        unsafe_allow_html=True)
            top = ms["factors"].reindex(ms["factors"].Points.abs().sort_values(ascending=False).index)
            top = top[top.Points.abs() > 0.05].head(3)
            if len(top):
                st.markdown("**Main factors right now**")
                for _, f in top.iterrows():
                    sign = "+" if f.Points > 0 else "−"
                    colr = "#16a34a" if f.Points > 0 else "#dc2626"
                    st.markdown(f'<div class="chk"><span>{f.Factor}<br><small style="color:#64748b">{f.Why}</small>'
                                f'</span><b style="color:{colr}">{sign}{abs(f.Points):.1f}</b></div>',
                                unsafe_allow_html=True)
            else:
                st.markdown("No deductions: no unresolved issues, overdue tasks or monitoring concerns.")
        with s2:
            fg = fig_factors(ms["factors"])
            st.pyplot(fg)
            plt.close(fg)
        with st.expander("How the score is calculated"):
            st.markdown(
                "Start at **100**, subtract penalties, add a capped bonus, then limit to 0–100.\n\n"
                "- **Unresolved issues** (max −40): Urgent 12 · High 8 · Medium 4 · Low 1.5 per record, × status "
                "factor (Open 1.0, Follow-up Required 0.8, In Progress 0.6).\n"
                "- **Overdue inspections** (max −25): Urgent 8 · High 6 · Medium 3 · Low 1.5, increased by up to "
                "2× for tasks up to 30 days overdue.\n"
                "- **Power-anomaly frequency** (max −10), **sensor-quality problems** (max −5) and "
                "**estimated performance loss** (max −5) from the monitoring results.\n"
                "- **Unacknowledged detected events** (max −10): High 2 · Medium 1 · Low 0.3 per Open event.\n"
                "- **Bonus** (max +15): tasks that are **Completed and acknowledged (verified)** add Urgent 4 · "
                "High 3 · Medium 2 · Low 1; completed but unverified tasks count one third.\n\n"
                "Plant 1 and Plant 2 are scored separately from their own records and monitoring data.")
        show_df(ms["factors"].assign(Points=ms["factors"].Points.round(1)))

        if len(plant_ids) > 1:
            st.markdown("#### Plant comparison")
            cols = st.columns(len(plant_ids))
            for col, pid in zip(cols, plant_ids):
                if pid == sel_pid:
                    ind_p, ev_p = ind, events
                else:
                    with st.spinner(f"Analysing {PNAME[pid]}..."):
                        Rp = run_plant(pid, d_from, d_to, k, min_dev, contamination, mode, min_len, tariff)
                    ind_p, ev_p = Rp["ind"], Rp["events"]
                sc_p = maintenance_score(rec_all[rec_all.plant == PNAME[pid]], ind_p, ev_p, ref_date)
                n_rec = int((rec_all.plant == PNAME[pid]).sum())
                col.markdown(score_card(PNAME[pid], sc_p["score"],
                                        f"{n_rec} maintenance record(s) · {ind_p['n_events']} detected events"),
                             unsafe_allow_html=True)

        st.markdown("#### Automatic maintenance recommendations")
        for colr, title, evidence, action in build_recommendations(ind, ms["flagged"]):
            st.markdown(f'<div class="rec" style="--c:{colr}"><b>{title}</b><br><small>{evidence}</small><br>'
                        f'→ {action}</div>', unsafe_allow_html=True)

    # ------------------------------- Records ------------------------------------
    with mt[1]:
        st.markdown("#### Add a maintenance record")
        a1, a2 = st.columns(2)
        plant_choice = a1.selectbox("Plant", list(PNAME.values()),
                                    index=list(PNAME.values()).index(sel_label), key="new_plant")
        cat_choice = a2.selectbox("Issue category", CATEGORIES, key="new_cat")
        inv_opts = sorted(raw[raw.plant == NAME2PID[plant_choice]].inverter.unique())
        sug = SUGGESTED_ACTION.get(cat_choice, "")
        if sug:
            st.caption(f"Suggested action for this category: {sug}")
        with st.form("add_record", clear_on_submit=True):
            f1, f2, f3 = st.columns(3)
            insp = f1.date_input("Inspection date", value=dt.date.today())
            asset_sel = f2.selectbox("Inverter / asset", inv_opts)
            asset_txt = f3.text_input("Or solar panel / other asset ID", placeholder="e.g. STR-12-PNL-07")
            g1, g2, g3 = st.columns(3)
            prio = g1.selectbox("Priority", PRIORITIES, index=1)
            tech = g2.text_input("Assigned technician")
            stat = g3.selectbox("Status", MSTATUS)
            desc = st.text_area("Description of the issue")
            act = st.text_area("Recommended maintenance action", value=sug)
            h1, h2 = st.columns(2)
            due = h1.date_input("Next maintenance due date", value=dt.date.today() + dt.timedelta(days=30))
            ack_by = h2.text_input("Acknowledged by (name)")
            notes = st.text_area("Maintenance notes")
            ack = st.checkbox("Acknowledge / verify this record")
            go_add = st.form_submit_button("💾 Save record")
        if go_add:
            if not desc.strip():
                st.error("Please describe the issue before saving.")
            elif due < insp:
                st.error("The next due date cannot be earlier than the inspection date.")
            else:
                add_record({"plant": plant_choice, "inspection_date": str(insp),
                            "asset_id": asset_txt.strip() or asset_sel, "category": cat_choice, "priority": prio,
                            "description": desc.strip(), "action": act.strip(), "technician": tech.strip(),
                            "status": stat, "next_due": str(due), "notes": notes.strip(),
                            "acknowledged": int(ack), "ack_by": ack_by.strip() if ack else ""})
                st.session_state["flash"] = f"Record saved for {plant_choice}. The maintenance score has been updated."
                st.rerun()

        st.markdown("---")
        st.markdown("#### Update, acknowledge or close a record")
        if rec_all.empty:
            st.caption("No records yet. Add the first one above.")
        else:
            labels = {int(r_.id): f"#{r_.id} · {r_.plant} · {r_.asset_id} · {r_.category} · {r_.status}"
                      for r_ in rec_all.itertuples()}
            rid = st.selectbox("Record", list(labels), format_func=lambda i: labels[i], key="upd_pick")
            row = rec_all[rec_all.id == rid].iloc[0]
            with st.form(f"upd_{rid}"):
                u1, u2, u3 = st.columns(3)
                n_stat = u1.selectbox("Status", MSTATUS, index=MSTATUS.index(row.status)
                                      if row.status in MSTATUS else 0)
                n_prio = u2.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(row.priority)
                                      if row.priority in PRIORITIES else 1)
                n_tech = u3.text_input("Assigned technician", value=row.technician or "")
                u4, u5 = st.columns(2)
                n_due = u4.date_input("Next maintenance due date",
                                      value=row.next_due.date() if pd.notna(row.next_due)
                                      else dt.date.today() + dt.timedelta(days=30))
                n_by = u5.text_input("Acknowledged by (name)", value=row.ack_by or "")
                n_act = st.text_area("Recommended maintenance action", value=row.action or "")
                n_notes = st.text_area("Maintenance notes", value=row.notes or "")
                n_ack = st.checkbox("Acknowledged / verified", value=bool(row.acknowledged))
                go_upd = st.form_submit_button("✅ Update record")
            if go_upd:
                fields = {"status": n_stat, "priority": n_prio, "technician": n_tech.strip(),
                          "next_due": str(n_due), "action": n_act.strip(), "notes": n_notes.strip(),
                          "acknowledged": int(n_ack), "ack_by": n_by.strip() if n_ack else ""}
                if n_ack and not row.acknowledged:
                    fields["ack_at"] = _now()
                if not n_ack:
                    fields["ack_at"] = ""
                update_record(rid, fields)
                st.session_state["flash"] = f"Record #{rid} updated. The maintenance score has been recalculated."
                st.rerun()

    # --------------------------- Analytics & history ---------------------------
    with mt[2]:
        stt = ms["stats"]
        k1 = st.columns(6)
        k1[0].markdown(kpi("Total records", stt["total"], sel_label, "#3b82f6"), unsafe_allow_html=True)
        k1[1].markdown(kpi("Open issues", stt["open"], f"{stt['unresolved']} unresolved in total", "#ef4444"),
                       unsafe_allow_html=True)
        k1[2].markdown(kpi("Completed tasks", stt["completed"], "status Completed", "#16a34a"),
                       unsafe_allow_html=True)
        k1[3].markdown(kpi("Overdue inspections", stt["overdue"], f"as of {pd.Timestamp(ref_date):%d %b %Y}",
                           "#f59e0b"), unsafe_allow_html=True)
        k1[4].markdown(kpi("High-priority unresolved", stt["high_unres"], "High + Urgent", "#dc2626"),
                       unsafe_allow_html=True)
        k1[5].markdown(kpi("Completion", f"{stt['completion']:.0f}%", "completed ÷ total", "#8b5cf6"),
                       unsafe_allow_html=True)
        st.write("")
        fr = ms["flagged"]
        if fr.empty:
            st.caption("No maintenance records for this plant yet. Add one in the Records tab to see charts.")
        else:
            g1, g2 = st.columns(2)
            with g1:
                f_ = fig_status(fr)
                st.pyplot(f_)
                plt.close(f_)
            with g2:
                f_ = fig_category(fr)
                st.pyplot(f_)
                plt.close(f_)

        st.markdown("#### Maintenance history")
        flt = st.columns(3)
        pl_f = flt[0].multiselect("Plant", list(PNAME.values()), default=[sel_label])
        pr_f = flt[1].multiselect("Priority", PRIORITIES, default=PRIORITIES)
        st_f = flt[2].multiselect("Status", MSTATUS, default=MSTATUS)
        hist = flag_overdue(rec_all, ref_date)
        if len(hist):
            hist = hist[hist.plant.isin(pl_f) & hist.priority.isin(pr_f) & hist.status.isin(st_f)]
        hv = history_view(hist) if len(hist) else pd.DataFrame(columns=["ID", "Plant", "Inspection date", "Asset ID",
                                                                        "Category", "Priority", "Status",
                                                                        "Technician", "Next due", "Overdue",
                                                                        "Acknowledged", "Description",
                                                                        "Recommended action", "Notes"])
        show_df(style_history(hv) if len(hv) else hv)
        st.download_button("⬇ Export maintenance history (CSV)", hv.to_csv(index=False).encode("utf-8"),
                           file_name="maintenance_history.csv", mime="text/csv", key="dl_hist")

    # ----------------------- Detected events acknowledgement -------------------
    with mt[3]:
        l1, l2 = st.columns([1, 1.4])
        with l1:
            st.markdown("#### Acknowledge a detected event")
            with st.form("log_form", clear_on_submit=True):
                ev_pick = st.selectbox("Event", ["(general note)"] + events["Event ID"].tolist())
                ev_stat = st.selectbox("Status", STATUS_OPTIONS, index=1)
                who = st.text_input("Your name")
                note = st.text_area("Note", placeholder="e.g. Inverter restarted, string fuse replaced")
                ok = st.form_submit_button("Save to log")
            if ok:
                if ev_pick == "(general note)":
                    add_log("", sel_label, "", ev_stat, who or "operator", note)
                else:
                    inv_name = events.set_index("Event ID").loc[ev_pick, "Inverter"]
                    add_log(ev_pick, sel_label, inv_name, ev_stat, who or "operator", note)
                st.session_state["flash"] = "Event log saved. The maintenance score has been updated."
                st.rerun()
        with l2:
            st.markdown("#### Event log") 
            lg = read_log()
            if "plant" in lg.columns and len(lg):
                lg = lg[(lg.plant == sel_label) | lg.plant.isna()]
            if lg.empty:
                st.caption("No entries yet.")
            else:
                show_df(lg.rename(columns={"created_at": "Time", "event_key": "Event", "plant": "Plant",
                                           "inverter": "Inverter", "status": "Status", "author": "By",
                                           "note": "Note"}).drop(columns=["id"]))

# --------------------------------- Reports ---------------------------------
with tabs[7]:
    summary = pd.DataFrame([{"period_start": d.ts.min(), "period_end": d.ts.max(), "health_score": round(hs["score"], 1),
                             "condition": label, "events": len(events), "lost_kwh": round(lost_kwh, 1),
                             "lost_revenue_inr": round(lost_rs, 0), "anomalous_readings_pct": round(hs["anom_pct"], 2),
                             "sensor_issue_pct": round(hs["sens_pct"], 2), "best_model": fit["best"]}])
    full = d.drop(columns=["SOURCE_KEY"]).copy()
    items = [("Event report", events), ("Anomalous readings", d[d.status == "Anomaly"]),
             ("Sensor-quality issues", d[d.q_any]), ("Model comparison", fit["metrics"]),
             ("Summary", summary), ("Full results", full),
             ("Maintenance history", history_view(flag_overdue(rec_all, ref_date)) if len(rec_all) else rec_all)]
    cols = st.columns(3)
    for i, (name, frame) in enumerate(items):
        with cols[i % 3]:
            st.markdown(f'<div class="card"><h3>{name}</h3><span style="color:#64748b">{len(frame):,} rows</span></div>',
                        unsafe_allow_html=True)
            st.download_button(f"⬇ Download {name}.csv", frame.to_csv(index=False).encode("utf-8"),
                               file_name=f"{name.lower().replace(' ', '_')}.csv", mime="text/csv",
                               key=f"dl{i}")
