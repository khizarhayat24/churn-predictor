"""
Churn Intelligence — Executive Overview
A Streamlit dashboard that drives ChurnModelEngine (engine.py) end to end:
upload -> clean -> train 3 candidate models -> evaluate -> pick best by F1 ->
visualize -> export artifacts. Visual language (colors, type, glass cards,
KPI strip, sidebar nav) is carried over from the original HTML dashboard.
"""

import io
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import ChurnModelEngine

# ----------------------------------------------------------------------------
# DESIGN TOKENS
# ----------------------------------------------------------------------------
PRIMARY = "#3b82f6"
PRIMARY_CONTAINER = "#dbeafe"
ON_PRIMARY_CONTAINER = "#1e3a8a"
SECONDARY = "#10b981"
SECONDARY_DIM = "#059669"
SECONDARY_CONTAINER = "#d1fae5"
ON_SECONDARY_CONTAINER = "#064e3b"
TERTIARY = "#f43f5e"
TERTIARY_DIM = "#fb7185"
TERTIARY_CONTAINER = "#fecdd3"
ON_TERTIARY_CONTAINER = "#7b4147"
ERROR = "#ef4444"
ERROR_CONTAINER = "#fceaea"
BACKGROUND = "#f8fafc"
SURFACE = "#ffffff"
SURFACE_LOW = "#f8fafc"
SURFACE_HIGH = "#f1f5f9"
ON_SURFACE = "#0f172a"
ON_SURFACE_VARIANT = "#475569"
OUTLINE = "#94a3b8"
OUTLINE_VARIANT = "#e2e8f0"

MODEL_COLORS = {
    "Logistic Regression": PRIMARY,
    "Random Forest": SECONDARY,
    "XGBoost": TERTIARY,
}

REQUIRED_NUMERIC = ['Age', 'Monthly Spending', 'Tenure', 'Number of Purchases',
                     'Customer Support Requests', 'Login Frequency', 'Satisfaction Score']
REQUIRED_CATEGORICAL = ['Gender', 'City', 'Subscription Type']
REQUIRED_TARGET = 'Churn Status'

NAV_ITEMS = [
    ("overview",    "dashboard",    "Overview"),
    ("eda",         "bar_chart",    "EDA"),
    ("features",    "psychology",   "Feature Signals"),
    ("models",      "groups",       "Model Comparison"),
    ("segments",    "donut_small",  "Segmentation"),
    ("risk",        "warning",      "Risk Segments"),
    ("predict",     "person_search","Predict Customer"),
    ("reports",     "assessment",   "Reports & Export"),
]

PAGE_TITLES = {
    "overview": ("Churn Intelligence", "Executive overview of churn-prediction performance"),
    "eda":      ("Exploratory Data Analysis", "Demographics, spending patterns, churn distribution & correlations"),
    "features": ("Feature Signals", "What's actually driving churn risk, per the optimal model"),
    "models":   ("Model Comparison", "Head-to-head evaluation across all candidate models"),
    "segments": ("Customer Segmentation", "KMeans clusters — who your customers are and which groups churn most"),
    "risk":     ("Risk Segments", "Test-set customers ranked by predicted churn probability"),
    "predict":  ("Predict Customer", "Enter individual customer details to get a live churn probability"),
    "reports":  ("Reports & Export", "Download trained artifacts and a shareable summary"),
}

# ----------------------------------------------------------------------------
# PAGE CONFIG + CSS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence | Executive Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<style>
    html, body, .stApp {{
        background-color: {BACKGROUND} !important;
        font-family: 'Inter', sans-serif;
        color: {ON_SURFACE};
    }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stToolbar"] {{ display: none; }}

    .material-symbols-outlined {{
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }}

    /* ---- glass card ---- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(6px);
        border: 1px solid {OUTLINE_VARIANT} !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
    }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 4px 6px; }}

    /* ---- sidebar — always visible, collapse button hidden ---- */
    [data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {OUTLINE_VARIANT};
        min-width: 240px !important;
        max-width: 240px !important;
        transform: none !important;
        visibility: visible !important;
        display: block !important;
    }}
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] + div > button {{
        display: none !important;
    }}
    [data-testid="stSidebar"] .stButton button {{
        background-color: transparent;
        color: {ON_SURFACE_VARIANT};
        border: none;
        text-align: left;
        justify-content: flex-start;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        letter-spacing: 0.04em;
        font-weight: 600;
        text-transform: uppercase;
        padding: 10px 14px;
        border-radius: 8px;
        box-shadow: none;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background-color: {SURFACE_HIGH};
        color: {ON_SURFACE};
        border: none;
    }}
    [data-testid="stSidebar"] .stButton button[kind="primary"] {{
        background-color: {PRIMARY} !important;
        color: white !important;
        text-transform: uppercase;
        font-weight: 700;
        margin-top: 12px;
    }}
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{
        filter: brightness(1.1);
    }}

    .nav-active {{
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; border-radius: 8px;
        background-color: {PRIMARY_CONTAINER};
        color: {PRIMARY}; font-weight: 700;
        font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase;
        margin-bottom: 2px;
    }}

    .brand-title {{
        font-family: 'Hanken Grotesk', sans-serif; font-weight: 800; font-size: 21px;
        color: {ON_SURFACE}; margin-bottom: 0px;
    }}
    .brand-sub {{
        font-family: 'Inter', sans-serif; font-size: 12px; color: {ON_SURFACE_VARIANT};
        opacity: 0.75; margin-top: -4px;
    }}

    /* ---- header bar ---- */
    .top-header {{
        display:flex; justify-content:space-between; align-items:center;
        padding: 6px 4px 18px 4px;
    }}
    .top-title {{
        font-family:'Hanken Grotesk',sans-serif; font-weight:700; font-size:26px; color:{ON_SURFACE};
    }}
    .top-sub {{ font-family:'Inter',sans-serif; font-size:13px; color:{ON_SURFACE_VARIANT}; margin-top:-4px;}}
    .profile-pill {{
        display:flex; align-items:center; gap:10px; padding-left:14px; border-left:1px solid {OUTLINE_VARIANT};
    }}
    .avatar-circle {{
        width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
        color:white; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:13px;
        font-family:'Inter',sans-serif;
    }}
    .status-pill {{
        display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace;
        font-size:11px; padding:5px 10px; border-radius:999px; border:1px solid {OUTLINE_VARIANT};
        color:{ON_SURFACE_VARIANT};
    }}
    .pulse-dot {{
        width:7px;height:7px;border-radius:50%; background:{SECONDARY};
        box-shadow:0 0 0 0 rgba(16,185,129,0.6); animation: pulse 1.8s infinite;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0.55); }}
        70% {{ box-shadow: 0 0 0 6px rgba(16,185,129,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
    }}

    /* ---- kpi card internals ---- */
    .kpi-top {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;}}
    .kpi-label {{ font-family:'Inter',sans-serif; font-size:11px; letter-spacing:0.05em; text-transform:uppercase;
                  font-weight:600; color:{ON_SURFACE_VARIANT}; }}
    .kpi-value-row {{ display:flex; align-items:flex-end; gap:8px; }}
    .kpi-value {{ font-family:'Hanken Grotesk',sans-serif; font-weight:700; font-size:28px; color:{ON_SURFACE}; line-height:1;}}
    .kpi-badge {{ font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:500; padding-bottom:3px;}}
    .kpi-bar-track {{ margin-top:10px; height:5px; background:{SURFACE_HIGH}; border-radius:999px; overflow:hidden;}}
    .kpi-bar-fill {{ height:100%; border-radius:999px; }}
    .kpi-dots {{ margin-top:10px; display:flex; gap:4px; }}
    .kpi-dot {{ width:10px;height:10px;border-radius:50%; }}

    /* ---- section card heading ---- */
    .card-title {{ font-family:'Hanken Grotesk',sans-serif; font-weight:700; font-size:18px; color:{ON_SURFACE}; }}
    .card-sub {{ font-family:'Inter',sans-serif; font-size:12.5px; color:{ON_SURFACE_VARIANT}; margin-top:-2px;}}

    /* ---- donut ---- */
    .donut-wrap {{ display:flex; flex-direction:column; align-items:center; padding:8px 0 4px 0;}}
    .donut {{ width:178px;height:178px;border-radius:50%; display:flex; align-items:center; justify-content:center; }}
    .donut-center {{ width:134px;height:134px;border-radius:50%;background:white;display:flex;
                      flex-direction:column; align-items:center; justify-content:center; }}
    .donut-value {{ font-family:'Hanken Grotesk',sans-serif; font-weight:700; font-size:32px; color:{ON_SURFACE}; line-height:1;}}
    .donut-label {{ font-family:'Inter',sans-serif; font-size:10.5px; text-transform:uppercase; letter-spacing:0.04em;
                     color:{ON_SURFACE_VARIANT}; margin-top:4px; text-align:center; padding:0 10px;}}

    /* ---- comparison table ---- */
    table.cmp-table {{ width:100%; border-collapse:collapse; }}
    table.cmp-table th {{
        text-align:left; padding:10px 14px; background:{SURFACE_LOW};
        font-family:'Inter',sans-serif; font-size:10.5px; letter-spacing:0.05em; text-transform:uppercase;
        color:{ON_SURFACE_VARIANT}; border-bottom:1px solid {OUTLINE_VARIANT};
    }}
    table.cmp-table td {{
        padding:10px 14px; border-bottom:1px solid {OUTLINE_VARIANT};
        font-family:'Inter',sans-serif; font-size:13.5px; color:{ON_SURFACE};
    }}
    table.cmp-table tr:last-child td {{ border-bottom:none; }}
    .status-dot {{ width:8px;height:8px;border-radius:50%; display:inline-block; margin-right:7px; }}
    .mono-val {{ font-family:'JetBrains Mono',monospace; font-size:13px; }}
    .chip {{
        display:inline-block; font-family:'Inter',sans-serif; font-size:11px; font-weight:700;
        padding:3px 9px; border-radius:999px;
    }}

    /* ---- chat panel ---- */
    .chat-bubble-ai {{
        background:{SURFACE_HIGH}; border:1px solid {OUTLINE_VARIANT}; border-radius:10px;
        padding:10px 12px; font-size:13px; color:{ON_SURFACE}; margin-bottom:10px; max-width:92%;
    }}
    .chat-bubble-user {{
        background:{PRIMARY_CONTAINER}; color:{ON_PRIMARY_CONTAINER}; border-radius:10px;
        padding:10px 12px; font-size:13px; margin-bottom:10px; max-width:92%; margin-left:auto;
    }}

    /* ---- footer ---- */
    .footer-bar {{
        display:flex; justify-content:space-between; align-items:center; padding:14px 6px;
        border-top:1px solid {OUTLINE_VARIANT}; margin-top:18px;
        font-family:'JetBrains Mono',monospace; font-size:10.5px; color:{OUTLINE};
    }}

    [data-testid="stMetricValue"] {{ font-family:'Hanken Grotesk',sans-serif; }}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
VALID_PAGES = [key for key, _, _ in NAV_ITEMS]

_qp_page = st.query_params.get("page", "overview")
if _qp_page not in VALID_PAGES:
    _qp_page = "overview"

if "page" not in st.session_state:
    st.session_state.page = _qp_page
elif st.session_state.page not in VALID_PAGES:
    st.session_state.page = "overview"

if "engine" not in st.session_state:
    st.session_state.engine = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "chat" not in st.session_state:
    st.session_state.chat = []
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def generate_sample_dataframe(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Customer ID": [f"CUST-{i:05d}" for i in range(n)],
        "Age": rng.integers(18, 70, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "City": rng.choice(["New York", "Chicago", "Houston", "Phoenix", "Miami"], n),
        "Subscription Type": rng.choice(["Basic", "Standard", "Premium"], n),
        "Monthly Spending": np.round(rng.uniform(10, 200, n), 2),
        "Tenure": rng.integers(1, 60, n),
        "Number of Purchases": rng.integers(0, 50, n),
        "Customer Support Requests": rng.integers(0, 10, n),
        "Login Frequency": rng.integers(0, 30, n),
        "Satisfaction Score": rng.integers(1, 10, n),
        "Last Activity Date": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
    })
    logit = (-1.5 + 0.35 * (10 - df["Satisfaction Score"]) + 0.25 * df["Customer Support Requests"]
             - 0.15 * df["Login Frequency"] - 0.02 * df["Tenure"])
    churn_prob = 1 / (1 + np.exp(-logit / 3))
    df["Churn Status"] = np.where(rng.random(n) < churn_prob, "Yes", "No")
    return df


def validate_schema(df: pd.DataFrame):
    required = set(REQUIRED_NUMERIC + REQUIRED_CATEGORICAL + [REQUIRED_TARGET])
    return sorted(required - set(df.columns))


def run_training(df: pd.DataFrame, source_name: str):
    missing = validate_schema(df)
    if missing:
        st.error(
            "This file is missing required columns: **" + ", ".join(missing) +
            "**. Please match the expected schema and try again."
        )
        return
    with st.spinner("Cleaning data, fitting preprocessor, and training Logistic Regression, "
                     "Random Forest and XGBoost…"):
        engine = ChurnModelEngine()
        try:
            engine.train_and_evaluate(df)
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return
    st.session_state.engine = engine
    st.session_state.trained = True
    st.session_state.source_name = source_name
    st.session_state.raw_df = df.copy()          # ← EDA ke liye save
    auc = engine.metrics_report[engine.best_model_name]["ROC-AUC"]
    f1 = engine.metrics_report[engine.best_model_name]["F1 Score"]
    n_risk = int((engine.test_results["Churn Probability"] > 0.6).sum())
    top_feats = engine.get_feature_importance(3)
    feat_txt = ", ".join(f[0] for f in top_feats) if top_feats else "the engineered feature set"
    st.session_state.chat = [
        ("ai", f"Training complete on {engine.n_records} records. **{engine.best_model_name}** was "
               f"selected as the optimal configuration with an F1 score of {f1:.2f} and ROC-AUC of "
               f"{auc:.2f}. {n_risk} test-set customers are flagged high-risk "
               f"(probability > 0.6). Leading risk signals: {feat_txt}.")
    ]
    st.rerun()


def reset_engine():
    st.session_state.trained = False
    st.session_state.engine = None
    st.session_state.chat = []
    st.session_state.source_name = None
    st.session_state.raw_df = None              # ← reset pe clear
    st.session_state.page = "overview"
    st.query_params["page"] = "overview"


def status_for_auc(auc):
    if auc >= 0.80:
        return "Strong", SECONDARY
    if auc >= 0.65:
        return "Moderate", PRIMARY
    return "Weak", TERTIARY


def kpi_card(col, label, icon, icon_color, value, badge_text, badge_color, bar_pct=None, bar_color=None, dots=None):
    with col:
        with st.container(border=True):
            bar_html = ""
            if bar_pct is not None:
                bar_html = (f'<div class="kpi-bar-track"><div class="kpi-bar-fill" '
                            f'style="width:{bar_pct}%;background:{bar_color};"></div></div>')
            dots_html = ""
            if dots:
                dots_html = '<div class="kpi-dots">' + "".join(
                    f'<div class="kpi-dot" style="background:{c};opacity:{o};"></div>' for c, o in dots
                ) + "</div>"
            st.markdown(
                f"""
                <div class="kpi-top">
                    <span class="kpi-label">{label}</span>
                    <span class="material-symbols-outlined" style="color:{icon_color};font-size:20px;">{icon}</span>
                </div>
                <div class="kpi-value-row">
                    <span class="kpi-value">{value}</span>
                    <span class="kpi-badge" style="color:{badge_color};">{badge_text}</span>
                </div>
                {bar_html}{dots_html}
                """,
                unsafe_allow_html=True,
            )


def sparkline_svg(values_0to1, color):
    xs = [0, 25, 50, 75, 100]
    pts = []
    for x, v in zip(xs, values_0to1):
        v = max(0.0, min(1.0, v))
        y = 27 - (v * 23)
        pts.append(f"{x},{y:.1f}")
    path = " L".join(pts)
    return (f'<svg width="96" height="32" viewBox="0 0 100 30">'
            f'<path d="M{path}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def render_donut(pct, label, color):
    pct_disp = max(0.0, min(100.0, pct))
    st.markdown(
        f"""
        <div class="donut-wrap">
            <div class="donut" style="background: conic-gradient({color} 0% {pct_disp}%, {SURFACE_HIGH} {pct_disp}% 100%);">
                <div class="donut-center">
                    <p class="donut-value">{pct_disp:.1f}%</p>
                    <p class="donut-label">{label}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_theme(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=ON_SURFACE_VARIANT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=OUTLINE_VARIANT, zeroline=False)
    fig.update_yaxes(gridcolor=OUTLINE_VARIANT, zeroline=False)
    return fig


def feature_importance_figure(engine, top_n=8):
    pairs = engine.get_feature_importance(top_n)
    if not pairs:
        return None
    pairs = pairs[::-1]
    names = [p[0] for p in pairs]
    vals = [float(p[1]) for p in pairs]
    colors = [PRIMARY if i % 2 == 0 else SECONDARY for i in range(len(vals))]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    return plotly_theme(fig, height=max(260, 34 * len(vals)))


def metric_comparison_figure(engine):
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fig = go.Figure()
    for model_name, metrics in engine.metrics_report.items():
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_names,
            y=[metrics[m] for m in metric_names],
            marker_color=MODEL_COLORS.get(model_name, OUTLINE),
        ))
    fig.update_layout(barmode="group")
    return plotly_theme(fig, height=360)


def model_comparison_table_html(engine, compact=True):
    rows = []
    for model_name, metrics in engine.metrics_report.items():
        auc = metrics["ROC-AUC"]
        status_label, status_color = status_for_auc(auc)
        spark_vals = [metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
                      metrics["F1 Score"], metrics["ROC-AUC"]]
        spark = sparkline_svg(spark_vals, MODEL_COLORS.get(model_name, PRIMARY))
        is_best = model_name == engine.best_model_name
        action = (f'<span class="chip" style="background:{PRIMARY_CONTAINER};color:{ON_PRIMARY_CONTAINER};">Active</span>'
                  if is_best else '<span style="color:' + OUTLINE + ';font-size:12px;">—</span>')
        rows.append(f"""
        <tr>
            <td>{model_name}</td>
            <td><span class="status-dot" style="background:{status_color};"></span>
                <span class="mono-val">{status_label} ({auc:.2f})</span></td>
            <td>{spark}</td>
            <td style="text-align:right;">{action}</td>
        </tr>""")
    return f"""
    <table class="cmp-table">
        <thead><tr>
            <th>Model</th><th>ROC-AUC Status</th><th>Metric Trend</th><th style="text-align:right;">Status</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


def answer_query(engine, query: str) -> str:
    q = query.lower()
    best = engine.best_model_name
    m = engine.metrics_report[best]
    if any(k in q for k in ["best", "winner", "selected", "optimal"]):
        return (f"**{best}** was selected as the optimal model — it has the highest F1 score "
                f"({m['F1 Score']:.2f}) among the three candidates evaluated.")
    if "auc" in q or "roc" in q:
        ranking = sorted(engine.metrics_report.items(), key=lambda kv: kv[1]["ROC-AUC"], reverse=True)
        lines = ", ".join(f"{n} ({v['ROC-AUC']:.2f})" for n, v in ranking)
        return f"ROC-AUC ranking: {lines}."
    if "accura" in q:
        ranking = sorted(engine.metrics_report.items(), key=lambda kv: kv[1]["Accuracy"], reverse=True)
        lines = ", ".join(f"{n} ({v['Accuracy']:.2f})" for n, v in ranking)
        return f"Accuracy ranking: {lines}."
    if "risk" in q or "churn" in q and "feature" not in q:
        n_high = int((engine.test_results["Churn Probability"] > 0.7).sum())
        n_med = int(((engine.test_results["Churn Probability"] > 0.4) &
                      (engine.test_results["Churn Probability"] <= 0.7)).sum())
        return (f"In the held-out test set, {n_high} customers are high-risk (probability > 0.70) "
                f"and {n_med} are medium-risk (0.40–0.70). Overall dataset churn rate is "
                f"{engine.overall_churn_rate:.1f}%.")
    if "feature" in q or "driver" in q or "important" in q:
        top = engine.get_feature_importance(5)
        if not top:
            return "Feature importance isn't available for this model type."
        lines = ", ".join(f"{n} ({v:.2f})" for n, v in top)
        return f"Top churn drivers per {best}: {lines}."
    if "record" in q or "data" in q or "size" in q:
        return f"{engine.n_records} records were processed after cleaning and de-duplication."
    return (f"I can answer questions about the best model, ROC-AUC, accuracy, risk counts, "
            f"top churn drivers, or dataset size. Currently **{best}** leads with F1 "
            f"{m['F1 Score']:.2f} and ROC-AUC {m['ROC-AUC']:.2f}.")


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom:24px;">
            <p class="brand-title">Churn Intelligence</p>
            <p class="brand-sub">Precision Retention Engine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for key, icon, label in NAV_ITEMS:
        if st.session_state.page == key:
            st.markdown(
                f'<div class="nav-active">'
                f'<span class="material-symbols-outlined" style="font-size:18px;">{icon}</span>'
                f'<span>{label}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(label, key=f"nav_{key}", icon=":material/" + icon + ":", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    if st.button("Run New Analysis", type="primary", use_container_width=True, icon=":material/refresh:"):
        reset_engine()
        st.rerun()

    if st.session_state.trained:
        eng = st.session_state.engine
        st.markdown(
            f"""
            <div style="margin-top:24px;padding:12px;border-top:1px solid {OUTLINE_VARIANT};">
                <p class="font-label-caps" style="font-family:'Inter',sans-serif;font-size:10.5px;
                   letter-spacing:0.05em;text-transform:uppercase;color:{ON_SURFACE_VARIANT};">
                   Active Source</p>
                <p style="font-size:12.5px;color:{ON_SURFACE};margin-top:2px;">
                   {st.session_state.source_name or 'Uploaded dataset'}</p>
                <p style="font-size:11px;color:{ON_SURFACE_VARIANT};margin-top:6px;">
                   {eng.n_records} records · {eng.best_model_name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
# Guard: if cached page key no longer exists (e.g. after an update), reset
if st.session_state.page not in PAGE_TITLES:
    st.session_state.page = "overview"
    st.query_params["page"] = "overview"

title, subtitle = PAGE_TITLES[st.session_state.page]
status_text = "MODEL TRAINED" if st.session_state.trained else "AWAITING DATA"
status_color = SECONDARY if st.session_state.trained else OUTLINE

header_col1, header_col2 = st.columns([3, 1.3])
with header_col1:
    st.markdown(
        f"""
        <div class="top-header">
            <div>
                <p class="top-title">{title}</p>
                <p class="top-sub">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown(
        f"""
        <div style="display:flex;justify-content:flex-end;align-items:center;gap:14px;padding-top:8px;">
            <span class="status-pill"><span class="pulse-dot" style="background:{status_color};
                  animation:{'pulse 1.8s infinite' if st.session_state.trained else 'none'};"></span>{status_text}</span>
            <div class="profile-pill">
                <div style="text-align:right;">
                    <p style="font-size:11px;font-weight:700;color:{ON_SURFACE};margin:0;line-height:1.1;">khizar hayat</p>
                    <p style="font-size:9.5px;color:{ON_SURFACE_VARIANT};margin:0;text-transform:uppercase;letter-spacing:0.04em;">Ai Engineer</p>
                </div>
                <div class="avatar-circle">KH</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(f"<div style='height:1px;background:{OUTLINE_VARIANT};margin-bottom:18px;'></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: OVERVIEW
# ----------------------------------------------------------------------------
def render_upload_gate():
    with st.container(border=True):
        st.markdown('<p class="card-title">Run a churn analysis</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="card-sub">Upload a customer dataset to clean, engineer features, train three '
            f'candidate models, and surface the optimal configuration. Required columns: '
            f'{", ".join(REQUIRED_NUMERIC + REQUIRED_CATEGORICAL + [REQUIRED_TARGET])}.</p>',
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2 = st.columns([3, 1.4])
        with c1:
            uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        with c2:
            use_sample = st.button("Use Sample Dataset", use_container_width=True)

        if use_sample:
            run_training(generate_sample_dataframe(), "Synthetic sample dataset")
        elif uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read this file as CSV: {exc}")
                df = None
            if df is not None:
                missing = validate_schema(df)
                if missing:
                    st.error("Missing required columns: **" + ", ".join(missing) + "**")
                else:
                    st.success(f"Loaded {len(df)} rows. Ready to train.")
                    if st.button("Run Training", type="primary"):
                        run_training(df, uploaded.name)

        st.write("")
        template_df = generate_sample_dataframe(n=20)
        st.download_button(
            "Download CSV template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_data_template.csv",
            mime="text/csv",
        )


def render_overview():
    if not st.session_state.trained:
        render_upload_gate()
        return

    engine = st.session_state.engine
    best = engine.metrics_report[engine.best_model_name]
    auc_status, auc_color = status_for_auc(best["ROC-AUC"])
    n_risk = int((engine.test_results["Churn Probability"] > 0.6).sum())
    n_test = len(engine.test_results)
    risk_pct = (n_risk / n_test * 100) if n_test else 0

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Best Model", "military_tech", SECONDARY, engine.best_model_name,
              "Selected", PRIMARY, bar_pct=100, bar_color=SECONDARY)
    kpi_card(c2, "F1 Score", "analytics", PRIMARY, f"{best['F1 Score']:.2f}",
              "Optimal" if best["F1 Score"] >= 0.5 else "Needs Review",
              SECONDARY if best["F1 Score"] >= 0.5 else TERTIARY,
              bar_pct=best["F1 Score"] * 100, bar_color=PRIMARY)
    kpi_card(c3, "ROC-AUC", "trending_up", auc_color, f"{best['ROC-AUC']:.2f}",
              auc_status, auc_color, bar_pct=best["ROC-AUC"] * 100, bar_color=auc_color)
    kpi_card(c4, "High-Risk Customers", "error_outline", TERTIARY, str(n_risk),
              f"{risk_pct:.0f}% of test set", TERTIARY,
              dots=[(TERTIARY, 1), (TERTIARY, 0.5), (TERTIARY, 0.2)])

    st.write("")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        with st.container(border=True):
            st.markdown('<p class="card-title">Feature Signals</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Top drivers of churn risk identified by the optimal model</p>',
                        unsafe_allow_html=True)
            fig = feature_importance_figure(engine, top_n=8)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Feature importance is not available for this model type.")
    with col_right:
        with st.container(border=True):
            st.markdown('<p class="card-title">Dataset Churn Rate</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Share of customers labeled as churned</p>', unsafe_allow_html=True)
            render_donut(engine.overall_churn_rate, "Churned of Total Base", TERTIARY)

    st.write("")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        with st.container(border=True):
            st.markdown('<p class="card-title">Model Comparison</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">All candidates evaluated on the held-out test split</p>',
                        unsafe_allow_html=True)
            st.markdown(model_comparison_table_html(engine), unsafe_allow_html=True)
    with col_b:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
                                display:flex;align-items:center;justify-content:center;">
                        <span class="material-symbols-outlined" style="color:white;font-size:18px;">auto_awesome</span>
                    </div>
                    <div>
                        <p style="font-size:11px;font-weight:700;color:{ON_SURFACE};margin:0;letter-spacing:0.04em;text-transform:uppercase;">Churn Insight AI</p>
                        <p style="font-size:9.5px;color:{SECONDARY};margin:0;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;">Ready to Assist</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            chat_html = ""
            for role, msg in st.session_state.chat:
                cls = "chat-bubble-ai" if role == "ai" else "chat-bubble-user"
                chat_html += f'<div class="{cls}">{msg}</div>'
            st.markdown(f'<div style="max-height:260px;overflow-y:auto;">{chat_html}</div>', unsafe_allow_html=True)
            q = st.text_input("Ask about the model results...", key="insight_q",
                              label_visibility="collapsed", placeholder="Ask about the model results...")
            if st.button("Send", key="insight_send", use_container_width=True):
                if q.strip():
                    answer = answer_query(engine, q.strip())
                    st.session_state.chat.append(("user", q.strip()))
                    st.session_state.chat.append(("ai", answer))
                    st.rerun()


# ----------------------------------------------------------------------------
# PAGE: EDA
# ----------------------------------------------------------------------------
def render_eda():
    if not st.session_state.trained:
        render_upload_gate()
        return

    raw_df = st.session_state.get("raw_df", None)
    if raw_df is None:
        st.info("EDA data not available. Please re-run analysis.")
        return

    df = raw_df.copy()
    churn_yes = df[df["Churn Status"] == "Yes"]
    churn_no  = df[df["Churn Status"] == "No"]

    # ── KPI strip ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    total     = len(df)
    churned   = len(churn_yes)
    retained  = len(churn_no)
    churn_rate = round(churned / total * 100, 1)
    avg_spend  = round(df["Monthly Spending"].mean(), 1)

    def mini_kpi(col, label, value, color):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="kpi-top"><span class="kpi-label">{label}</span></div>'
                    f'<div class="kpi-value-row">'
                    f'<span class="kpi-value" style="font-size:22px;color:{color};">{value}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    mini_kpi(k1, "Total Customers",   f"{total:,}",     ON_SURFACE)
    mini_kpi(k2, "Churned",           f"{churned:,}",   TERTIARY)
    mini_kpi(k3, "Retained",          f"{retained:,}",  SECONDARY)
    mini_kpi(k4, "Churn Rate",        f"{churn_rate}%", TERTIARY)
    mini_kpi(k5, "Avg Monthly Spend", f"${avg_spend}",  PRIMARY)

    st.write("")

    # ── Row 1: Churn Distribution + Gender ───────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<p class="card-title">Churn Distribution</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Overall Yes vs No split across full dataset</p>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=["Retained (No)", "Churned (Yes)"],
                y=[retained, churned],
                marker_color=[SECONDARY, TERTIARY],
                text=[f"{retained:,}", f"{churned:,}"],
                textposition="outside",
                hovertemplate="%{x}: %{y}<extra></extra>",
            ))
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_theme(fig, 280), use_container_width=True, config={"displayModeBar": False})

    with col2:
        with st.container(border=True):
            st.markdown('<p class="card-title">Churn by Gender</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Male vs Female churn rate comparison</p>', unsafe_allow_html=True)
            gender_churn = df.groupby("Gender")["Churn Status"].apply(
                lambda x: (x == "Yes").sum() / len(x) * 100
            ).reset_index()
            gender_churn.columns = ["Gender", "Churn Rate (%)"]
            fig2 = go.Figure(go.Bar(
                x=gender_churn["Gender"],
                y=gender_churn["Churn Rate (%)"],
                marker_color=[PRIMARY, SECONDARY],
                text=[f"{v:.1f}%" for v in gender_churn["Churn Rate (%)"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            fig2.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
            st.plotly_chart(plotly_theme(fig2, 280), use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Row 2: Age + Subscription Type ───────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown('<p class="card-title">Age Distribution by Churn</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Churned vs retained customers across age groups</p>', unsafe_allow_html=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=churn_yes["Age"], name="Churned",  marker_color=TERTIARY,  opacity=0.75, nbinsx=20))
            fig3.add_trace(go.Histogram(x=churn_no["Age"],  name="Retained", marker_color=SECONDARY, opacity=0.75, nbinsx=20))
            fig3.update_layout(barmode="overlay", xaxis_title="Age", yaxis_title="Count")
            st.plotly_chart(plotly_theme(fig3, 280), use_container_width=True, config={"displayModeBar": False})

    with col4:
        with st.container(border=True):
            st.markdown('<p class="card-title">Churn by Subscription Type</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Which subscription tier churns the most</p>', unsafe_allow_html=True)
            sub_churn = df.groupby("Subscription Type")["Churn Status"].apply(
                lambda x: (x == "Yes").sum() / len(x) * 100
            ).reset_index()
            sub_churn.columns = ["Subscription Type", "Churn Rate (%)"]
            fig4 = go.Figure(go.Bar(
                x=sub_churn["Subscription Type"],
                y=sub_churn["Churn Rate (%)"],
                marker_color=[PRIMARY, SECONDARY, TERTIARY],
                text=[f"{v:.1f}%" for v in sub_churn["Churn Rate (%)"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            fig4.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
            st.plotly_chart(plotly_theme(fig4, 280), use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Row 3: Monthly Spending + Satisfaction Score ──────────────────────────
    col5, col6 = st.columns(2)
    with col5:
        with st.container(border=True):
            st.markdown('<p class="card-title">Monthly Spending by Churn</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Spending pattern difference between churned & retained</p>', unsafe_allow_html=True)
            fig5 = go.Figure()
            fig5.add_trace(go.Box(y=churn_yes["Monthly Spending"], name="Churned",  marker_color=TERTIARY,  boxmean=True))
            fig5.add_trace(go.Box(y=churn_no["Monthly Spending"],  name="Retained", marker_color=SECONDARY, boxmean=True))
            fig5.update_layout(yaxis_title="Monthly Spending ($)")
            st.plotly_chart(plotly_theme(fig5, 280), use_container_width=True, config={"displayModeBar": False})

    with col6:
        with st.container(border=True):
            st.markdown('<p class="card-title">Satisfaction Score vs Churn Rate</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">How satisfaction score relates to churn</p>', unsafe_allow_html=True)
            sat_churn = df.groupby("Satisfaction Score")["Churn Status"].apply(
                lambda x: (x == "Yes").sum() / len(x) * 100
            ).reset_index()
            sat_churn.columns = ["Score", "Churn Rate (%)"]
            fig6 = go.Figure(go.Bar(
                x=sat_churn["Score"],
                y=sat_churn["Churn Rate (%)"],
                marker_color=[TERTIARY if v >= 40 else (PRIMARY if v >= 20 else SECONDARY) for v in sat_churn["Churn Rate (%)"]],
                text=[f"{v:.0f}%" for v in sat_churn["Churn Rate (%)"]],
                textposition="outside",
                hovertemplate="Score %{x}: %{y:.1f}% churn<extra></extra>",
            ))
            fig6.update_layout(showlegend=False, xaxis_title="Satisfaction Score (1–10)", yaxis_title="Churn Rate (%)")
            st.plotly_chart(plotly_theme(fig6, 280), use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Row 4: Tenure + Login Frequency ──────────────────────────────────────
    col7, col8 = st.columns(2)
    with col7:
        with st.container(border=True):
            st.markdown('<p class="card-title">Tenure Distribution by Churn</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Do newer customers churn more?</p>', unsafe_allow_html=True)
            fig7 = go.Figure()
            fig7.add_trace(go.Histogram(x=churn_yes["Tenure"], name="Churned",  marker_color=TERTIARY,  opacity=0.75, nbinsx=20))
            fig7.add_trace(go.Histogram(x=churn_no["Tenure"],  name="Retained", marker_color=SECONDARY, opacity=0.75, nbinsx=20))
            fig7.update_layout(barmode="overlay", xaxis_title="Tenure (months)", yaxis_title="Count")
            st.plotly_chart(plotly_theme(fig7, 280), use_container_width=True, config={"displayModeBar": False})

    with col8:
        with st.container(border=True):
            st.markdown('<p class="card-title">Login Frequency vs Churn</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Engagement level impact on customer retention</p>', unsafe_allow_html=True)
            fig8 = go.Figure()
            fig8.add_trace(go.Box(y=churn_yes["Login Frequency"], name="Churned",  marker_color=TERTIARY,  boxmean=True))
            fig8.add_trace(go.Box(y=churn_no["Login Frequency"],  name="Retained", marker_color=SECONDARY, boxmean=True))
            fig8.update_layout(yaxis_title="Login Frequency (per month)")
            st.plotly_chart(plotly_theme(fig8, 280), use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Row 5: Correlation Heatmap (full width) ───────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="card-title">Correlation Heatmap</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">Pearson correlation between all numeric features and churn</p>', unsafe_allow_html=True)
        numeric_cols = ["Age", "Monthly Spending", "Tenure", "Number of Purchases",
                        "Customer Support Requests", "Login Frequency", "Satisfaction Score"]
        corr_df = df[numeric_cols].copy()
        corr_df["Churn"] = (df["Churn Status"] == "Yes").astype(int)
        corr_matrix = corr_df.corr().round(2)
        fig9 = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale=[[0, TERTIARY], [0.5, SURFACE_HIGH], [1, SECONDARY]],
            zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
        ))
        fig9.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter, sans-serif", color=ON_SURFACE_VARIANT, size=11))
        st.plotly_chart(fig9, use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Row 6: Support Requests + Top Cities ─────────────────────────────────
    col9, col10 = st.columns(2)
    with col9:
        with st.container(border=True):
            st.markdown('<p class="card-title">Support Requests vs Churn</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">More support tickets = higher churn risk?</p>', unsafe_allow_html=True)
            sr_churn = df.groupby("Customer Support Requests")["Churn Status"].apply(
                lambda x: (x == "Yes").sum() / len(x) * 100
            ).reset_index()
            sr_churn.columns = ["Support Requests", "Churn Rate (%)"]
            fig10 = go.Figure(go.Scatter(
                x=sr_churn["Support Requests"],
                y=sr_churn["Churn Rate (%)"],
                mode="lines+markers",
                line=dict(color=TERTIARY, width=2.5),
                marker=dict(size=7, color=TERTIARY),
                hovertemplate="Requests %{x}: %{y:.1f}%<extra></extra>",
            ))
            fig10.update_layout(xaxis_title="Support Requests", yaxis_title="Churn Rate (%)")
            st.plotly_chart(plotly_theme(fig10, 280), use_container_width=True, config={"displayModeBar": False})

    with col10:
        with st.container(border=True):
            st.markdown('<p class="card-title">Top 10 Cities by Churn Rate</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Geographic churn concentration</p>', unsafe_allow_html=True)
            city_churn = df.groupby("City")["Churn Status"].apply(
                lambda x: (x == "Yes").sum() / len(x) * 100
            ).reset_index()
            city_churn.columns = ["City", "Churn Rate (%)"]
            city_churn = city_churn.sort_values("Churn Rate (%)", ascending=True).tail(10)
            fig11 = go.Figure(go.Bar(
                x=city_churn["Churn Rate (%)"],
                y=city_churn["City"],
                orientation="h",
                marker_color=PRIMARY,
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig11.update_layout(xaxis_title="Churn Rate (%)")
            st.plotly_chart(plotly_theme(fig11, 280), use_container_width=True, config={"displayModeBar": False})



# ----------------------------------------------------------------------------
# PAGE: CUSTOMER SEGMENTATION
# ----------------------------------------------------------------------------
def render_segments():
    if not st.session_state.trained:
        render_upload_gate()
        return
    engine = st.session_state.engine

    if not hasattr(engine, 'segment_df') or engine.segment_df is None:
        st.warning("Segmentation not available. Click **Run New Analysis** in the sidebar to retrain with the new engine.")
        return

    seg = engine.segment_df.copy()
    seg.columns = ["Segment", "Customers", "Churn Rate (%)", "Avg Age",
                   "Avg Spending ($)", "Avg Tenure (mo)", "Avg Satisfaction",
                   "Avg Support Req", "Avg Logins/mo"]

    SEGMENT_COLORS = [PRIMARY, SECONDARY, TERTIARY, "#a855f7"]
    SEGMENT_NAMES  = ["Champions", "At-Risk", "Loyal", "New Customers"]

    # ── KPI strip ────────────────────────────────────────────────────────────
    cols = st.columns(engine.n_segments)
    for i, row in seg.iterrows():
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        name  = SEGMENT_NAMES[i % len(SEGMENT_NAMES)]
        with cols[i]:
            with st.container(border=True):
                st.markdown(
                    f'''
                    <div class="kpi-top">
                        <span class="kpi-label">Segment {int(row["Segment"])} — {name}</span>
                        <span class="material-symbols-outlined" style="color:{color};font-size:20px;">donut_small</span>
                    </div>
                    <div class="kpi-value-row">
                        <span class="kpi-value" style="font-size:22px;">{int(row["Customers"]):,}</span>
                        <span class="kpi-badge" style="color:{color};">customers</span>
                    </div>
                    <div class="kpi-bar-track">
                        <div class="kpi-bar-fill" style="width:{min(row["Churn Rate (%)"], 100)}%;background:{color};"></div>
                    </div>
                    <p style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{color};margin-top:6px;">
                        {row["Churn Rate (%)"]}% churn rate</p>
                    ''',
                    unsafe_allow_html=True,
                )

    st.write("")

    # ── Churn Rate by Segment bar ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<p class="card-title">Churn Rate by Segment</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Which cluster is highest risk</p>', unsafe_allow_html=True)
            fig1 = go.Figure(go.Bar(
                x=[f"Seg {int(r['Segment'])} — {SEGMENT_NAMES[i % len(SEGMENT_NAMES)]}" for i, r in seg.iterrows()],
                y=seg["Churn Rate (%)"],
                marker_color=[SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i in range(len(seg))],
                text=[f"{v}%" for v in seg["Churn Rate (%)"]],
                textposition="outside",
                hovertemplate="%{x}: %{y}%<extra></extra>",
            ))
            fig1.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
            st.plotly_chart(plotly_theme(fig1, 280), use_container_width=True, config={"displayModeBar": False})

    with col2:
        with st.container(border=True):
            st.markdown('<p class="card-title">Segment Size Distribution</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-sub">Customer count per cluster</p>', unsafe_allow_html=True)
            fig2 = go.Figure(go.Pie(
                labels=[f"Seg {int(r['Segment'])} — {SEGMENT_NAMES[i % len(SEGMENT_NAMES)]}" for i, r in seg.iterrows()],
                values=seg["Customers"],
                marker_colors=SEGMENT_COLORS,
                hole=0.45,
                hovertemplate="%{label}: %{value} customers<extra></extra>",
            ))
            fig2.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=ON_SURFACE_VARIANT, size=12),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Radar / Spider comparison ─────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="card-title">Segment Profile Comparison</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">Normalized feature averages per cluster — bigger area = higher engagement</p>', unsafe_allow_html=True)

        radar_cols = ["Avg Age", "Avg Spending ($)", "Avg Tenure (mo)",
                      "Avg Satisfaction", "Avg Support Req", "Avg Logins/mo"]
        radar_labels = ["Age", "Spending", "Tenure", "Satisfaction", "Support Req", "Logins"]

        # Normalize 0-1 per column
        norm = seg[radar_cols].copy()
        for col in radar_cols:
            mn, mx = norm[col].min(), norm[col].max()
            norm[col] = (norm[col] - mn) / (mx - mn + 1e-9)

        fig3 = go.Figure()
        for i, row in norm.iterrows():
            vals = list(row[radar_cols]) + [row[radar_cols[0]]]   # close polygon
            lbs  = radar_labels + [radar_labels[0]]
            name = f"Seg {int(seg.iloc[i]['Segment'])} — {SEGMENT_NAMES[i % len(SEGMENT_NAMES)]}"
            fig3.add_trace(go.Scatterpolar(
                r=vals, theta=lbs, fill="toself", name=name,
                line_color=SEGMENT_COLORS[i % len(SEGMENT_COLORS)],
                opacity=0.65,
            ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=380, margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=ON_SURFACE_VARIANT, size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Detailed segment table ────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="card-title">Segment Detail Table</p>', unsafe_allow_html=True)
        display_seg = seg.copy()
        display_seg["Segment"] = display_seg["Segment"].apply(
            lambda x: f"Segment {int(x)} — {SEGMENT_NAMES[int(x) % len(SEGMENT_NAMES)]}"
        )
        st.dataframe(
            display_seg,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Churn Rate (%)": st.column_config.ProgressColumn(
                    "Churn Rate (%)", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
            },
        )

    st.write("")

    # ── Hyperparameter optimization results ───────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="card-title">Hyperparameter Optimization Results</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="card-sub">RandomizedSearchCV was run on <b>{engine.best_model_name}</b> '
            f'(best base model by F1) — optimized parameters shown below.</p>',
            unsafe_allow_html=True,
        )
        if engine.best_params:
            params_df = pd.DataFrame(
                [{"Parameter": k, "Optimized Value": str(v)} for k, v in engine.best_params.items()]
            )
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hyperparameter search was run for this model type.")

# ----------------------------------------------------------------------------
# PAGE: FEATURE SIGNALS
# ----------------------------------------------------------------------------
def render_features():
    if not st.session_state.trained:
        render_upload_gate()
        return
    engine = st.session_state.engine
    with st.container(border=True):
        st.markdown('<p class="card-title">Top Churn Drivers</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="card-sub">Computed from the selected model — {engine.best_model_name}</p>',
                    unsafe_allow_html=True)
        fig = feature_importance_figure(engine, top_n=12)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Feature importance is not available for this model type.")

    pairs = engine.get_feature_importance(12)
    if pairs:
        with st.container(border=True):
            st.markdown('<p class="card-title">Importance Detail</p>', unsafe_allow_html=True)
            imp_df = pd.DataFrame(pairs, columns=["Feature", "Importance"])
            st.dataframe(
                imp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Importance": st.column_config.ProgressColumn(
                        "Importance", min_value=0.0,
                        max_value=float(imp_df["Importance"].max()) if len(imp_df) else 1.0, format="%.3f"
                    )
                },
            )


# ----------------------------------------------------------------------------
# PAGE: MODEL COMPARISON
# ----------------------------------------------------------------------------
def render_models():
    if not st.session_state.trained:
        render_upload_gate()
        return
    engine = st.session_state.engine

    with st.container(border=True):
        st.markdown('<p class="card-title">Metric Comparison</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">Accuracy, precision, recall, F1 and ROC-AUC across all candidates</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(metric_comparison_figure(engine), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown('<p class="card-title">Full Evaluation Table</p>', unsafe_allow_html=True)
        metrics_df_raw = {
            name: {k: v for k, v in m.items() if k != "Confusion Matrix"}
            for name, m in engine.metrics_report.items()
        }
        metrics_df = pd.DataFrame(metrics_df_raw).T.reset_index().rename(columns={"index": "Model"})
        metrics_df.insert(1, "Selected", metrics_df["Model"].eq(engine.best_model_name).map({True: "★", False: ""}))
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Accuracy":  st.column_config.ProgressColumn("Accuracy",  min_value=0.0, max_value=1.0, format="%.3f"),
                "Precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.3f"),
                "Recall":    st.column_config.ProgressColumn("Recall",    min_value=0.0, max_value=1.0, format="%.3f"),
                "F1 Score":  st.column_config.ProgressColumn("F1 Score",  min_value=0.0, max_value=1.0, format="%.3f"),
                "ROC-AUC":   st.column_config.ProgressColumn("ROC-AUC",   min_value=0.0, max_value=1.0, format="%.3f"),
            },
        )

    with st.container(border=True):
        st.markdown('<p class="card-title">Confusion Matrices</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">True/False Positive and Negative counts on the held-out test split</p>', unsafe_allow_html=True)
        cm_cols = st.columns(len(engine.metrics_report))
        for col, (model_name, metrics) in zip(cm_cols, engine.metrics_report.items()):
            cm = metrics.get("Confusion Matrix")
            if cm is None:
                continue
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            color = MODEL_COLORS.get(model_name, PRIMARY)
            is_best = model_name == engine.best_model_name
            best_badge = f' <span style="font-size:10px;color:{color};">★ Selected</span>' if is_best else ""
            with col:
                st.markdown(
                    f"""
                    <div style="text-align:center;margin-bottom:8px;">
                        <span style="font-family:'Inter',sans-serif;font-size:12px;font-weight:700;color:{ON_SURFACE};">{model_name}{best_badge}</span>
                    </div>
                    <table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:13px;text-align:center;">
                        <tr>
                            <td style="padding:4px;color:{ON_SURFACE_VARIANT};font-size:10px;"></td>
                            <td style="padding:4px;color:{ON_SURFACE_VARIANT};font-size:10px;font-weight:600;">Pred No</td>
                            <td style="padding:4px;color:{ON_SURFACE_VARIANT};font-size:10px;font-weight:600;">Pred Yes</td>
                        </tr>
                        <tr>
                            <td style="padding:4px;color:{ON_SURFACE_VARIANT};font-size:10px;font-weight:600;">Act No</td>
                            <td style="padding:8px;background:{SECONDARY_CONTAINER};color:{ON_SECONDARY_CONTAINER};border-radius:6px;font-weight:700;">{tn}<br><span style="font-size:9px;font-weight:400;">TN</span></td>
                            <td style="padding:8px;background:{ERROR_CONTAINER};color:{ERROR};border-radius:6px;font-weight:700;">{fp}<br><span style="font-size:9px;font-weight:400;">FP</span></td>
                        </tr>
                        <tr>
                            <td style="padding:4px;color:{ON_SURFACE_VARIANT};font-size:10px;font-weight:600;">Act Yes</td>
                            <td style="padding:8px;background:{ERROR_CONTAINER};color:{ERROR};border-radius:6px;font-weight:700;">{fn}<br><span style="font-size:9px;font-weight:400;">FN</span></td>
                            <td style="padding:8px;background:{SECONDARY_CONTAINER};color:{ON_SECONDARY_CONTAINER};border-radius:6px;font-weight:700;">{tp}<br><span style="font-size:9px;font-weight:400;">TP</span></td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )


# ----------------------------------------------------------------------------
# PAGE: RISK SEGMENTS
# ----------------------------------------------------------------------------
def render_risk():
    if not st.session_state.trained:
        render_upload_gate()
        return
    engine = st.session_state.engine
    results = engine.test_results

    n_total = len(results)
    n_high = int((results["Churn Probability"] > 0.7).sum())
    n_med  = int(((results["Churn Probability"] > 0.4) & (results["Churn Probability"] <= 0.7)).sum())
    n_low  = n_total - n_high - n_med

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Test Customers", "groups",        PRIMARY,  str(n_total), "Held-out set",       ON_SURFACE_VARIANT)
    kpi_card(c2, "High Risk",      "error_outline", TERTIARY, str(n_high),  "> 0.70 probability", TERTIARY)
    kpi_card(c3, "Medium Risk",    "warning",       PRIMARY,  str(n_med),   "0.40 – 0.70",        PRIMARY)
    kpi_card(c4, "Low Risk",       "check_circle",  SECONDARY,str(n_low),   "< 0.40 probability", SECONDARY)

    st.write("")
    with st.container(border=True):
        st.markdown('<p class="card-title">Customer Risk Table</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">Ranked by predicted churn probability on the held-out test split</p>', unsafe_allow_html=True)

        fc1, fc2 = st.columns([1, 2])
        with fc1:
            threshold = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
        with fc2:
            search = st.text_input("Search Customer ID", placeholder="e.g. CUST-00042")

        filtered = results[results["Churn Probability"] >= threshold]
        if search.strip():
            filtered = filtered[filtered["Customer ID"].astype(str).str.contains(search.strip(), case=False)]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            height=420,
            column_config={
                "Churn Probability": st.column_config.ProgressColumn("Churn Probability", min_value=0.0, max_value=1.0, format="%.2f"),
                "Actual Churn":      st.column_config.NumberColumn("Actual Churn"),
                "Predicted Churn":   st.column_config.NumberColumn("Predicted Churn"),
            },
        )
        st.download_button(
            "Download filtered segment (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="churn_risk_segment.csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------------------
# PAGE: PREDICT CUSTOMER
# ----------------------------------------------------------------------------
def render_predict():
    if not st.session_state.trained:
        render_upload_gate()
        return
    engine = st.session_state.engine

    with st.container(border=True):
        st.markdown('<p class="card-title">Individual Customer Prediction</p>', unsafe_allow_html=True)
        st.markdown(
            "<p class=\"card-sub\">Enter a customer's details below to receive a live churn probability "
            "and the top factors contributing to that prediction.</p>",
            unsafe_allow_html=True,
        )
        st.write("")

        col1, col2, col3 = st.columns(3)
        with col1:
            age              = st.number_input("Age", min_value=18, max_value=100, value=35)
            monthly_spending = st.number_input("Monthly Spending ($)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
            tenure           = st.number_input("Tenure (months)", min_value=0, max_value=240, value=24)
        with col2:
            num_purchases    = st.number_input("Number of Purchases", min_value=0, max_value=500, value=10)
            support_requests = st.number_input("Customer Support Requests", min_value=0, max_value=50, value=2)
            login_frequency  = st.number_input("Login Frequency (per month)", min_value=0, max_value=100, value=10)
        with col3:
            satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=10, value=7)
            gender             = st.selectbox("Gender", ["Male", "Female"])
            city               = st.text_input("City", value="New York")
            subscription_type  = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

        st.write("")
        predict_btn = st.button("Predict Churn", type="primary")

    if predict_btn:
        input_df = pd.DataFrame([{
            "Age": age, "Monthly Spending": monthly_spending, "Tenure": tenure,
            "Number of Purchases": num_purchases, "Customer Support Requests": support_requests,
            "Login Frequency": login_frequency, "Satisfaction Score": satisfaction_score,
            "Gender": gender, "City": city, "Subscription Type": subscription_type,
        }])
        try:
            X_proc = engine.preprocessor.transform(input_df[engine.numerical_cols + engine.categorical_cols])
            prob   = float(engine.best_model.predict_proba(X_proc)[0][1])
            pred   = int(prob >= 0.5)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        risk_label = "High Risk"   if prob >= 0.7 else ("Medium Risk" if prob >= 0.4 else "Low Risk")
        risk_color = TERTIARY      if prob >= 0.7 else (PRIMARY       if prob >= 0.4 else SECONDARY)
        verdict    = "LIKELY TO CHURN" if pred == 1 else "LIKELY TO STAY"

        st.write("")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            with st.container(border=True):
                render_donut(prob * 100, "Churn Probability", risk_color)
                st.markdown(
                    f"""
                    <div style="text-align:center;margin-top:8px;">
                        <span class="chip" style="background:{risk_color}20;color:{risk_color};font-size:13px;padding:6px 14px;">{risk_label}</span>
                        <p style="font-family:'Hanken Grotesk',sans-serif;font-weight:700;font-size:16px;color:{risk_color};margin-top:10px;">{verdict}</p>
                        <p style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{ON_SURFACE_VARIANT};">Model: {engine.best_model_name}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with res_col2:
            with st.container(border=True):
                st.markdown('<p class="card-title">Contributing Factors</p>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="card-sub">Top feature importances from the trained model — '
                    'higher values indicate stronger influence on churn risk.</p>',
                    unsafe_allow_html=True,
                )
                top_factors = engine.get_feature_importance(10)
                if top_factors:
                    factors_df = pd.DataFrame(top_factors, columns=["Feature", "Importance"])
                    st.dataframe(
                        factors_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Importance": st.column_config.ProgressColumn(
                                "Importance", min_value=0.0,
                                max_value=float(factors_df["Importance"].max()), format="%.3f"
                            )
                        },
                    )
                else:
                    st.info("Feature importance not available for this model type.")


# ----------------------------------------------------------------------------
# PAGE: REPORTS & EXPORT
# ----------------------------------------------------------------------------
def render_reports():
    if not st.session_state.trained:
        render_upload_gate()
        return
    engine = st.session_state.engine
    best = engine.metrics_report[engine.best_model_name]

    with st.container(border=True):
        st.markdown('<p class="card-title">Run Summary</p>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Best Model",        engine.best_model_name)
        s2.metric("F1 Score",          f"{best['F1 Score']:.3f}")
        s3.metric("ROC-AUC",           f"{best['ROC-AUC']:.3f}")
        s4.metric("Records Processed", engine.n_records)

    with st.container(border=True):
        st.markdown('<p class="card-title">Export Trained Artifacts</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="card-sub">Exports the fitted preprocessor, the selected best model, and run metadata.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Prepare artifacts for download", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path, preproc_path, meta_path = engine.export_artifacts(output_dir=tmpdir)
                with open(model_path,  "rb") as f: st.session_state["_model_bytes"]  = f.read()
                with open(preproc_path,"rb") as f: st.session_state["_preproc_bytes"] = f.read()
                with open(meta_path,   "rb") as f: st.session_state["_meta_bytes"]   = f.read()
            st.success("Artifacts ready below.")

        if "_model_bytes" in st.session_state:
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("best_model.pkl",    st.session_state["_model_bytes"],  file_name="best_model.pkl",    mime="application/octet-stream")
            with d2:
                st.download_button("preprocessor.pkl",  st.session_state["_preproc_bytes"],file_name="preprocessor.pkl",  mime="application/octet-stream")
            with d3:
                st.download_button("metadata.json",     st.session_state["_meta_bytes"],   file_name="metadata.json",     mime="application/json")

    with st.container(border=True):
        st.markdown('<p class="card-title">Shareable Summary Report</p>', unsafe_allow_html=True)
        top_feats     = engine.get_feature_importance(5)
        feat_lines    = "\n".join(f"- {n}: {v:.3f}" for n, v in top_feats) if top_feats else "- Not available"
        metrics_lines = "\n".join(
            f"- {name}: Accuracy {m['Accuracy']:.3f} · Precision {m['Precision']:.3f} · "
            f"Recall {m['Recall']:.3f} · F1 {m['F1 Score']:.3f} · ROC-AUC {m['ROC-AUC']:.3f}"
            for name, m in engine.metrics_report.items()
        )
        report_md = f"""# Churn Intelligence — Run Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Source: {st.session_state.source_name or 'Uploaded dataset'}

## Result
- Records processed: {engine.n_records}
- Overall churn rate: {engine.overall_churn_rate:.2f}%
- Best model (selected by F1): {engine.best_model_name}

## Model Comparison
{metrics_lines}

## Top Churn Drivers
{feat_lines}
"""
        st.download_button("Download summary (Markdown)", report_md.encode("utf-8"),
                            file_name="churn_run_summary.md", mime="text/markdown")
        with st.expander("Preview report"):
            st.markdown(report_md)


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
ROUTES = {
    "overview": render_overview,
    "eda":      render_eda,
    "features": render_features,
    "models":   render_models,
    "segments": render_segments,
    "risk":     render_risk,
    "predict":  render_predict,
    "reports":  render_reports,
}
ROUTES[st.session_state.page]()

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
records_txt = str(st.session_state.engine.n_records) if st.session_state.trained else "—"
model_txt   = st.session_state.engine.best_model_name if st.session_state.trained else "—"
st.markdown(
    f"""
    <div class="footer-bar">
        <div style="display:flex;gap:24px;align-items:center;">
            <span style="display:flex;align-items:center;gap:6px;">
                <span class="pulse-dot" style="background:{SECONDARY};"></span> ENGINE OPERATIONAL
            </span>
            <span>RECORDS: {records_txt}</span>
            <span>MODEL: {model_txt}</span>
        </div>
        <div style="opacity:0.6;">CHURN INTELLIGENCE ENGINE V1.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)
