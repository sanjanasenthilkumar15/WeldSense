"""
app.py — BattleEdge Live Inspection Dashboard
=============================================
Run from the project root:
    streamlit run dashboard/app.py

Implements Steps 13-16 of the implementation plan:
  Step 13 — Streamlit inspection UI + PASS/REJECT result
  Step 14 — 3-column sensor output (image, thermal, spectrogram + audio)
  Step 15 — Plotly risk gauge
  Step 16 — Inspection history table with summary metrics
"""

import os
import sys
import glob
import random
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── path setup so we can import pipeline from the project root ────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.database    import init_db, save_result, load_history, get_summary
from pipeline.pipeline_core import inspect_cell

# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────
IMG_DIR     = os.path.join(ROOT, "data", "images")
THERMAL_DIR = os.path.join(ROOT, "data", "thermal")
AUDIO_DIR   = os.path.join(ROOT, "data", "audio")

# Defect → (image_prefix, thermal_prefix, audio_type)
DEFECT_MAP = {
    "good_weld":      ("good_weld",    "normal",      "normal"),
    "porosity":       ("porosity",     "porosity",    "anomaly"),
    "burn_through":   ("burn_through", "misalignment","anomaly"),
    "contamination":  ("contamination","normal",      "anomaly"),
    "lack_of_fusion": ("lack_of_fusion","cold_weld",  "anomaly"),
    "spatter":        ("spatter",      "normal",      "anomaly"),
    "cold_weld":      ("good_weld",   "cold_weld",    "anomaly"),
    "misalignment":   ("good_weld",   "misalignment", "anomaly"),
}

DEFECT_LABELS = list(DEFECT_MAP.keys())


def _pick_file(directory: str, prefix: str, ext: str) -> str | None:
    """Return a random file matching directory/prefix_*.ext, or None."""
    pattern = os.path.join(directory, f"{prefix}_*.{ext}")
    matches = glob.glob(pattern)
    return random.choice(matches) if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BattleEdge — EV Weld Inspection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #131329 60%, #0f1a2e 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111128 0%, #0a0a1f 100%);
    border-right: 1px solid #2a2a50;
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
}

/* Decision banners */
.banner-pass {
    background: linear-gradient(90deg, #0a2a1a, #0d3a22);
    border: 1px solid #1a6b3a;
    border-left: 5px solid #22c55e;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #22c55e;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.banner-reject {
    background: linear-gradient(90deg, #2a0a0a, #3a0d0d);
    border: 1px solid #6b1a1a;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #ef4444;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

/* Section heading */
.section-title {
    color: #a0a8c8;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.5rem 0 0.5rem 0;
}

/* Stat grid */
.stat-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
.stat-label { font-size: 0.78rem; color: #7080a0; margin-top: 0.1rem; }

/* Table header colour override */
th { color: #a0b0d0 !important; }

/* Risk colours */
.risk-low    { color: #22c55e; }
.risk-medium { color: #f59e0b; }
.risk-high   { color: #ef4444; }

/* Pulse animation for REJECT banner */
@keyframes pulse-red { 0%,100%{opacity:1} 50%{opacity:0.7} }
.banner-reject { animation: pulse-red 2s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Init DB on first run
# ─────────────────────────────────────────────────────────────────────────────
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ BattleEdge")
    st.markdown("*EV Battery Cell Weld Inspector*")
    st.divider()

    st.markdown("### 🔧 Inspection Setup")
    cell_id = st.text_input("Cell ID", value="CELL-001", key="cell_id_input")

    sim_defect = st.selectbox(
        "Simulate defect type",
        DEFECT_LABELS,
        format_func=lambda x: x.replace("_", " ").title(),
        key="defect_selector",
    )

    st.markdown("#### Process Parameters *(optional)*")
    with st.expander("Override weld parameters"):
        voltage    = st.slider("Voltage (V)",   15.0, 35.0, 22.0, 0.5)
        current    = st.slider("Current (A)",   100.0, 300.0, 180.0, 5.0)
        weld_speed = st.slider("Weld Speed (m/s)", 0.2, 1.0, 0.5, 0.05)
        use_custom = st.checkbox("Use these values", value=False)

    run_btn = st.button("▶  Run Inspection", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### 🗄 Database")
    col_db1, col_db2 = st.columns(2)
    summary = get_summary()
    col_db1.metric("Total", summary["total"])
    col_db2.metric("Rejected", summary["n_reject"])

    if st.button("🗑 Reset Database", use_container_width=True):
        from pipeline.database import clear_db
        clear_db()
        st.success("Database cleared.")
        st.rerun()

    st.divider()
    st.caption("BattleEdge v1.0 · Digital Simulation")
    st.caption("Tata Motors Nexon EV — prototype demo")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
'>⚡ BattleEdge — AI Weld Inspection</h1>
<p style='color:#7080a0; font-size:0.9rem; margin-top:0;'>
  Edge-AI warranty risk scoring · Tata Motors Nexon EV production line
</p>
""", unsafe_allow_html=True)

tab_inspect, tab_history, tab_analytics = st.tabs(
    ["🔬 Inspect Cell", "📋 Inspection History", "📊 Analytics"]
)


# ─────────────────────────────────────────────────────────────────────────────
# ── TAB 1: INSPECT CELL ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with tab_inspect:

    # ── Run inspection ────────────────────────────────────────────────────────
    if run_btn:
        img_prefix, th_prefix, aud_type = DEFECT_MAP[sim_defect]

        image_file   = _pick_file(IMG_DIR,     img_prefix, "jpg")
        thermal_file = _pick_file(THERMAL_DIR, th_prefix,  "png")
        audio_file   = _pick_file(AUDIO_DIR,   aud_type,   "wav")

        missing = [n for n, f in [("image", image_file),
                                   ("thermal", thermal_file),
                                   ("audio", audio_file)] if f is None]
        if missing:
            st.error(
                f"Missing data files for: {', '.join(missing)}.\n\n"
                "Please run:  `python data/generate_synthetic.py`"
            )
            st.stop()

        kwargs = dict(
            cell_id      = cell_id,
            image_path   = image_file,
            audio_path   = audio_file,
            thermal_path = thermal_file,
            sim_defect   = sim_defect,
        )
        if use_custom:
            kwargs.update(voltage=voltage, current=current, weld_speed=weld_speed)

        with st.spinner("Running BattleEdge multi-sensor pipeline…"):
            try:
                result = inspect_cell(**kwargs)
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Run `python pipeline/train.py` to train the models first.")
                st.stop()

        save_result(result)

        # ── Decision banner ───────────────────────────────────────────────────
        decision = result["decision"]
        icon     = "✅ PASS" if decision == "PASS" else "❌ REJECT"
        css_cls  = "banner-pass" if decision == "PASS" else "banner-reject"
        st.markdown(
            f'<div class="{css_cls}">{icon} — Cell {result["cell_id"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # ── Step 15: Risk Gauge ───────────────────────────────────────────────
        score = result["risk_score"]

        def risk_gauge(score: int):
            color = "#22c55e" if score < 40 else ("#f59e0b" if score < 65 else "#ef4444")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                title={"text": "Warranty Risk Score", "font": {"color": "#a0b0d0", "size": 14}},
                number={"font": {"color": color, "size": 42}},
                delta={
                    "reference": 65,
                    "increasing": {"color": "#ef4444"},
                    "decreasing": {"color": "#22c55e"},
                    "font": {"size": 14},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#4a5568",
                        "tickfont": {"color": "#7080a0", "size": 11},
                    },
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40], "color": "rgba(34,197,94,0.10)"},
                        {"range": [40, 65], "color": "rgba(245,158,11,0.10)"},
                        {"range": [65,100], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 3},
                        "thickness": 0.85,
                        "value": 65,
                    },
                },
            ))
            fig.update_layout(
                height=260,
                margin=dict(t=40, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"},
            )
            return fig

        col_gauge, col_metrics = st.columns([1, 1])

        with col_gauge:
            st.plotly_chart(risk_gauge(score), use_container_width=True)

        with col_metrics:
            st.markdown("<div class='section-title'>Inspection Results</div>",
                        unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Risk Score",     f"{result['risk_score']} / 100")
            m2.metric("Anomaly Score",  f"{result['anomaly_score']:.3f}")
            m3, m4 = st.columns(2)
            defect_display = result["defect_type"].replace("_", " ").title()
            m3.metric("Defect Type",    defect_display)
            m4.metric("Latency",        f"{result['latency_ms']} ms")

            st.markdown("<div class='section-title'>Thermal Features</div>",
                        unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3)
            t1.metric("Peak",  f"{result['thermal']['peak']:.3f}")
            t2.metric("Mean",  f"{result['thermal']['mean']:.3f}")
            t3.metric("Std",   f"{result['thermal']['std']:.3f}",
                       delta="⚠ uneven" if result["thermal"]["std"] > 0.20 else "✓ even",
                       delta_color="inverse")

            # Defect probability breakdown
            st.markdown("<div class='section-title'>Defect Probabilities</div>",
                        unsafe_allow_html=True)
            proba_df = (
                pd.Series(result["defect_proba"])
                .sort_values(ascending=False)
                .head(4)
                .reset_index()
            )
            proba_df.columns = ["Defect", "Probability"]
            proba_df["Defect"] = proba_df["Defect"].str.replace("_", " ").str.title()
            fig_proba = px.bar(
                proba_df, x="Probability", y="Defect", orientation="h",
                color="Probability",
                color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                range_x=[0, 1],
            )
            fig_proba.update_layout(
                height=180,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                yaxis=dict(tickfont=dict(color="#a0b0d0", size=11)),
                xaxis=dict(tickfont=dict(color="#7080a0", size=10),
                           gridcolor="rgba(255,255,255,0.05)"),
                font={"family": "Inter", "color": "#a0b0d0"},
            )
            st.plotly_chart(fig_proba, use_container_width=True)

        # ── Step 14: Sensor Outputs ───────────────────────────────────────────
        st.divider()
        st.markdown("<div class='section-title'>Sensor Outputs</div>",
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📷 Camera (Visual)**")
            st.image(image_file,
                     caption=f"Weld image — {sim_defect.replace('_',' ').title()}",
                     use_container_width=True)

        with c2:
            st.markdown("**🌡 IR Sensor (Thermal)**")
            st.image(thermal_file,
                     caption=f"Heat map  std={result['thermal']['std']:.3f}",
                     use_container_width=True)

        with c3:
            st.markdown("**🎤 Microphone (Acoustic)**")
            spec_file = audio_file.replace(".wav", ".png")
            if os.path.exists(spec_file):
                st.image(spec_file, caption="Mel spectrogram",
                         use_container_width=True)
            else:
                st.warning("Spectrogram PNG not found.")
            st.audio(audio_file)

        # ── Weld parameters used ──────────────────────────────────────────────
        with st.expander("🔩 Weld Parameters Used"):
            wp = result["weld_params"]
            p1, p2, p3 = st.columns(3)
            p1.metric("Voltage",    f"{wp['voltage']} V")
            p2.metric("Current",    f"{wp['current']} A")
            p3.metric("Weld Speed", f"{wp['weld_speed']} m/s")

    else:
        # ── Placeholder when no inspection has been run ───────────────────────
        st.markdown("""
<div style='
    text-align:center;
    padding: 4rem 2rem;
    color: #4a5568;
    border: 2px dashed #2a3050;
    border-radius: 16px;
    margin-top: 2rem;
'>
    <div style='font-size:3rem;margin-bottom:1rem;'>⚡</div>
    <div style='font-size:1.1rem;font-weight:600;color:#6070a0;'>
        Select a defect type and press ▶ Run Inspection
    </div>
    <div style='font-size:0.85rem;margin-top:0.5rem;'>
        The pipeline will read all 3 sensor modalities and compute a warranty risk score.
    </div>
</div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── TAB 2: INSPECTION HISTORY  — Step 16 ─────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    df = load_history(n=100)

    if df.empty:
        st.info("No inspections yet — run your first inspection in the Inspect Cell tab.")
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Summary</div>",
                    unsafe_allow_html=True)
        s = get_summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inspected",  s["total"])
        c2.metric("Rejected",         s["n_reject"],
                  delta=f"{s['rejection_rate_pct']:.0f}% rejection rate",
                  delta_color="inverse")
        c3.metric("Avg Risk Score",   f"{s['avg_risk']:.1f}")
        c4.metric("Avg Latency",      f"{s['avg_latency_ms']:.0f} ms")

        # ── Risk trend sparkline ──────────────────────────────────────────────
        st.markdown("<div class='section-title'>Risk Score Over Time</div>",
                    unsafe_allow_html=True)
        df_chart = df.sort_values("id")[["id", "risk_score", "decision"]].copy()
        df_chart["colour"] = df_chart["decision"].map(
            {"PASS": "#22c55e", "REJECT": "#ef4444"}
        )
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_chart["id"],
            y=df_chart["risk_score"],
            mode="lines+markers",
            line=dict(color="#60a5fa", width=2),
            marker=dict(
                color=df_chart["colour"],
                size=8,
                line=dict(color="#1e293b", width=1),
            ),
            name="Risk Score",
        ))
        fig_trend.add_hline(y=65, line_dash="dash", line_color="#ef4444",
                            annotation_text="Reject threshold (65)",
                            annotation_font_color="#ef4444")
        fig_trend.update_layout(
            height=220,
            margin=dict(t=10, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis=dict(title="Inspection #", gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(color="#7080a0")),
            yaxis=dict(title="Risk Score", range=[0, 105],
                       gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(color="#7080a0")),
            font={"family": "Inter", "color": "#a0b0d0"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Styled table ──────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>All Records</div>",
                    unsafe_allow_html=True)

        display_cols = ["id", "cell_id", "timestamp", "defect_type",
                        "risk_score", "decision", "latency_ms"]
        df_display = df[display_cols].copy()
        df_display["defect_type"] = (
            df_display["defect_type"].str.replace("_", " ").str.title()
        )
        df_display.columns = ["#", "Cell ID", "Timestamp", "Defect",
                               "Risk", "Decision", "Latency (ms)"]

        def colour_row(row):
            if row["Decision"] == "REJECT":
                return ["background-color:rgba(239,68,68,0.10)"] * len(row)
            return ["background-color:rgba(34,197,94,0.04)"] * len(row)

        styled = df_display.style.apply(colour_row, axis=1).format(
            {"Risk": "{:.0f}"}
        )
        st.dataframe(styled, use_container_width=True, height=400)

        # ── Download button ───────────────────────────────────────────────────
        csv_bytes = df_display.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download History CSV",
            data=csv_bytes,
            file_name=f"battleedge_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── TAB 3: ANALYTICS ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    df = load_history(n=999_999)

    if df.empty or len(df) < 2:
        st.info("Run at least 2 inspections to see analytics.")
    else:
        s = get_summary()

        col_l, col_r = st.columns(2)

        # ── Defect distribution donut ─────────────────────────────────────────
        with col_l:
            st.markdown("<div class='section-title'>Defect Type Distribution</div>",
                        unsafe_allow_html=True)
            defect_counts = df["defect_type"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=[x.replace("_", " ").title() for x in defect_counts.index],
                values=defect_counts.values,
                hole=0.55,
                marker_colors=[
                    "#60a5fa","#a78bfa","#f472b6",
                    "#34d399","#fbbf24","#fb923c","#f87171","#94a3b8"
                ],
                textfont=dict(color="#e2e8f0"),
            ))
            fig_donut.update_layout(
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(color="#a0b0d0", size=11)),
                font={"family": "Inter"},
                annotations=[dict(
                    text=f"<b>{len(df)}</b><br>cells",
                    font=dict(size=18, color="#e2e8f0", family="Inter"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Risk score distribution histogram ─────────────────────────────────
        with col_r:
            st.markdown("<div class='section-title'>Risk Score Distribution</div>",
                        unsafe_allow_html=True)
            fig_hist = go.Figure(go.Histogram(
                x=df["risk_score"],
                nbinsx=20,
                marker_color="#60a5fa",
                opacity=0.8,
            ))
            fig_hist.add_vline(x=65, line_dash="dash", line_color="#ef4444",
                               annotation_text="Threshold",
                               annotation_font_color="#ef4444")
            fig_hist.update_layout(
                height=280,
                margin=dict(t=10, b=30, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis=dict(title="Risk Score", gridcolor="rgba(255,255,255,0.04)",
                           tickfont=dict(color="#7080a0")),
                yaxis=dict(title="Count",     gridcolor="rgba(255,255,255,0.04)",
                           tickfont=dict(color="#7080a0")),
                font={"family": "Inter", "color": "#a0b0d0"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Risk by defect type (box plot) ────────────────────────────────────
        st.markdown("<div class='section-title'>Risk Score by Defect Type</div>",
                    unsafe_allow_html=True)
        df_plot = df.copy()
        df_plot["defect_label"] = (
            df_plot["defect_type"].str.replace("_", " ").str.title()
        )
        fig_box = px.box(
            df_plot, x="defect_label", y="risk_score",
            color="decision",
            color_discrete_map={"PASS": "#22c55e", "REJECT": "#ef4444"},
            labels={"defect_label": "Defect Type", "risk_score": "Risk Score"},
        )
        fig_box.add_hline(y=65, line_dash="dash", line_color="#ef4444")
        fig_box.update_layout(
            height=320,
            margin=dict(t=10, b=40, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis=dict(tickfont=dict(color="#a0b0d0"), gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(tickfont=dict(color="#7080a0"), gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(font=dict(color="#a0b0d0")),
            font={"family": "Inter", "color": "#a0b0d0"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # ── Thermal std vs risk scatter ───────────────────────────────────────
        st.markdown("<div class='section-title'>Thermal Uniformity vs Risk Score</div>",
                    unsafe_allow_html=True)
        df_th = df.dropna(subset=["thermal_std", "risk_score"])
        if not df_th.empty:
            fig_scatter = px.scatter(
                df_th,
                x="thermal_std", y="risk_score",
                color="decision",
                color_discrete_map={"PASS": "#22c55e", "REJECT": "#ef4444"},
                hover_data=["cell_id", "defect_type"],
                labels={"thermal_std": "Thermal Std Dev (unevenness)",
                        "risk_score":  "Risk Score"},
                trendline="ols",
            )
            fig_scatter.add_hline(y=65, line_dash="dash", line_color="#ef4444")
            fig_scatter.update_layout(
                height=300,
                margin=dict(t=10, b=40, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis=dict(tickfont=dict(color="#7080a0"),
                           gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(tickfont=dict(color="#7080a0"),
                           gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(font=dict(color="#a0b0d0")),
                font={"family": "Inter", "color": "#a0b0d0"},
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Key insights table ────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Key Insights</div>",
                    unsafe_allow_html=True)
        insights = []
        for defect, cnt in s["defect_counts"].items():
            sub = df[df["defect_type"] == defect]
            avg_risk = sub["risk_score"].mean()
            reject_r = (sub["decision"] == "REJECT").mean() * 100
            insights.append({
                "Defect Type":    defect.replace("_", " ").title(),
                "Count":          int(cnt),
                "Avg Risk":       f"{avg_risk:.1f}",
                "Rejection Rate": f"{reject_r:.0f}%",
            })
        if insights:
            ins_df = pd.DataFrame(insights).sort_values("Count", ascending=False)
            st.dataframe(ins_df, use_container_width=True, hide_index=True)
