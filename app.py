import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_file(candidates):
    for candidate in candidates:
        path = BASE_DIR / candidate
        if path.exists():
            return path
    return None


DATA_PATH = find_file([
    "Data/battery_cycle_level_dataset_CLEAN_FINAL.csv",
    "Data/battery_cycle_level_dataset.csv",
    "data/battery_cycle_level_dataset_CLEAN_FINAL.csv",
    "data/battery_cycle_level_dataset.csv",
    "battery_cycle_level_dataset_CLEAN_FINAL.csv",
])

SOH_MODEL_PATH = find_file([
    "models/final_soh_model.joblib",
    "final_soh_model.joblib",
])

RUL_MODEL_PATH = find_file([
    
    "models/gradient_boosting_rul_model.joblib",
    "gradient_boosting_rul_model (1).joblib",
    "gradient_boosting_rul_model.joblib",
    "models/random_forest_rul_model.joblib",
    "random_forest_rul_model.joblib",
])

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0, 190, 255, 0.08), transparent 28%),
            radial-gradient(circle at 85% 20%, rgba(0, 255, 210, 0.06), transparent 25%),
            #020b18;
        color: #eaf6ff;
    }

    [data-testid="stHeader"] {
        background: rgba(2, 11, 24, 0.92);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .topbar {
        border-bottom: 1px solid rgba(54, 174, 255, 0.30);
        padding: 8px 8px 14px 8px;
        margin-bottom: 18px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border: 2px solid #18d8ff;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #18e5ff;
        font-size: 25px;
        box-shadow: 0 0 22px rgba(0, 210, 255, 0.28);
        background: rgba(0, 190, 255, 0.08);
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: #f4fbff;
    }

    .brand-subtitle {
        color: #8da9c5;
        font-size: 13px;
        margin-top: 1px;
    }

    .hero {
        min-height: 285px;
        border: 1px solid rgba(25, 197, 255, 0.48);
        border-radius: 18px;
        padding: 32px 40px;
        margin-bottom: 18px;
        background:
            radial-gradient(circle at 78% 45%, rgba(0, 218, 255, 0.18), transparent 24%),
            linear-gradient(110deg, rgba(4, 24, 47, 0.98), rgba(3, 17, 34, 0.92));
        box-shadow: inset 0 0 45px rgba(0, 167, 255, 0.06), 0 0 24px rgba(0, 150, 255, 0.05);
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 390px;
        height: 190px;
        right: 90px;
        bottom: -105px;
        border-radius: 50%;
        border: 1px solid rgba(0, 224, 255, 0.25);
        box-shadow: 0 -30px 0 rgba(0, 224, 255, 0.07), 0 -60px 0 rgba(0, 224, 255, 0.04);
    }

    .hero-badge {
        display: inline-block;
        border: 1px solid rgba(0, 211, 255, 0.65);
        border-radius: 20px;
        padding: 7px 16px;
        color: #71ddff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.7px;
        background: rgba(0, 180, 255, 0.08);
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: clamp(38px, 5vw, 62px);
        line-height: 0.98;
        font-weight: 800;
        margin: 0;
        color: #f8fcff;
        max-width: 680px;
    }

    .hero-title span {
        color: #27cfff;
        text-shadow: 0 0 24px rgba(39, 207, 255, 0.20);
    }

    .hero-text {
        color: #a9c0d6;
        font-size: 16px;
        line-height: 1.55;
        max-width: 650px;
        margin-top: 16px;
    }

    .battery-art {
        position: absolute;
        right: 8%;
        top: 47px;
        width: 310px;
        height: 155px;
        border: 2px solid rgba(39, 220, 255, 0.62);
        border-radius: 22px;
        transform: perspective(600px) rotateY(-12deg) rotateX(3deg);
        background: linear-gradient(135deg, rgba(14, 49, 73, 0.9), rgba(5, 20, 38, 0.92));
        box-shadow: 0 0 35px rgba(0, 215, 255, 0.22), inset 0 0 35px rgba(0, 195, 255, 0.09);
    }

    .battery-art::before {
        content: "";
        position: absolute;
        left: 18px;
        right: 18px;
        top: 18px;
        bottom: 18px;
        border-radius: 13px;
        background: repeating-linear-gradient(90deg, rgba(0, 255, 200, 0.18) 0 25px, transparent 25px 35px);
        box-shadow: inset 0 0 30px rgba(0, 255, 220, 0.12);
    }

    .battery-art::after {
        content: "⚡";
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 55px;
        color: #25efff;
        text-shadow: 0 0 30px rgba(37, 239, 255, 0.75);
    }

    .section-card {
        border: 1px solid rgba(70, 151, 208, 0.28);
        border-radius: 16px;
        padding: 22px;
        background: linear-gradient(145deg, rgba(7, 29, 52, 0.97), rgba(4, 18, 34, 0.97));
        min-height: 100%;
        box-shadow: inset 0 0 25px rgba(0, 150, 255, 0.025);
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #f3f8fd;
        margin-bottom: 3px;
    }

    .section-subtitle {
        color: #8faac3;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .result-card {
        border: 1px solid rgba(55, 165, 228, 0.22);
        background: rgba(7, 29, 52, 0.78);
        border-radius: 14px;
        padding: 17px;
        min-height: 118px;
    }

    .result-label {
        color: #9db5ca;
        font-size: 13px;
        margin-bottom: 9px;
    }

    .result-value {
        color: #f4fbff;
        font-size: 27px;
        font-weight: 800;
    }

    .result-icon {
        font-size: 25px;
        margin-bottom: 6px;
    }

    .gauge-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 4px 0 15px;
    }

    .gauge {
        width: 205px;
        height: 205px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow: 0 0 30px rgba(0, 202, 255, 0.15);
    }

    .gauge::before {
        content: "";
        position: absolute;
        width: 166px;
        height: 166px;
        border-radius: 50%;
        background: #061a2f;
        box-shadow: inset 0 0 25px rgba(0, 0, 0, 0.35);
    }

    .gauge-content {
        position: relative;
        z-index: 2;
        text-align: center;
    }

    .gauge-value {
        font-size: 34px;
        font-weight: 800;
        color: #f7fcff;
    }

    .gauge-label {
        color: #b3c9dc;
        font-size: 13px;
    }

    .status-pill {
        border-radius: 10px;
        padding: 11px 14px;
        text-align: center;
        font-weight: 700;
        margin-top: 14px;
    }

    .status-healthy {
        background: rgba(0, 225, 166, 0.10);
        border: 1px solid rgba(0, 225, 166, 0.40);
        color: #36efbf;
    }

    .status-aging {
        background: rgba(255, 194, 62, 0.10);
        border: 1px solid rgba(255, 194, 62, 0.40);
        color: #ffd05e;
    }

    .status-degraded {
        background: rgba(255, 91, 91, 0.10);
        border: 1px solid rgba(255, 91, 91, 0.40);
        color: #ff7777;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #1dd8ff;
        background: linear-gradient(90deg, #078ecb, #06c7dd);
        color: white;
        font-weight: 800;
        min-height: 46px;
        box-shadow: 0 0 18px rgba(0, 201, 255, 0.16);
    }

    .stButton > button:hover {
        border-color: #70efff;
        background: linear-gradient(90deg, #0aa1dc, #10d6e9);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: rgba(5, 23, 43, 0.9);
        border-color: rgba(67, 146, 203, 0.48);
        border-radius: 9px;
    }

    label, .stSelectbox label {
        color: #c4d7e8 !important;
    }

    .footer {
        text-align: center;
        color: #68839c;
        font-size: 12px;
        padding: 20px 0 4px;
    }

    @media (max-width: 900px) {
        .battery-art {
            display: none;
        }
        .hero {
            padding: 28px 25px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA + MODELS
# ============================================================

@st.cache_data

def load_data(path):
    return pd.read_csv(path)


@st.cache_resource

def load_model(path):
    return joblib.load(path)


if DATA_PATH is None or SOH_MODEL_PATH is None or RUL_MODEL_PATH is None:
    st.error("Required dataset/model files were not found.")
    st.write("Expected files:")
    st.code(
        "Data/battery_cycle_level_dataset_CLEAN_FINAL.csv\n"
        "models/final_soh_model.joblib\n"
        "models/gradient_boosting_rul_model.joblib"
    )
    st.stop()


df = load_data(str(DATA_PATH))
soh_model = load_model(str(SOH_MODEL_PATH))
rul_model = load_model(str(RUL_MODEL_PATH))

# ============================================================
# FEATURE ENGINEERING
# Must match the training script exactly.
# ============================================================

FEATURES = [
    "cycle",
    "voltage",
    "temperature",
    "capacity",
    "cycle_squared",
    "capacity_per_cycle",
    "voltage_temperature_ratio",
    "capacity_cycle_interaction",
    "voltage_temperature_interaction",
    "temperature_squared",
]


def create_features(cycle, voltage, temperature, capacity):
    data = pd.DataFrame({
        "cycle": [cycle],
        "voltage": [voltage],
        "temperature": [temperature],
        "capacity": [capacity],
    })

    data["cycle_squared"] = data["cycle"] ** 2
    data["capacity_per_cycle"] = data["capacity"] / (data["cycle"] + 1)
    data["voltage_temperature_ratio"] = data["voltage"] / (data["temperature"] + 1)
    data["capacity_cycle_interaction"] = data["capacity"] * data["cycle"]
    data["voltage_temperature_interaction"] = data["voltage"] * data["temperature"]
    data["temperature_squared"] = data["temperature"] ** 2

    return data[FEATURES]


def health_stage(soh_percent):
    if soh_percent >= 90:
        return "Healthy", "Stable"
    if soh_percent >= 80:
        return "Aging", "Degrading"
    return "Degraded", "Needs Attention"


def clamp_soh(value):
    return float(np.clip(value, 0, 1))

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-icon">⚡</div>
            <div>
                <div class="brand-title">EV Battery Intelligence</div>
                <div class="brand-subtitle">Predictive Battery Health System</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">⚙ AI-POWERED EV BATTERY ANALYTICS</div>
        <h1 class="hero-title">EV Battery<br><span>Health Prediction</span></h1>
        <div class="hero-text">
            Analyze battery condition and predict State of Health and
            Remaining Useful Life using machine learning.
        </div>
        <div class="battery-art"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# INPUT + ASSESSMENT
# ============================================================

left, right = st.columns([1.02, 1.0], gap="medium")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙ Battery Parameters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter battery operating conditions to generate an ML prediction.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        cycle = st.number_input(
            "Cycle Number",
            min_value=1.0,
            value=100.0,
            step=1.0,
            format="%.0f",
        )
        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0,
            step=0.1,
            format="%.2f",
        )

    with c2:
        voltage = st.number_input(
            "Voltage (V)",
            min_value=0.0,
            value=3.70,
            step=0.01,
            format="%.3f",
        )
        capacity = st.number_input(
            "Capacity (Ah)",
            min_value=0.0,
            value=1.80,
            step=0.01,
            format="%.3f",
        )

    predict = st.button("⚡  Predict Battery Health", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">▥ Battery Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Prediction results will appear here after entering the battery parameters.</div>',
        unsafe_allow_html=True,
    )

    if predict:
        try:
            input_data = create_features(cycle, voltage, temperature, capacity)

            raw_soh = float(soh_model.predict(input_data)[0])
            soh_percent = clamp_soh(raw_soh) * 100.0

            rul_prediction = max(0.0, float(rul_model.predict(input_data)[0]))

            stage, degradation = health_stage(soh_percent)

            if stage == "Healthy":
                gauge_color = "#20dfff"
                status_class = "status-healthy"
            elif stage == "Aging":
                gauge_color = "#ffd05e"
                status_class = "status-aging"
            else:
                gauge_color = "#ff6d73"
                status_class = "status-degraded"

            progress = min(100.0, max(0.0, soh_percent))

            st.markdown(
                f"""
                <div class="gauge-wrap">
                    <div class="gauge" style="background: conic-gradient({gauge_color} {progress}%, rgba(40,75,105,0.45) {progress}% 100%);">
                        <div class="gauge-content">
                            <div class="gauge-value">{soh_percent:.1f}%</div>
                            <div class="gauge-label">State of Health</div>
                            <div style="font-size:24px;margin-top:7px;">♥</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            r1, r2 = st.columns(2)

            with r1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-icon">💚</div>
                        <div class="result-label">State of Health</div>
                        <div class="result-value">{soh_percent:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-icon">↻</div>
                        <div class="result-label">Remaining Useful Life</div>
                        <div class="result-value">{rul_prediction:.1f} cycles</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            r3, r4 = st.columns(2)

            with r3:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-icon">▥</div>
                        <div class="result-label">Health Stage</div>
                        <div class="result-value">{stage}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r4:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-icon">🛡</div>
                        <div class="result-label">Degradation Status</div>
                        <div class="result-value">{degradation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="status-pill {status_class}">Battery Status: {stage.upper()}</div>',
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    else:
        st.markdown(
            """
            <div class="gauge-wrap">
                <div class="gauge" style="background: conic-gradient(#20dfff 0%, rgba(40,75,105,0.45) 0% 100%);">
                    <div class="gauge-content">
                        <div class="gauge-value">-- %</div>
                        <div class="gauge-label">State of Health</div>
                        <div style="font-size:24px;margin-top:7px;color:#1ddcff;">♥</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        e1, e2 = st.columns(2)
        with e1:
            st.markdown('<div class="result-card"><div class="result-label">State of Health</div><div class="result-value">-- %</div></div>', unsafe_allow_html=True)
        with e2:
            st.markdown('<div class="result-card"><div class="result-label">Remaining Useful Life</div><div class="result-value">-- cycles</div></div>', unsafe_allow_html=True)

        e3, e4 = st.columns(2)
        with e3:
            st.markdown('<div class="result-card"><div class="result-label">Health Stage</div><div class="result-value">--</div></div>', unsafe_allow_html=True)
        with e4:
            st.markdown('<div class="result-card"><div class="result-label">Degradation Status</div><div class="result-value">--</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# DEGRADATION ANALYSIS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⌁ Battery Degradation Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Track battery health degradation across charge and discharge cycles.</div>',
    unsafe_allow_html=True,
)

available_batteries = sorted(df["battery_id"].astype(str).unique()) if "battery_id" in df.columns else []

if available_batteries:
    selected_battery = st.selectbox(
        "Reference Battery Profile",
        ["All Batteries"] + available_batteries,
        index=0,
    )
else:
    selected_battery = "All Batteries"

plot_df = df.copy()

if selected_battery != "All Batteries" and "battery_id" in plot_df.columns:
    plot_df = plot_df[plot_df["battery_id"].astype(str) == selected_battery]

fig, ax = plt.subplots(figsize=(14, 4.7))
fig.patch.set_facecolor("#06172a")
ax.set_facecolor("#06172a")

if not plot_df.empty:
    if "battery_id" in plot_df.columns and selected_battery == "All Batteries":
        for battery_id, group in plot_df.groupby("battery_id"):
            ax.plot(
                group["cycle"],
                group["soh"] * 100,
                linewidth=1.5,
                alpha=0.60,
            )
    else:
        ax.plot(
            plot_df["cycle"],
            plot_df["soh"] * 100,
            linewidth=2.2,
        )

if predict:
    ax.scatter(
        [cycle],
        [soh_percent],
        s=80,
        marker="o",
        zorder=5,
        label="Current Prediction",
    )
    ax.legend(facecolor="#071d34", edgecolor="#2a577a", labelcolor="white")

ax.set_xlabel("Cycle Number", color="#9db8cf")
ax.set_ylabel("Battery Health (%)", color="#9db8cf")
ax.tick_params(colors="#91abc1")
ax.grid(True, alpha=0.13)
for spine in ax.spines.values():
    spine.set_color("#244863")

ax.set_ylim(0, 105)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">EV Battery Intelligence • Machine Learning-Based State-of-Health & Remaining Useful Life Prediction</div>',
    unsafe_allow_html=True,
)
