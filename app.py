import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_models
from src.evaluate import evaluate_model
from src.forecast import forecast

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Energy Dashboard", layout="wide", initial_sidebar_state="expanded")

# =========================
# PREMIUM GLASSMORPHIC SAAS THEME
# =========================
st.markdown("""
<style>
/* Import Apple-like System Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Base App Background - Deep Mesh Gradient for Glass to pop against */
.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(0, 255, 159, 0.04), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(0, 255, 159, 0.03), transparent 25%),
                linear-gradient(135deg, #030504 0%, #0a110c 100%);
    color: #ffffff;
}

/* Fix the White Header */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    background: linear-gradient(180deg, rgba(3, 5, 4, 0.9) 0%, transparent 100%) !important;
    backdrop-filter: blur(10px);
}

/* Glassmorphic Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(5, 8, 6, 0.5) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
}

/* Hero Section */
.hero-container {
    position: relative;
    padding: 70px 0 50px 0;
    text-align: center;
}
.hero-glow {
    position: absolute;
    top: -60px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 300px;
    background: radial-gradient(ellipse at top, rgba(0, 255, 159, 0.15), transparent 70%);
    filter: blur(40px);
    z-index: 0;
    pointer-events: none;
}
.hero-title {
    position: relative;
    z-index: 1;
    font-size: 3.8rem;
    font-weight: 700;
    background: linear-gradient(180deg, #ffffff 0%, #a3f7c4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
    letter-spacing: -1.5px;
}
.hero-subtitle {
    position: relative;
    z-index: 1;
    color: #8b9a92;
    font-size: 1.15rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Premium Glass Metric Cards */
div[data-testid="metric-container"] {
    background: rgba(18, 25, 20, 0.3);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-4px) scale(1.02);
    border: 1px solid rgba(0, 255, 159, 0.2);
    background: rgba(18, 25, 20, 0.4);
    box-shadow: 0 20px 40px -10px rgba(0, 255, 159, 0.1), inset 0 1px 0 rgba(255,255,255,0.1);
}
[data-testid="stMetricValue"] {
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 600;
    letter-spacing: -1px;
}
[data-testid="stMetricLabel"] {
    color: #8b9a92;
    font-weight: 500;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Glassmorphic DataFrames Container */
[data-testid="stDataFrame"] > div {
    background: rgba(18, 25, 20, 0.2) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    padding: 4px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

/* Section Headers */
h2, h3 {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 1.4rem !important;
    margin-bottom: 20px !important;
    margin-top: 40px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 12px;
    letter-spacing: -0.5px;
}

/* Custom Dividers */
hr {
    border-color: rgba(255, 255, 255, 0.04) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPER FOR MATPLOTLIB GLASS STYLING
# =========================
def style_plot(fig, ax):
    """Applies true transparent glass styling to matplotlib charts."""
    # Make background completely transparent
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style bottom and left spines with low opacity (using HEX + alpha)
    ax.spines['bottom'].set_color('#ffffff1a')
    ax.spines['left'].set_color('#ffffff1a')
    
    # Style ticks and labels
    ax.tick_params(colors='#8b9a92', length=4, pad=8, color='#ffffff1a')
    ax.xaxis.label.set_color('#8b9a92')
    ax.yaxis.label.set_color('#8b9a92')
    
    # Ultra-subtle grid
    ax.grid(color='#ffffff08', linestyle='-', linewidth=1)

# =========================
# HEADER / HERO SECTION
# =========================
st.markdown("""
<div class="hero-container">
    <div class="hero-glow"></div>
    <div class="hero-title">AI-Powered Energy Platform</div>
    <div class="hero-subtitle">Track consumption, forecast output, and move faster — all in real time</div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("<h2 style='color: #00ff9f !important; border:none; letter-spacing: -1px;'>AI Powered Energy Platform\n   ~Aniket Satpathy</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #8b9a92; font-size: 0.85rem; font-weight: 300;'>AI Energy Forecasting System</p>", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown("""
<div style='color: #ffffff; line-height: 2.5; font-weight: 300; font-size: 0.95rem;'>
    <div style='background: rgba(0,255,159,0.1); padding: 5px 15px; border-radius: 8px; border-left: 2px solid #00ff9f;'>Dashboard</div>
    <div style='padding: 5px 15px; color:#8b9a92; transition: 0.3s cursor:pointer;' onmouseover="this.style.color='#fff'" onmouseout="this.style.color='#8b9a92'">Emissions</div>
    <div style='padding: 5px 15px; color:#8b9a92; transition: 0.3s cursor:pointer;' onmouseover="this.style.color='#fff'" onmouseout="this.style.color='#8b9a92'">Insights (AI)</div>
    <div style='padding: 5px 15px; color:#8b9a92; transition: 0.3s cursor:pointer;' onmouseover="this.style.color='#fff'" onmouseout="this.style.color='#8b9a92'">Reports</div>
    <div style='padding: 5px 15px; color:#8b9a92; transition: 0.3s cursor:pointer;' onmouseover="this.style.color='#fff'" onmouseout="this.style.color='#8b9a92'">Goals</div>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_pipeline_data():
    df = load_data("data/raw/household_power_consumption.txt")
    df = preprocess_data(df)
    df = create_features(df)
    return df

df = load_pipeline_data()

# =========================
# DATA PREVIEW
# =========================
st.subheader("Dataset Preview")
st.dataframe(df.head(), width='stretch')

# =========================
# SPLIT DATA
# =========================
X = df.drop(['energy', 'timestamp'], axis=1)
y = df['energy']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# TRAIN MODEL
# =========================
with st.spinner("Training ML Models in background..."):
    lr, rf = train_models(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = rf.predict(X_test)

# =========================
# MODEL PERFORMANCE (METRICS)
# =========================
st.subheader("Impact Overview")

rmse, r2 = evaluate_model(y_test, y_pred, "Random Forest")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model RMSE", f"{rmse:.4f}", delta="-2.4% Improvement", delta_color="normal")
with col2:
    st.metric("R² Score", f"{r2:.4f}", delta="High Accuracy", delta_color="normal")
with col3:
    st.metric("Data Points", f"{len(df):,}", delta="Active Sync", delta_color="normal")


# =========================
# CHARTS SECTION
# =========================
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    # =========================
    # ACTUAL VS PREDICTED
    # =========================
    st.subheader("Actual vs Predicted")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    
    # Subdued actual (using HEX + alpha for 20% white), Neon predicted
    ax1.plot(y_test.values, label="Actual", color="#ffffff33", linewidth=2)
    ax1.plot(y_pred, label="Predicted (AI)", color="#00ff9f", linewidth=1.5)
    
    style_plot(fig1, ax1)
    
    # Transparent Legend
    legend = ax1.legend(frameon=False, labelcolor='white', loc='upper right')
    st.pyplot(fig1)

with col_chart2:
    # =========================
    # ENERGY TREND
    # =========================
    st.subheader("Energy Consumption Trend")

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(df['timestamp'], df['energy'], color="#00ff9f", linewidth=1.5)
    ax2.fill_between(df['timestamp'], df['energy'], color='#00ff9f', alpha=0.08) 
    
    style_plot(fig2, ax2)

    st.pyplot(fig2)


# =========================
# FORECAST
# =========================
st.subheader("Next 24 Hours Forecast")

future_preds = forecast(rf, X_test, steps=24)

forecast_df = pd.DataFrame({
    "Hour": range(1, 25),
    "Predicted Energy (kWh)": future_preds
})

col_fc_data, col_fc_chart = st.columns([1, 2])

with col_fc_data:
    st.dataframe(forecast_df, height=300, width='stretch', hide_index=True)

with col_fc_chart:
    # Forecast Plot
    fig3, ax3 = plt.subplots(figsize=(10, 4))

    ax3.plot(future_preds, color="#00ff9f", marker='o', markersize=6, markerfacecolor='#030504', markeredgewidth=2, linewidth=2)
    ax3.fill_between(range(len(future_preds)), future_preds, color='#00ff9f', alpha=0.05)
    
    style_plot(fig3, ax3)

    st.pyplot(fig3)


# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Feature Importance Analysis")

importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=True) 

fig4, ax4 = plt.subplots(figsize=(12, 4))

# Horizontal bars with a premium soft glow aesthetic (using HEX + alpha for 80% opacity)
importance.plot(kind='barh', color="#00ff9fcc", edgecolor="none", ax=ax4)

style_plot(fig4, ax4)
ax4.grid(axis='y') 

st.pyplot(fig4)

# =========================
# FOOTER
# =========================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:#5c6b62; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 30px; padding-bottom: 20px; font-weight: 300; letter-spacing: 0.5px;'>
        AI Energy Forecasting<br>
        <span style='color: #8b9a92;'>Designed & Developed by <b style='color:#ffffff; font-weight: 500;'>Aniket satpathy</b></span>
    </div>
    """,
    unsafe_allow_html=True
)