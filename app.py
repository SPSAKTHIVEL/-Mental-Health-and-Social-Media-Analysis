# app.py — Professional Mental Health Analytics Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Analytics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Industrial Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
.stApp { background-color: #0d1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.blue::before  { background: linear-gradient(90deg, #1f6feb, #388bfd); }
.kpi-card.green::before { background: linear-gradient(90deg, #238636, #3fb950); }
.kpi-card.orange::before{ background: linear-gradient(90deg, #9e6a03, #d29922); }
.kpi-card.red::before   { background: linear-gradient(90deg, #b62324, #f85149); }
.kpi-card.purple::before{ background: linear-gradient(90deg, #6e40c9, #a371f7); }

.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b949e;
}
.kpi-delta {
    font-size: 0.8rem;
    margin-top: 6px;
    font-weight: 600;
}
.kpi-delta.up   { color: #3fb950; }
.kpi-delta.down { color: #f85149; }

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e6edf3;
    border-left: 4px solid #388bfd;
    padding-left: 12px;
    margin: 28px 0 16px 0;
    letter-spacing: 0.3px;
}

/* Prediction panel */
.pred-normal {
    background: linear-gradient(135deg, #0f2419, #1a3a24);
    border: 1px solid #3fb950;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-stressed {
    background: linear-gradient(135deg, #1c1a08, #33290a);
    border: 1px solid #d29922;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-atrisk {
    background: linear-gradient(135deg, #200d0d, #3a1212);
    border: 1px solid #f85149;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-title {
    font-size: 1rem;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}
.pred-value {
    font-size: 2.8rem;
    font-weight: 800;
}
.pred-emoji { font-size: 2.2rem; margin-bottom: 8px; }
.pred-desc  { font-size: 0.9rem; color: #8b949e; margin-top: 10px; }

/* Plotly charts dark background override */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.9rem;
}

/* Divider */
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("mental_health_model.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("mental_health_social_media_dataset.csv")

model = load_model()
df    = load_data()

# ─────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────
mental_state_map  = {0: "At Risk", 1: "Stressed", 2: "Normal"}
mental_state_emoji= {0: "🔴", 1: "🟡", 2: "🟢"}
mental_state_desc = {
    0: "High risk indicators detected. Immediate attention recommended.",
    1: "Moderate stress indicators. Consider lifestyle adjustments.",
    2: "Mental state appears healthy. Keep up the balance!"
}
mental_state_color = {0: "#f85149", 1: "#d29922", 2: "#3fb950"}

# ─────────────────────────────────────────────
# SIDEBAR — INPUT PANEL
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2rem;'>🧠</div>
        <div style='font-size:1.1rem; font-weight:800; color:#e6edf3;'>Mental Health AI</div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:4px;'>Analytics Dashboard v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👤 Subject Profile")

    age    = st.slider("Age", 10, 80, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("### 📱 Social Media")
    platform            = st.selectbox("Platform", df["platform"].unique())
    daily_screen_time   = st.slider("Daily Screen Time (min)", 0, 600, 300)
    social_media_time   = st.slider("Social Media Time (min)", 0, 480, 150)
    negative_interactions = st.slider("Negative Interactions", 0, 50, 1)
    positive_interactions = st.slider("Positive Interactions", 0, 50, 2)

    st.markdown("### 🏃 Lifestyle")
    sleep_hours       = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
    physical_activity = st.slider("Physical Activity (min)", 0, 180, 30)

    st.markdown("### 🧪 Mental Indicators")
    anxiety_level = st.slider("Anxiety Level", 1, 10, 5)
    stress_level  = st.slider("Stress Level",  1, 10, 5)
    mood_level    = st.slider("Mood Level",    1, 10, 5)

    st.markdown("---")
    predict_btn = st.button("⚡ RUN PREDICTION", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# ENCODE INPUT
# ─────────────────────────────────────────────
gender_map   = {"Male": 1, "Female": 0}
platform_map = {v: k for k, v in enumerate(df["platform"].unique())}

input_data = pd.DataFrame({
    "age":                        [age],
    "gender":                     [gender_map[gender]],
    "platform":                   [platform_map[platform]],
    "daily_screen_time_min":      [daily_screen_time],
    "social_media_time_min":      [social_media_time],
    "negative_interactions_count":[negative_interactions],
    "positive_interactions_count":[positive_interactions],
    "sleep_hours":                [sleep_hours],
    "physical_activity_min":      [physical_activity],
    "anxiety_level":              [anxiety_level],
    "stress_level":               [stress_level],
    "mood_level":                 [mood_level],
})

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
prediction      = None
pred_proba      = None

if predict_btn:
    prediction  = model.predict(input_data)[0]
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(input_data)[0]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
    <div style='font-size:0.8rem; color:#8b949e; font-weight:600; letter-spacing:2px; text-transform:uppercase;'>DATA SCIENCE — MENTAL HEALTH AI</div>
    <div style='font-size:2.4rem; font-weight:800; color:#e6edf3; line-height:1.2; margin-top:4px;'>
        🧠 Social Media Mental Health <span style='color:#388bfd;'>Analytics</span>
    </div>
    <div style='font-size:0.95rem; color:#8b949e; margin-top:6px;'>
        Real-time prediction &amp; population-level insights powered by Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
total    = len(df)
at_risk  = int((df["mental_state"] == "At_Risk").sum()) if "At_Risk" in df["mental_state"].values else int((df["mental_state"] == 0).sum())
stressed = int((df["mental_state"] == "Stressed").sum()) if "Stressed" in df["mental_state"].values else int((df["mental_state"] == 1).sum())
normal   = total - at_risk - stressed
avg_stress = round(df["stress_level"].mean(), 1) if "stress_level" in df.columns else 0
avg_sleep  = round(df["sleep_hours"].mean(), 1)  if "sleep_hours" in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class='kpi-card blue'>
        <div class='kpi-value' style='color:#388bfd;'>{total:,}</div>
        <div class='kpi-label'>Total Records</div>
        <div class='kpi-delta up'>▲ Dataset Size</div>
    </div>""", unsafe_allow_html=True)

with c2:
    pct = round(normal/total*100, 1) if total else 0
    st.markdown(f"""
    <div class='kpi-card green'>
        <div class='kpi-value' style='color:#3fb950;'>{normal:,}</div>
        <div class='kpi-label'>Normal</div>
        <div class='kpi-delta up'>▲ {pct}% of population</div>
    </div>""", unsafe_allow_html=True)

with c3:
    pct = round(stressed/total*100, 1) if total else 0
    st.markdown(f"""
    <div class='kpi-card orange'>
        <div class='kpi-value' style='color:#d29922;'>{stressed:,}</div>
        <div class='kpi-label'>Stressed</div>
        <div class='kpi-delta down'>▼ {pct}% of population</div>
    </div>""", unsafe_allow_html=True)

with c4:
    pct = round(at_risk/total*100, 1) if total else 0
    st.markdown(f"""
    <div class='kpi-card red'>
        <div class='kpi-value' style='color:#f85149;'>{at_risk:,}</div>
        <div class='kpi-label'>At Risk</div>
        <div class='kpi-delta down'>▼ {pct}% of population</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class='kpi-card purple'>
        <div class='kpi-value' style='color:#a371f7;'>{avg_stress}</div>
        <div class='kpi-label'>Avg Stress Score</div>
        <div class='kpi-delta'>😴 Avg Sleep: {avg_sleep}h</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Prediction Center", "📊 Live Analytics", "🔬 Feature Intelligence", "📁 Data Explorer"])

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", family="Inter"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True),
)

# ══════════════════════════════════════════════
# TAB 1 — PREDICTION CENTER
# ══════════════════════════════════════════════
with tab1:

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.markdown("<div class='section-header'>📋 Input Summary</div>", unsafe_allow_html=True)

        # Radar chart of user inputs (normalised 0-1)
        radar_categories = ["Sleep", "Mood", "Physical\nActivity", "Positive\nInteractions", "Stress\n(inv)", "Anxiety\n(inv)"]
        radar_values_raw = [
            sleep_hours / 12,
            mood_level / 10,
            physical_activity / 180,
            positive_interactions / 50,
            (10 - stress_level) / 10,
            (10 - anxiety_level) / 10,
        ]
        radar_values = radar_values_raw + [radar_values_raw[0]]
        radar_theta  = radar_categories + [radar_categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values, theta=radar_theta, fill="toself",
            name="Subject",
            line=dict(color="#388bfd", width=2),
            fillcolor="rgba(56,139,253,0.15)"
        ))
        fig_radar.update_layout(
            **PLOT_LAYOUT,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#30363d", tickfont=dict(size=9, color="#8b949e"), showticklabels=False),
                angularaxis=dict(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(size=11, color="#c9d1d9"))
            ),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Input data table
        display_df = pd.DataFrame({
            "Feature": ["Age","Gender","Platform","Screen Time (min)","Social Media (min)",
                        "Neg. Interactions","Pos. Interactions","Sleep (h)",
                        "Physical Activity (min)","Anxiety","Stress","Mood"],
            "Value": [age, gender, platform, daily_screen_time, social_media_time,
                      negative_interactions, positive_interactions, sleep_hours,
                      physical_activity, anxiety_level, stress_level, mood_level]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with right_col:
        st.markdown("<div class='section-header'>🎯 Prediction Result</div>", unsafe_allow_html=True)

        if prediction is not None:
            label = mental_state_map[prediction]
            emoji = mental_state_emoji[prediction]
            desc  = mental_state_desc[prediction]
            color = mental_state_color[prediction]
            css_cls = {0:"pred-atrisk", 1:"pred-stressed", 2:"pred-normal"}[prediction]

            st.markdown(f"""
            <div class='{css_cls}'>
                <div class='pred-emoji'>{emoji}</div>
                <div class='pred-title'>Predicted Mental State</div>
                <div class='pred-value' style='color:{color};'>{label}</div>
                <div class='pred-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart for confidence
            if pred_proba is not None:
                confidence = round(pred_proba[prediction] * 100, 1)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence,
                    title=dict(text="Model Confidence (%)", font=dict(color="#8b949e", size=14)),
                    number=dict(font=dict(size=40, color=color), suffix="%"),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#8b949e", tickfont=dict(color="#8b949e")),
                        bar=dict(color=color, thickness=0.25),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 50],  color="rgba(248,81,73,0.15)"),
                            dict(range=[50, 75], color="rgba(210,153,34,0.15)"),
                            dict(range=[75, 100],color="rgba(63,185,80,0.15)"),
                        ],
                        threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=confidence)
                    ),
                ))
                fig_gauge.update_layout(**{**PLOT_LAYOUT, "height": 220})
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Probability bars for all classes
                st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
                for i, (cls, prob) in enumerate(zip(mental_state_map.values(), pred_proba)):
                    clr = mental_state_color[i]
                    pct = round(prob * 100, 1)
                    st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <div style='display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:4px;'>
                            <span style='color:#c9d1d9; font-weight:600;'>{mental_state_emoji[i]} {cls}</span>
                            <span style='color:{clr}; font-weight:700;'>{pct}%</span>
                        </div>
                        <div style='background:#21262d; border-radius:99px; height:8px; overflow:hidden;'>
                            <div style='width:{pct}%; height:100%; background:{clr}; border-radius:99px;
                                        transition: width 0.6s ease;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Download button
            dl_df = input_data.copy()
            dl_df["predicted_mental_state"] = label
            st.download_button(
                label="⬇️ Download Prediction CSV",
                data=dl_df.to_csv(index=False),
                file_name="mental_health_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style='background:#161b22; border:2px dashed #30363d; border-radius:16px;
                        padding:60px 30px; text-align:center; margin-top:20px;'>
                <div style='font-size:3rem; margin-bottom:12px;'>⚡</div>
                <div style='font-size:1.3rem; font-weight:700; color:#e6edf3;'>Ready to Predict</div>
                <div style='font-size:0.9rem; color:#8b949e; margin-top:8px;'>
                    Fill in the details in the sidebar and click<br>
                    <strong style='color:#388bfd;'>RUN PREDICTION</strong> to see your results.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — LIVE ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📊 Population Analytics</div>", unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        # Mental State Distribution — Donut
        ms_counts = df["mental_state"].value_counts().reset_index()
        ms_counts.columns = ["Mental State", "Count"]
        fig_pie = px.pie(
            ms_counts, names="Mental State", values="Count",
            hole=0.6,
            color="Mental State",
            color_discrete_map={"Normal":"#3fb950","Stressed":"#d29922","At_Risk":"#f85149"},
            title="Mental State Distribution"
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                              textfont=dict(color="#e6edf3"),
                              marker=dict(line=dict(color="#0d1117", width=2)))
        fig_pie.update_layout(**PLOT_LAYOUT, height=340,
                              title_font=dict(size=15, color="#e6edf3"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_c2:
        # Platform breakdown
        if "platform" in df.columns:
            plat_ms = df.groupby(["platform","mental_state"]).size().reset_index(name="count")
            fig_bar = px.bar(
                plat_ms, x="platform", y="count", color="mental_state",
                barmode="stack",
                title="Mental State by Platform",
                color_discrete_map={"Normal":"#3fb950","Stressed":"#d29922","At_Risk":"#f85149"},
                labels={"count":"Count","platform":"Platform","mental_state":"Mental State"}
            )
            fig_bar.update_layout(**PLOT_LAYOUT, height=340, showlegend=True,
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
                                  title_font=dict(size=15, color="#e6edf3"))
            st.plotly_chart(fig_bar, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        # Sleep vs Stress scatter
        if "sleep_hours" in df.columns and "stress_level" in df.columns:
            fig_scatter = px.scatter(
                df, x="sleep_hours", y="stress_level", color="mental_state",
                opacity=0.7, title="Sleep Hours vs Stress Level",
                color_discrete_map={"Normal":"#3fb950","Stressed":"#d29922","At_Risk":"#f85149"},
                labels={"sleep_hours":"Sleep Hours","stress_level":"Stress Level","mental_state":"Mental State"},
                trendline="ols"
            )
            fig_scatter.update_layout(**PLOT_LAYOUT, height=320,
                                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
                                      title_font=dict(size=15, color="#e6edf3"))
            st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_c2:
        # Screen time vs anxiety
        if "daily_screen_time_min" in df.columns and "anxiety_level" in df.columns:
            fig_scatter2 = px.scatter(
                df, x="daily_screen_time_min", y="anxiety_level", color="mental_state",
                opacity=0.7, title="Screen Time vs Anxiety Level",
                color_discrete_map={"Normal":"#3fb950","Stressed":"#d29922","At_Risk":"#f85149"},
                labels={"daily_screen_time_min":"Screen Time (min)","anxiety_level":"Anxiety Level"},
            )
            fig_scatter2.update_layout(**PLOT_LAYOUT, height=320,
                                       legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
                                       title_font=dict(size=15, color="#e6edf3"))
            st.plotly_chart(fig_scatter2, use_container_width=True)

    # Box plots
    st.markdown("<div class='section-header'>📦 Distribution by Mental State</div>", unsafe_allow_html=True)
    metric_col = st.selectbox(
        "Select metric to compare",
        [c for c in ["stress_level","anxiety_level","mood_level","sleep_hours","daily_screen_time_min","physical_activity_min"] if c in df.columns]
    )
    fig_box = px.box(
        df, x="mental_state", y=metric_col, color="mental_state",
        color_discrete_map={"Normal":"#3fb950","Stressed":"#d29922","At_Risk":"#f85149"},
        title=f"{metric_col.replace('_',' ').title()} by Mental State",
        points="outliers",
    )
    fig_box.update_layout(**PLOT_LAYOUT, height=350, showlegend=False,
                          title_font=dict(size=15, color="#e6edf3"))
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — FEATURE INTELLIGENCE
# ══════════════════════════════════════════════
with tab3:
    col_fi, col_hm = st.columns([1, 1.2], gap="large")

    with col_fi:
        st.markdown("<div class='section-header'>🏆 Feature Importance</div>", unsafe_allow_html=True)
        if hasattr(model, "feature_importances_"):
            feat_names = input_data.columns.tolist()
            importances = model.feature_importances_
            feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True)

            fig_fi = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                title="Feature Importance (Model)",
                color="Importance",
                color_continuous_scale=[[0,"#1f6feb"],[0.5,"#388bfd"],[1,"#a371f7"]],
            )
            fig_fi.update_layout(**PLOT_LAYOUT, height=400,
                                 coloraxis_showscale=False,
                                 title_font=dict(size=15, color="#e6edf3"))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Model does not expose feature importances.")

    with col_hm:
        st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
        num_df = df.select_dtypes(include=np.number)
        corr   = num_df.corr()
        fig_hm = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Matrix",
            text_auto=".2f",
        )
        fig_hm.update_layout(**PLOT_LAYOUT, height=440,
                             title_font=dict(size=15, color="#e6edf3"),
                             coloraxis_colorbar=dict(tickfont=dict(color="#8b949e")))
        fig_hm.update_traces(textfont=dict(size=9, color="#e6edf3"))
        st.plotly_chart(fig_hm, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>📁 Dataset Explorer</div>", unsafe_allow_html=True)

    c_rows, c_cols, c_miss = st.columns(3)
    with c_rows: st.metric("Rows", f"{df.shape[0]:,}")
    with c_cols: st.metric("Columns", df.shape[1])
    with c_miss: st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df, use_container_width=True, height=380)

    st.markdown("<div class='section-header'>📈 Descriptive Statistics</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.download_button(
        label="⬇️ Download Full Dataset",
        data=df.to_csv(index=False),
        file_name="mental_health_dataset.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#484f58; font-size:0.78rem; padding:10px 0;'>
    🧠 Mental Health Analytics Dashboard &nbsp;|&nbsp; Powered by Machine Learning &nbsp;|&nbsp; For educational purposes only
</div>
""", unsafe_allow_html=True)