"""
====================================================================
HOSPITAL RESOURCE OPTIMIZER - NEON CYBERPUNK EDITION
====================================================================

- Sidebar navigation
- 12 interactive Plotly charts with neon colors
- ML predictions with recommendations
- Glowing, futuristic CSS

RUN: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================================================================
# PAGE CONFIG + NEON CSS
# ====================================================================
st.set_page_config(page_title="🏥 Hospital Optimizer", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    /* ---- GLOBAL DARK BACKGROUND ---- */
    .main { background-color: #0A0A0F; }
    .stSidebar {
        background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
        border-right: 2px solid #00E5FF;
        box-shadow: 0 0 30px rgba(0, 229, 255, 0.2);
    }
    /* ---- GLOWING HEADERS ---- */
    h1, h2, h3 {
        color: #6d7de8  !important;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.7), 0 0 40px rgba(0, 229, 255, 0.3);
        letter-spacing: 1px;
    }
    /* ---- METRIC CARDS (glass + neon) ---- */
    .stMetric {
        background: rgba(20, 20, 40, 0.7) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid #00E5FF !important;
        border-radius: 16px !important;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.15), inset 0 0 15px rgba(0, 229, 255, 0.05);
        padding: 15px !important;
    }
    .stMetric label {
        color: #00E5FF !important;
        font-weight: bold !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }
    .stMetric .stMetricValue {
        color: #FFFFFF !important;
        font-size: 2.2rem !important;
        text-shadow: 0 0 20px rgba(255, 0, 229, 0.4);
    }
    /* ---- NEON BUTTONS ---- */
    .stButton button {
        background: linear-gradient(135deg, #00E5FF, #8B5CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 0 30px rgba(0, 229, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 50px rgba(0, 229, 255, 0.8) !important;
        background: linear-gradient(135deg, #00E5FF, #FF00E5) !important;
    }
    /* ---- INPUT FIELDS (dark glass) ---- */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background: rgba(20, 20, 40, 0.8) !important;
        color: white !important;
        border: 1px solid #00E5FF !important;
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);
        padding: 0.6rem !important;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
        border-color: #FF00E5 !important;
        box-shadow: 0 0 30px rgba(255, 0, 229, 0.3) !important;
    }
    /* ---- SIDEBAR RADIO BUTTONS (glowing) ---- */
    .stRadio label {
        color: #CCCCFF !important;
        font-size: 1.1rem !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        transition: all 0.2s;
    }
    .stRadio label:hover {
        background: rgba(0, 229, 255, 0.1);
        text-shadow: 0 0 15px #00E5FF;
    }
    .stRadio .st-bu {
        background-color: #00E5FF !important;
        box-shadow: 0 0 20px #00E5FF !important;
    }
    /* ---- EXPANDER (glass) ---- */
    .streamlit-expanderHeader {
        background: rgba(20, 20, 40, 0.6) !important;
        border: 1px solid #8B5CF6 !important;
        border-radius: 12px !important;
        color: #white !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
    }
    /* ---- TAB LABELS ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(20, 20, 40, 0.6) !important;
        border-radius: 20px !important;
        border: 1px solid #8B5CF6 !important;
        padding: 8px 20px !important;
        color: #CCCCFF !important;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        border-color: #00E5FF !important;
        box-shadow: 0 0 30px rgba(0, 229, 255, 0.3);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #00E5FF, #8B5CF6) !important;
        color: white !important;
        border-color: #00E5FF !important;
        box-shadow: 0 0 40px rgba(0, 229, 255, 0.5);
    }
    /* ---- SCROLLBAR ---- */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #; }
    ::-webkit-scrollbar-thumb { background: #00E5FF; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# DATA LOADING (cached)
# ====================================================================
@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("hospital_resource_dataset.csv")
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    data = data.drop_duplicates()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    encoders = {}
    for col in ["Day", "Month", "Season", "Holiday", "Weather"]:
        le = LabelEncoder()
        data[col + "_Code"] = le.fit_transform(data[col])
        encoders[col] = le
    data["Total_Patient_Load"] = (
        data["Flu_Cases"] + data["Emergency_Cases"] +
        data["Scheduled_Appointments"] + data["ICU_Patients"]
    )
    return data, encoders

@st.cache_resource
def train_models(data):
    FEATURES = ["Flu_Cases", "Emergency_Cases", "Scheduled_Appointments",
                "ICU_Patients", "Holiday_Code", "Weather_Code", "Temperature",
                "Total_Patient_Load"]
    TARGETS = ["Beds_Required", "Doctors_Required", "Nurses_Required",
               "Medicine_Required", "Equipment_Required"]
    X = data[FEATURES]
    models = {}
    results = {}
    for target in TARGETS:
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[target] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        }
        models[target] = model
    return models, results, FEATURES, TARGETS

data, encoders = load_and_prepare_data()
models, results, FEATURES, TARGETS = train_models(data)

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================
def predict_resources(user_input):
    holiday_code = encoders["Holiday"].transform([user_input["Holiday"]])[0]
    weather_code = encoders["Weather"].transform([user_input["Weather"]])[0]
    total = sum([user_input["Flu_Cases"], user_input["Emergency_Cases"],
                 user_input["Scheduled_Appointments"], user_input["ICU_Patients"]])
    one_day = pd.DataFrame([[
        user_input["Flu_Cases"], user_input["Emergency_Cases"],
        user_input["Scheduled_Appointments"], user_input["ICU_Patients"],
        holiday_code, weather_code, user_input["Temperature"], total
    ]], columns=FEATURES)
    preds = {target: int(round(models[target].predict(one_day)[0])) for target in TARGETS}
    preds["Total_Patient_Load"] = int(total)
    return preds

def get_recommendations(preds):
    suggestions = []
    if preds["Beds_Required"] > 90: suggestions.append("🛏️ BEDS      : High demand. Increase temporary beds.")
    if preds["Doctors_Required"] > 25: suggestions.append("👨‍⚕️ DOCTORS   : High demand. Call additional doctors.")
    if preds["Nurses_Required"] > 60: suggestions.append("👩‍⚕️ NURSES    : High demand. Arrange extra shift.")
    if preds["Medicine_Required"] > 500: suggestions.append("💊 MEDICINE  : Low buffer. Restock immediately.")
    if preds["Equipment_Required"] > 40: suggestions.append("🩺 EQUIPMENT : High demand. Prepare ICU equipment.")
    if not suggestions: suggestions.append("✅ Resources are sufficient. No extra action needed.")
    return suggestions

# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "🔮 Predict", "ℹ️ About"])
st.sidebar.markdown("---")


# ====================================================================
# PAGES
# ====================================================================
if page == "🏠 Home":
    st.title("🏥 Hospital Resource Optimizer")
    
    st.markdown("""
    **Predictive Resource Planning** with a futuristic twist.

    - 📊 **Dashboard** – 12 interactive neon charts
    - 🔮 **Predict** – Enter today's numbers and get instant forecasts
    - 💡 **Smart recommendations** to prevent shortages

    Built with Streamlit, Plotly & Scikit‑learn.
    """)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Days", len(data))
    with col2: st.metric("Avg Patient Load", int(data["Total_Patient_Load"].mean()))
    with col3: st.metric("Avg Beds Required", int(data["Beds_Required"].mean()))
    st.info("💡 Explore the Dashboard or Predict sections!")

elif page == "📊 Dashboard":
    st.title("📊  Dashboard")
    st.markdown("Hover, zoom, and interact with the charts.")

    tab1, tab2, tab3 = st.tabs(["📈 Trends", "📊 Distributions", "⚕️ Resources"])

    # ---- NEON COLOR PALETTE ----
    NEON_CYAN = "#00E5FF"
    NEON_MAGENTA = "#FF00E5"
    NEON_YELLOW = "#FFEA00"
    NEON_GREEN = "#00FF87"
    NEON_ORANGE = "#FF6B00"
    NEON_PURPLE = "#8B5CF6"
    NEON_PINK = "#FF0080"
    NEON_BLUE = "#0066FF"

    with tab1:
        rolling = 30
        # Combined trends
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data["Date"], y=data["Flu_Cases"].rolling(rolling).mean(),
                                  mode="lines", name="Flu", line=dict(color=NEON_CYAN, width=2.5)))
        fig1.add_trace(go.Scatter(x=data["Date"], y=data["Emergency_Cases"].rolling(rolling).mean(),
                                  mode="lines", name="Emergency", line=dict(color=NEON_MAGENTA, width=2.5)))
        fig1.add_trace(go.Scatter(x=data["Date"], y=data["Scheduled_Appointments"].rolling(rolling).mean(),
                                  mode="lines", name="Scheduled", line=dict(color=NEON_GREEN, width=2.5)))
        fig1.add_trace(go.Scatter(x=data["Date"], y=data["ICU_Patients"].rolling(rolling).mean(),
                                  mode="lines", name="ICU", line=dict(color=NEON_YELLOW, width=2.5)))
        fig1.update_layout(template="plotly_dark", height=400,
                           title="Patient Trends (30d avg)",
                           xaxis=dict(gridcolor="#333355", zerolinecolor="#333355"),
                           yaxis=dict(gridcolor="#333355", zerolinecolor="#333355"))
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # Flu
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data["Date"], y=data["Flu_Cases"], mode="lines",
                                      name="Daily", line=dict(color=NEON_CYAN, width=1), opacity=0.3))
            fig2.add_trace(go.Scatter(x=data["Date"], y=data["Flu_Cases"].rolling(rolling).mean(),
                                      mode="lines", name="30d avg", line=dict(color=NEON_MAGENTA, width=3)))
            fig2.update_layout(template="plotly_dark", height=350, title="Flu Cases")
            st.plotly_chart(fig2, use_container_width=True)

            # Emergency
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data["Date"], y=data["Emergency_Cases"], mode="lines",
                                      name="Daily", line=dict(color=NEON_ORANGE, width=1), opacity=0.3))
            fig3.add_trace(go.Scatter(x=data["Date"], y=data["Emergency_Cases"].rolling(rolling).mean(),
                                      mode="lines", name="30d avg", line=dict(color=NEON_MAGENTA, width=3)))
            fig3.update_layout(template="plotly_dark", height=350, title="Emergency Cases")
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            # ICU
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=data["Date"], y=data["ICU_Patients"], mode="lines",
                                      name="Daily", line=dict(color=NEON_PURPLE, width=1), opacity=0.3))
            fig4.add_trace(go.Scatter(x=data["Date"], y=data["ICU_Patients"].rolling(rolling).mean(),
                                      mode="lines", name="30d avg", line=dict(color=NEON_CYAN, width=3)))
            fig4.update_layout(template="plotly_dark", height=350, title="ICU Patients")
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        # Weather
        weather_counts = data["Weather"].value_counts().reset_index()
        weather_counts.columns = ["Weather", "Count"]
        fig5 = px.bar(weather_counts, x="Weather", y="Count", color="Weather",
                      color_discrete_sequence=[NEON_CYAN, NEON_MAGENTA, NEON_YELLOW, NEON_GREEN, NEON_ORANGE, NEON_PURPLE],
                      title="Weather Distribution")
        fig5.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig5, use_container_width=True)

        # Holiday
        holiday_counts = data["Holiday"].value_counts().reset_index()
        holiday_counts.columns = ["Holiday", "Count"]
        fig6 = px.bar(holiday_counts, x="Holiday", y="Count", color="Holiday",
                      color_discrete_sequence=["#555577", NEON_YELLOW],
                      title="Holiday vs Normal Days")
        fig6.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig6, use_container_width=True)

        # Correlation Heatmap
        corr_cols = ["Temperature", "Flu_Cases", "Emergency_Cases", "Scheduled_Appointments",
                     "ICU_Patients", "Total_Patient_Load", "Beds_Required", "Doctors_Required",
                     "Nurses_Required", "Medicine_Required", "Equipment_Required"]
        corr = data[corr_cols].corr()
        fig7 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#000033"], [0.5, NEON_CYAN], [1, NEON_MAGENTA]],
            zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text:.2f}", textfont={"size": 9}
        ))
        fig7.update_layout(title="Correlation Heatmap", template="plotly_dark", height=450,
                           xaxis=dict(tickangle=45, tickfont=dict(color="white")),
                           yaxis=dict(tickfont=dict(color="white")))
        st.plotly_chart(fig7, use_container_width=True)

        # Medicine Stock Histogram
        fig9 = px.histogram(data, x="Medicine_Stock", nbins=30,
                            title="Medicine Stock Distribution",
                            labels={"Medicine_Stock": "Stock (units)"},
                            color_discrete_sequence=[NEON_GREEN])
        fig9.add_vline(x=data["Medicine_Stock"].mean(), line_dash="dash",
                       line_color=NEON_MAGENTA,
                       annotation_text=f"Mean: {int(data['Medicine_Stock'].mean())}")
        fig9.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig9, use_container_width=True)

    with tab3:
        # Seasonal demand
        seasonal = data.groupby("Season")[["Beds_Required", "Doctors_Required", "Equipment_Required"]].mean().reset_index()
        seasonal["Season"] = pd.Categorical(seasonal["Season"],
                                            categories=["Winter", "Spring", "Summer", "Autumn"], ordered=True)
        seasonal = seasonal.sort_values("Season")
        fig8 = go.Figure()
        fig8.add_trace(go.Bar(x=seasonal["Season"], y=seasonal["Beds_Required"], name="Beds", marker_color=NEON_CYAN))
        fig8.add_trace(go.Bar(x=seasonal["Season"], y=seasonal["Doctors_Required"], name="Doctors", marker_color=NEON_GREEN))
        fig8.add_trace(go.Bar(x=seasonal["Season"], y=seasonal["Equipment_Required"], name="Equipment", marker_color=NEON_MAGENTA))
        fig8.update_layout(title="Seasonal Resource Demand", barmode="group", template="plotly_dark", height=400)
        st.plotly_chart(fig8, use_container_width=True)

        # Doctors available vs required
        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(x=data["Date"], y=data["Doctors_On_Duty"].rolling(rolling).mean(),
                                   mode="lines", name="Available", line=dict(color=NEON_CYAN, width=2.5)))
        fig10.add_trace(go.Scatter(x=data["Date"], y=data["Doctors_Required"].rolling(rolling).mean(),
                                   mode="lines", name="Required", line=dict(color=NEON_MAGENTA, dash="dash", width=2.5)))
        fig10.update_layout(title="Doctors: Available vs Required (30d avg)", template="plotly_dark", height=400)
        st.plotly_chart(fig10, use_container_width=True)

        # Beds available vs required
        fig11 = go.Figure()
        fig11.add_trace(go.Scatter(x=data["Date"], y=data["Available_Beds"].rolling(rolling).mean(),
                                   mode="lines", name="Available", line=dict(color=NEON_GREEN, width=2.5)))
        fig11.add_trace(go.Scatter(x=data["Date"], y=data["Beds_Required"].rolling(rolling).mean(),
                                   mode="lines", name="Required", line=dict(color=NEON_ORANGE, dash="dash", width=2.5)))
        fig11.update_layout(title="Beds: Available vs Required (30d avg)", template="plotly_dark", height=400)
        st.plotly_chart(fig11, use_container_width=True)

        # Monthly patient load
        monthly = data.groupby("Month")["Total_Patient_Load"].mean().reindex(
            ["January","February","March","April","May","June",
             "July","August","September","October","November","December"]
        ).reset_index()
        monthly.columns = ["Month", "Avg_Load"]
        fig12 = px.bar(monthly, x="Month", y="Avg_Load",
                       title="Average Patient Load by Month",
                       labels={"Avg_Load": "Average Load"},
                       color_discrete_sequence=[NEON_CYAN])
        fig12.add_hline(y=monthly["Avg_Load"].mean(), line_dash="dash",
                        line_color=NEON_MAGENTA, annotation_text="Yearly avg")
        fig12.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig12, use_container_width=True)

    st.success("💡 All charts are fully interactive – hover, zoom, and pan!")

elif page == "🔮 Predict":
    st.title("🔮  Predictor")
    st.markdown("Enter today's patient numbers to get a forecast.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            flu = st.number_input("Flu Cases", 0, 200, 45)
            emergency = st.number_input("Emergency Cases", 0, 150, 30)
            appointments = st.number_input("Scheduled Appointments", 0, 400, 120)
            icu = st.number_input("ICU Patients", 0, 100, 15)
        with col2:
            holiday = st.selectbox("Holiday?", ["No", "Yes"])
            weather = st.selectbox("Weather", list(encoders["Weather"].classes_))
            temperature = st.number_input("Temperature (°C)", -20.0, 55.0, 22.0, 0.5)

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        user_input = {
            "Flu_Cases": flu, "Emergency_Cases": emergency,
            "Scheduled_Appointments": appointments, "ICU_Patients": icu,
            "Holiday": holiday, "Weather": weather, "Temperature": temperature
        }
        preds = predict_resources(user_input)
        recs = get_recommendations(preds)

        st.subheader("📋 Prediction Results")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Total Patient Load", preds["Total_Patient_Load"])
            st.metric("Beds Required", preds["Beds_Required"])
            st.metric("Doctors Required", preds["Doctors_Required"])
        with colB:
            st.metric("Nurses Required", preds["Nurses_Required"])
            st.metric("Medicine (units)", preds["Medicine_Required"])
            st.metric("Equipment Required", preds["Equipment_Required"])

        st.subheader("💡 Recommendations")
        for rec in recs:
            st.write(rec)

        with st.expander("📥 Input Details"):
            st.json(user_input)
        with st.expander("📊 Model Performance"):
            perf_df = pd.DataFrame(results).T
            st.dataframe(perf_df.style.format("{:.3f}"))

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Hospital Resource Optimizer

    - **Dataset**: 2+ years of daily hospital records
    - **Models**: Random Forest Regressor (5 targets)
    - **Features**: Flu, Emergency, Appointments, ICU, Holiday, Weather, Temperature, Total Load
    - **Targets**: Beds, Doctors, Nurses, Medicine, Equipment

    **Tech**: Python, Streamlit, Plotly, Scikit‑learn

    **Credits**: AI Assistant – 2025
    """)
    st.markdown("### 📊 Model Performance")
    perf_df = pd.DataFrame(results).T
    st.dataframe(perf_df.style.format("{:.3f}"))
    st.caption("MAE, RMSE, R² – lower is better for MAE/RMSE, higher for R²")