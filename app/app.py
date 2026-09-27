"""
=========================================================
AI Powered Health Risk Assessment System

Author: Sumiran
=========================================================
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.pipeline.prediction_pipeline import PredictionPipeline

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Health Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown(
    """
<style>

.block-container{
    padding-top:4rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

.main-title{
    font-size:42px;
    font-weight:700;
    color:#F4D03F;
}

.subtitle{
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}

.metric-card{
    background:#1f2937;
    padding:15px;
    border-radius:12px;
}

.footer{
    text-align:center;
    color:gray;
}


</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# LOAD PIPELINE
# -------------------------------------------------------

pipeline = PredictionPipeline()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/heart-with-pulse.png",
    width=80,
)

st.sidebar.title("AI Health Risk Assessment")

st.sidebar.markdown("---")

st.sidebar.success("✅ LightGBM Classifier")

st.sidebar.metric(
    "Accuracy",
    "96.58%"
)

st.sidebar.metric(
    "Training Samples",
    "552,070"
)

st.sidebar.metric(
    "Features",
    "13"
)

st.sidebar.metric(
    "Health Classes",
    "3"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Top Predictive Features")

st.sidebar.write("😴 Sleep Duration")

st.sidebar.write("😟 Stress Level")

st.sidebar.write("🏃 Physical Activity")

st.sidebar.write("⚖️ BMI")

st.sidebar.write("❤️ Heart Rate")

st.sidebar.markdown("---")

st.sidebar.info(
"""
This application predicts an individual's
health condition using a trained
LightGBM Machine Learning model.

Built with:

• Python

• Streamlit

• Scikit-Learn

• LightGBM
"""
)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.title("🩺 AI Powered Health Risk Assessment")

st.markdown(
    '<p class="subtitle">Predict an individual\'s health condition using Machine Learning.</p>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# INPUT SECTION
# -------------------------------------------------------

st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

# -------------------------------------------------------
# LEFT COLUMN
# -------------------------------------------------------

with col1:

    sleep_duration = st.slider(
        "😴 Sleep Duration (Hours)",
        1.0,
        12.0,
        7.0,
    )

    heart_rate = st.number_input(
        "❤️ Heart Rate",
        40,
        180,
        72,
    )

    bmi = st.number_input(
        "⚖️ BMI",
        10.0,
        50.0,
        23.5,
    )

    calorie_expenditure = st.number_input(
        "🔥 Calories Burned",
        500,
        5000,
        2200,
    )

    step_count = st.number_input(
        "👣 Step Count",
        0,
        50000,
        8000,
    )

    exercise_duration = st.slider(
        "🏋️ Exercise Duration (Minutes)",
        0,
        180,
        45,
    )

    water_intake = st.slider(
        "💧 Water Intake (Litres)",
        0.0,
        6.0,
        2.5,
    )

# -------------------------------------------------------
# RIGHT COLUMN
# -------------------------------------------------------

with col2:

    diet_type = st.selectbox(
        "🥗 Diet Type",
        [
            "balanced",
            "veg",
            "non-veg"
        ]
    )

    stress_level = st.selectbox(
        "😟 Stress Level",
        [
            "low",
            "medium",
            "high"
        ]
    )

    sleep_quality = st.selectbox(
        "🌙 Sleep Quality",
        [
            "poor",
            "average",
            "good"
        ]
    )

    physical_activity_level = st.selectbox(
        "🏃 Physical Activity",
        [
            "sedentary",
            "moderate",
            "active"
        ]
    )

    smoking_alcohol = st.selectbox(
        "🚬 Smoking / Alcohol",
        [
            "no",
            "occasional",
            "yes"
        ]
    )

    gender = st.selectbox(
        "👤 Gender",
        [
            "male",
            "female",
            "other"
        ]
    )

st.markdown("")

predict = st.button(
    "🚀 Analyze Health Condition",
    use_container_width=True
)
# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if predict:

    input_df = pd.DataFrame({

        "sleep_duration": [sleep_duration],
        "heart_rate": [heart_rate],
        "bmi": [bmi],
        "calorie_expenditure": [calorie_expenditure],
        "step_count": [step_count],
        "exercise_duration": [exercise_duration],
        "water_intake": [water_intake],
        "diet_type": [diet_type],
        "stress_level": [stress_level],
        "sleep_quality": [sleep_quality],
        "physical_activity_level": [physical_activity_level],
        "smoking_alcohol": [smoking_alcohol],
        "gender": [gender]

    })

    result = pipeline.get_prediction(input_df)

    prediction = result["prediction"]

    confidence = float(result["confidence"] * 100)

    probabilities = result["probability"]

    # -------------------------------------------------------
    # HEALTH SCORE
    # -------------------------------------------------------

    if prediction == "fit":
        health_score = min(100, round(confidence))

    elif prediction == "at-risk":
        health_score = max(40, 100 - round(confidence * 0.5))

    else:
        health_score = max(15, 100 - round(confidence))

    st.markdown("---")

    st.header("📊 Prediction Dashboard")

    # -------------------------------------------------------
    # METRIC CARDS
    # -------------------------------------------------------

    m1, m2, m3 = st.columns(3)

    with m1:

        if prediction == "fit":
            emoji = "🟢"

        elif prediction == "at-risk":
            emoji = "🟡"

        else:
            emoji = "🔴"

        st.metric(
            "Prediction",
            f"{emoji} {prediction.upper()}"
        )

    with m2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with m3:

        st.metric(
            "Health Score",
            f"{health_score}/100"
        )

    st.markdown("---")

    g1, g2 = st.columns([1, 2])

    with g1:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health_score,

                title={
                    "text": "Health Score"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "royalblue"
                    },

                    "steps": [

                        {
                            "range": [0, 40],
                            "color": "red"
                        },

                        {
                            "range": [40, 70],
                            "color": "gold"
                        },

                        {
                            "range": [70, 100],
                            "color": "green"
                        }

                    ]

                }

            )

        )

        fig.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------------------------------
    # PREDICTION PROBABILITY
    # -------------------------------------------------------

    st.subheader("📈 Prediction Probability")

    probability_df = pd.DataFrame({
        "Health Condition": pipeline.label_encoder.classes_,
        "Probability": (probabilities * 100).round(2)
    })

    # Custom colors
    color_map = {
        "fit": "#2ECC71",  # Green
        "at-risk": "#F1C40F",  # Yellow
        "unhealthy": "#E74C3C"  # Red
    }

    fig = px.bar(
        probability_df,
        x="Probability",
        y="Health Condition",
        orientation="h",
        color="Health Condition",
        color_discrete_map=color_map,
        text="Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        showlegend=False,
        xaxis_title="Probability (%)",
        yaxis_title="",
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False
        }
    )
    # -------------------------------------------------------
    # PROBABILITY TABLE
    # -------------------------------------------------------

    with st.expander("📋 View Probability Table"):

        st.dataframe(
            probability_df.rename(
                columns={
                    "Probability": "Probability (%)"
                }
            ),
            width="stretch",
            hide_index=True
        )
    # -------------------------------------------------------
    # HEALTH SUMMARY
    # -------------------------------------------------------

    st.subheader("🩺 Health Summary")

    if prediction == "fit":

        st.success("""
### Excellent Overall Health

Your lifestyle indicators suggest a healthy condition.

✅ Good sleep pattern

✅ Healthy activity level

✅ Low health risk

Continue maintaining these habits.
""")

    elif prediction == "at-risk":

        st.warning("""
### Moderate Health Risk

Some health indicators require improvement.

⚠ Higher stress

⚠ Lifestyle imbalance

⚠ Increased future risk

Small lifestyle changes can greatly improve health.
""")

    else:

        st.error("""
### High Health Risk

The model detected several risk indicators.

❗ Immediate lifestyle improvement recommended

❗ Regular health monitoring advised

❗ Consider consulting a healthcare professional.
""")

    # -------------------------------------------------------
    # PERSONALIZED RECOMMENDATIONS
    # -------------------------------------------------------

    st.subheader("💡 Personalized Recommendations")

    recommendations = []

    if sleep_duration < 7:
        recommendations.append(
            "😴 Aim for at least 7–8 hours of sleep."
        )

    if water_intake < 2:
        recommendations.append(
            "💧 Increase daily water intake."
        )

    if step_count < 8000:
        recommendations.append(
            "👣 Walk at least 8,000–10,000 steps daily."
        )

    if exercise_duration < 30:
        recommendations.append(
            "🏋 Exercise for at least 30 minutes."
        )

    if stress_level == "high":
        recommendations.append(
            "🧘 Practice meditation or stress management."
        )

    if smoking_alcohol == "yes":
        recommendations.append(
            "🚭 Reduce smoking and alcohol consumption."
        )

    if diet_type != "balanced":
        recommendations.append(
            "🥗 Include a balanced diet with fruits and vegetables."
        )

    if len(recommendations) == 0:

        st.success(
            "🎉 Great! Your current lifestyle looks well balanced."
        )

    else:

        for tip in recommendations:
            st.write("•", tip)

# =======================================================
# PREDICTION HISTORY
# =======================================================

if "history" not in st.session_state:
    st.session_state.history = []

if predict:

    st.session_state.history.insert(
        0,
        {
            "Prediction": prediction.title(),
            "Confidence": f"{confidence:.2f}%",
            "Health Score": f"{health_score}/100"
        }
    )

    st.session_state.history = st.session_state.history[:5]

# =======================================================
# HISTORY
# =======================================================

if len(st.session_state.history) > 0:

    st.markdown("---")

    st.subheader("📜 Recent Predictions")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

# =======================================================
# MODEL INFORMATION
# =======================================================

st.markdown("---")

st.subheader("🤖 Model Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Algorithm",
        "LightGBM"
    )

with c2:
    st.metric(
        "Accuracy",
        "96.58%"
    )

with c3:
    st.metric(
        "Training Samples",
        "552K"
    )

with c4:
    st.metric(
        "Features",
        "13"
    )

# =======================================================
# ABOUT THE PROJECT
# =======================================================

with st.expander("📖 About This Project"):

    st.markdown("""
### AI Powered Health Risk Assessment

This application predicts an individual's health condition
using Machine Learning.

### Pipeline

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Model Evaluation
- SHAP Explainability
- Deployment using Streamlit

### Machine Learning Algorithm

LightGBM Classifier

### Performance

- Accuracy : **96.58%**
- Weighted F1 Score : **96.43%**
- Precision : **96.57%**
- Recall : **96.58%**

This project demonstrates a complete end-to-end
Machine Learning workflow from raw data to deployment.
""")

# =======================================================
# FOOTER
# =======================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;
padding:15px;
font-size:15px;
color:gray;'>

🩺 <b>AI Powered Health Risk Assessment</b><br><br>

Developed using
<b>Python</b> |
<b>Scikit-Learn</b> |
<b>LightGBM</b> |
<b>Streamlit</b>

<br><br>

© 2026 Sumiran

</div>
""",
    unsafe_allow_html=True
)