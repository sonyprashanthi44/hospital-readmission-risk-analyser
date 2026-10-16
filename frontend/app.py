from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.inference import load_bundle, load_metrics, predict_local

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").strip()
FORCE_LOCAL = os.getenv("FORCE_LOCAL", "").lower() in {"1", "true", "yes"}
USE_API = (not FORCE_LOCAL) and bool(API_URL)

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size:2.1rem;font-weight:700;color:#0f766e;margin-bottom:0}
    .subtitle {color:#64748b;margin-top:0.2rem;margin-bottom:1.2rem}
    .risk-High {color:#b91c1c;font-weight:700}
    .risk-Moderate {color:#c2410c;font-weight:700}
    .risk-Low {color:#15803d;font-weight:700}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    import httpx
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{API_URL}{path}")
        r.raise_for_status()
        return r.json()


def api_predict(body: dict):
    import httpx
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{API_URL}/predict", json=body)
        r.raise_for_status()
        return r.json()


@st.cache_resource
def boot_local():
    bundle = load_bundle()
    metrics = load_metrics()
    return bundle, metrics


st.markdown('<p class="main-title">Hospital Readmission Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">30-day risk scoring · Gradient Boosting · SHAP · UCI Diabetes 130-US hospitals</p>',
    unsafe_allow_html=True,
)

mode = "API" if USE_API else "Local model"
metrics = {}
diag_map = {}
age_rows = []
diag_rows = []
try:
    if USE_API:
        health = api_get("/health")
        if not health.get("model_loaded"):
            raise RuntimeError("API up but model not loaded")
        metrics = api_get("/metrics")["metrics"]
        diag_map = api_get("/meta/diag-map")
        age_rows = api_get("/analytics/age").get("by_age", [])
        diag_rows = api_get("/analytics/diagnosis").get("by_diagnosis", [])
    else:
        raise RuntimeError("local mode")
except Exception as api_exc:
    try:
        bundle, metrics = boot_local()
        diag_map = bundle.get("diag_map", {})
        analytics = metrics.get("analytics", {})
        age_rows = analytics.get("by_age", [])
        diag_rows = analytics.get("by_diagnosis", [])
        mode = f"Local model (API fallback: {api_exc})" if USE_API else "Local model"
        USE_API = False
    except Exception as exc:
        st.error(f"Failed to start: {exc}")
        st.stop()

st.caption(f"Runtime: **{mode}** · educational demo, not a clinical device")

tab_dash, tab_predict, tab_about = st.tabs(["📊 Population dashboard", "🩺 Patient risk", "ℹ️ About"])

with tab_dash:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    c2.metric("Avg Precision", f"{metrics.get('avg_precision', 0):.3f}")
    c3.metric("F1", f"{metrics.get('f1', 0):.3f}")
    c4.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    c5.metric("Brier", f"{metrics.get('brier', 0):.3f}")

    age = pd.DataFrame(age_rows)
    diag = pd.DataFrame(diag_rows)
    left, right = st.columns(2)
    with left:
        st.subheader("Risk by age band")
        if not age.empty:
            fig = px.bar(age, x="age_band", y="mean_risk", color="readmit_rate", color_continuous_scale="Teal",
                         labels={"mean_risk": "Mean predicted risk", "readmit_rate": "Observed rate"})
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Risk by primary diagnosis group")
        if not diag.empty:
            fig2 = px.bar(diag.head(12), x="mean_risk", y="diag_group", orientation="h", color="count",
                          color_continuous_scale="Blues",
                          labels={"mean_risk": "Mean predicted risk", "diag_group": "Diagnosis"})
            fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                               yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Global SHAP importance (|mean|)")
    shap_map = metrics.get("shap_mean_abs") or {}
    if shap_map:
        sdf = pd.DataFrame({"feature": list(shap_map.keys()), "mean_abs_shap": list(shap_map.values())})
        fig3 = px.bar(sdf.head(15), x="mean_abs_shap", y="feature", orientation="h", color="mean_abs_shap",
                      color_continuous_scale="Viridis")
        fig3.update_layout(height=420, yaxis=dict(categoryorder="total ascending"),
                           margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig3, use_container_width=True)

with tab_predict:
    st.subheader("Score a discharge encounter")
    with st.form("predict_form"):
        a, b, c = st.columns(3)
        age_band = a.selectbox(
            "Age band",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            index=6,
        )
        age_mid = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
                   "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}[age_band]
        gender = b.selectbox("Gender", ["Female", "Male"])
        race = c.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])

        a, b, c, d = st.columns(4)
        time_in_hospital = a.slider("Days in hospital", 1, 14, 4)
        num_lab_procedures = b.slider("Lab procedures", 0, 120, 40)
        num_procedures = c.slider("Procedures", 0, 6, 1)
        num_medications = d.slider("Medications", 1, 50, 16)

        a, b, c, d = st.columns(4)
        number_outpatient = a.number_input("Prior outpatient", 0, 40, 0)
        number_emergency = b.number_input("Prior emergency", 0, 40, 0)
        number_inpatient = c.number_input("Prior inpatient", 0, 20, 1)
        number_diagnoses = d.slider("Diagnoses count", 1, 16, 7)

        a, b, c = st.columns(3)
        diag_group = a.selectbox("Primary diagnosis group", sorted(diag_map.keys()) if diag_map else ["Diabetes", "Circulatory", "Other"])
        a1c = b.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
        glu = c.selectbox("Max glucose serum", ["None", "Norm", ">200", ">300"])

        a, b, c, d, e, f = st.columns(6)
        change = a.checkbox("Med change", False)
        diabetes_med = b.checkbox("Diabetes med", True)
        insulin = c.checkbox("Insulin", True)
        metformin = d.checkbox("Metformin", False)
        glipizide = e.checkbox("Glipizide", False)
        glyburide = f.checkbox("Glyburide", False)
        submitted = st.form_submit_button("Estimate 30-day risk", use_container_width=True, type="primary")

    if submitted:
        body = {
            "age_mid": age_mid, "age_band": age_band,
            "race_enc": {"Caucasian": 0, "AfricanAmerican": 1, "Hispanic": 2, "Asian": 3, "Other": 4}[race],
            "gender_enc": 0 if gender == "Female" else 1,
            "time_in_hospital": time_in_hospital, "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures, "num_medications": num_medications,
            "number_outpatient": number_outpatient, "number_emergency": number_emergency,
            "number_inpatient": number_inpatient, "number_diagnoses": number_diagnoses,
            "a1c_enc": {"None": 0, "Norm": 1, ">7": 2, ">8": 3}[a1c],
            "glu_enc": {"None": 0, "Norm": 1, ">200": 2, ">300": 3}[glu],
            "change_enc": int(change), "diabetesMed_enc": int(diabetes_med),
            "diag_group": diag_group, "diag_enc": diag_map.get(diag_group, 0),
            "insulin_enc": int(insulin), "metformin_enc": int(metformin),
            "glipizide_enc": int(glipizide), "glyburide_enc": int(glyburide), "pioglitazone_enc": 0,
        }
        with st.spinner("Scoring + SHAP..."):
            result = api_predict(body) if USE_API else predict_local(body, with_shap=True)
        p = result["readmission_probability"]
        lvl = result["risk_level"]
        m1, m2, m3 = st.columns(3)
        m1.metric("30-day probability", f"{p:.1%}")
        m2.markdown(f"**Risk level:** <span class='risk-{lvl}'>{lvl}</span>", unsafe_allow_html=True)
        m3.metric("Decision threshold", f"{result.get('threshold', 0.5):.2f}")
        shap_rows = result.get("top_shap") or []
        if shap_rows:
            sdf = pd.DataFrame(shap_rows)
            fig = px.bar(sdf, x="contribution", y="feature", orientation="h", color="contribution",
                         color_continuous_scale="RdBu_r", title="Top SHAP contributions for this patient")
            fig.update_layout(height=400, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sdf, use_container_width=True, hide_index=True)

with tab_about:
    st.markdown(
        """
        ### Data
        **UCI Diabetes 130-US hospitals (1999–2008)** — de-identified inpatient encounters used in readmission research.
        ### Model
        scikit-learn `HistGradientBoostingClassifier` + SHAP explanations.
        ### Links
        - GitHub: https://github.com/sonyprashanthi44/hospital-readmission-predictor
        """
    )

