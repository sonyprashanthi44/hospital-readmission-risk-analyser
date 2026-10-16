from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import HealthResponse, MetricsResponse, PatientFeatures, PredictResponse, ShapItem

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "readmission_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

STATE: dict[str, Any] = {"bundle": None, "metrics": None, "explainer": None}


def load_artifacts() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}. Run train_model.py")
    STATE["bundle"] = joblib.load(MODEL_PATH)
    STATE["metrics"] = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    try:
        import shap

        model = STATE["bundle"]["model"]
        bg = STATE["bundle"]["background"]
        STATE["explainer"] = shap.Explainer(model.predict_proba, bg)
    except Exception:  # noqa: BLE001
        STATE["explainer"] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Hospital Readmission Predictor",
    description="30-day readmission risk with Gradient Boosting + SHAP",
    version="1.0.0",
    lifespan=lifespan,
)

# Eager load for test clients / tooling that skip lifespan hooks
try:
    load_artifacts()
except FileNotFoundError:
    pass
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def risk_level(p: float) -> str:
    if p >= 0.65:
        return "High"
    if p >= 0.40:
        return "Moderate"
    return "Low"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=STATE["bundle"] is not None)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    if not STATE["metrics"]:
        raise HTTPException(503, "Metrics not available")
    m = {k: v for k, v in STATE["metrics"].items() if k not in {"report"}}
    m["report"] = STATE["metrics"].get("report")
    return MetricsResponse(metrics=m)


@app.get("/analytics/age")
def analytics_age() -> dict[str, Any]:
    analytics = (STATE["metrics"] or {}).get("analytics", {})
    return {"by_age": analytics.get("by_age", [])}


@app.get("/analytics/diagnosis")
def analytics_diagnosis() -> dict[str, Any]:
    analytics = (STATE["metrics"] or {}).get("analytics", {})
    return {"by_diagnosis": analytics.get("by_diagnosis", [])}


@app.get("/meta/diag-map")
def diag_map() -> dict[str, int]:
    return (STATE["bundle"] or {}).get("diag_map", {})


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PatientFeatures) -> PredictResponse:
    bundle = STATE["bundle"]
    if bundle is None:
        raise HTTPException(503, "Model not loaded")

    cols = bundle["feature_cols"]
    row = {c: getattr(payload, c) for c in cols}
    if payload.diag_group and payload.diag_group in bundle["diag_map"]:
        row["diag_enc"] = bundle["diag_map"][payload.diag_group]
    X = pd.DataFrame([row], columns=cols)
    model = bundle["model"]
    proba = float(model.predict_proba(X)[0, 1])

    top: list[ShapItem] = []
    explainer = STATE["explainer"]
    if explainer is not None:
        try:
            sv = explainer(X)
            values = sv.values
            if getattr(values, "ndim", 1) == 3:
                values = values[0, :, 1]
            else:
                values = values[0]
            pairs = sorted(
                (
                    ShapItem(feature=f, value=float(X.iloc[0][f]), contribution=float(c))
                    for f, c in zip(cols, values)
                ),
                key=lambda s: abs(s.contribution),
                reverse=True,
            )
            top = pairs[:8]
        except Exception:  # noqa: BLE001
            top = []

    return PredictResponse(
        readmission_probability=round(proba, 4),
        risk_level=risk_level(proba),
        top_shap=top,
    )
