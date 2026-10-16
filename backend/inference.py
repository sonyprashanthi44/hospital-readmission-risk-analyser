from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "readmission_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def risk_level(p: float) -> str:
    if p >= 0.65:
        return "High"
    if p >= 0.40:
        return "Moderate"
    return "Low"


def predict_local(features: dict[str, Any], with_shap: bool = True) -> dict[str, Any]:
    bundle = load_bundle()
    cols = bundle["feature_cols"]
    row = {c: features.get(c, 0) for c in cols}
    if features.get("diag_group") and features["diag_group"] in bundle.get("diag_map", {}):
        row["diag_enc"] = bundle["diag_map"][features["diag_group"]]
    X = pd.DataFrame([row], columns=cols)
    model = bundle["model"]
    proba = float(model.predict_proba(X)[0, 1])

    top: list[dict[str, Any]] = []
    if with_shap:
        try:
            import shap

            bg = bundle["background"]
            explainer = shap.Explainer(model.predict_proba, bg)
            sv = explainer(X)
            values = sv.values
            if getattr(values, "ndim", 1) == 3:
                values = values[0, :, 1]
            else:
                values = values[0]
            pairs = sorted(
                (
                    {"feature": f, "value": float(X.iloc[0][f]), "contribution": float(c)}
                    for f, c in zip(cols, values)
                ),
                key=lambda s: abs(s["contribution"]),
                reverse=True,
            )
            top = pairs[:8]
        except Exception:
            top = []

    return {
        "readmission_probability": round(proba, 4),
        "risk_level": risk_level(proba),
        "threshold": 0.5,
        "top_shap": top,
        "model_version": "hgb-v1",
    }