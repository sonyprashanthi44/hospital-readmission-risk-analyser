from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age_mid: float = Field(65, ge=0, le=100, description="Age midpoint of band")
    race_enc: int = Field(0, ge=0, le=4, description="0 Caucasian 1 AfricanAmerican 2 Hispanic 3 Asian 4 Other")
    gender_enc: int = Field(0, ge=0, le=1, description="0 Female 1 Male")
    time_in_hospital: float = Field(4, ge=1, le=14)
    num_lab_procedures: float = Field(40, ge=0, le=150)
    num_procedures: float = Field(1, ge=0, le=10)
    num_medications: float = Field(15, ge=0, le=80)
    number_outpatient: float = Field(0, ge=0, le=50)
    number_emergency: float = Field(0, ge=0, le=50)
    number_inpatient: float = Field(0, ge=0, le=30)
    number_diagnoses: float = Field(7, ge=1, le=20)
    a1c_enc: int = Field(0, ge=0, le=3)
    glu_enc: int = Field(0, ge=0, le=3)
    change_enc: int = Field(0, ge=0, le=1)
    diabetesMed_enc: int = Field(1, ge=0, le=1)
    diag_enc: int = Field(0, ge=0, le=20)
    insulin_enc: int = Field(0, ge=0, le=1)
    metformin_enc: int = Field(0, ge=0, le=1)
    glipizide_enc: int = Field(0, ge=0, le=1)
    glyburide_enc: int = Field(0, ge=0, le=1)
    pioglitazone_enc: int = Field(0, ge=0, le=1)

    # optional labels for UI
    age_band: str | None = None
    diag_group: str | None = None


class ShapItem(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictResponse(BaseModel):
    readmission_probability: float
    risk_level: str
    threshold: float = 0.5
    top_shap: list[ShapItem]
    model_version: str = "hgb-v1"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
