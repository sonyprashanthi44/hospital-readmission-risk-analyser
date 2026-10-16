"""Download UCI Diabetes 130-US hospitals data and engineer features for 30-day readmission."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}
RACE_MAP = {"Caucasian": 0, "AfricanAmerican": 1, "Hispanic": 2, "Asian": 3, "Other": 4, "?": 4}
GENDER_MAP = {"Female": 0, "Male": 1, "Unknown/Invalid": 0}
A1C_MAP = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
GLU_MAP = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
CHANGE_MAP = {"No": 0, "Ch": 1}
MED_MAP = {"No": 0, "Yes": 1}


def icd9_group(code: str) -> str:
    c = str(code).strip()
    if not c or c in {"?", "nan"}:
        return "Unknown"
    if c.startswith("V"):
        return "V-codes"
    if c.startswith("E"):
        return "E-codes"
    try:
        n = float(c)
    except ValueError:
        return "Other"
    if 390 <= n <= 459 or n == 785:
        return "Circulatory"
    if 460 <= n <= 519 or n == 786:
        return "Respiratory"
    if 520 <= n <= 579 or n == 787:
        return "Digestive"
    if n == 250 or 250 <= n < 251:
        return "Diabetes"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


def download() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    csv_path = RAW / "diabetic_data.csv"
    if csv_path.exists():
        return csv_path
    print("Downloading UCI Diabetes 130-US hospitals dataset...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name.endswith("diabetic_data.csv"):
                csv_path.write_bytes(zf.read(name))
                break
        else:
            raise FileNotFoundError("diabetic_data.csv not found in archive")
    return csv_path


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop encounter/patient IDs and high-missing weight/payer
    drop_cols = [
        "encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty",
        "diag_2", "diag_3", "max_glu_serum",
    ]
    # Keep max_glu_serum actually - remove from drop
    drop_cols = [c for c in drop_cols if c != "max_glu_serum"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Target: readmitted <30 days
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    df["age_mid"] = df["age"].map(AGE_MAP).fillna(55)
    df["age_band"] = df["age"].astype(str)
    df["race_enc"] = df["race"].map(RACE_MAP).fillna(4)
    df["gender_enc"] = df["gender"].map(GENDER_MAP).fillna(0)
    df["a1c_enc"] = df["A1Cresult"].map(A1C_MAP).fillna(0)
    df["glu_enc"] = df["max_glu_serum"].map(GLU_MAP).fillna(0) if "max_glu_serum" in df.columns else 0
    df["change_enc"] = df["change"].map(CHANGE_MAP).fillna(0)
    df["diabetesMed_enc"] = df["diabetesMed"].map(MED_MAP).fillna(0)

    df["diag_group"] = df["diag_1"].apply(icd9_group)
    diag_cats = sorted(df["diag_group"].unique())
    diag_map = {k: i for i, k in enumerate(diag_cats)}
    df["diag_enc"] = df["diag_group"].map(diag_map)

    # Medication flags of clinical interest
    for med in ["insulin", "metformin", "glipizide", "glyburide", "pioglitazone"]:
        if med in df.columns:
            df[f"{med}_enc"] = (df[med] != "No").astype(int)
        else:
            df[f"{med}_enc"] = 0

    num_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    feature_cols = [
        "age_mid", "race_enc", "gender_enc", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "a1c_enc", "glu_enc", "change_enc",
        "diabetesMed_enc", "diag_enc", "insulin_enc", "metformin_enc", "glipizide_enc",
        "glyburide_enc", "pioglitazone_enc",
    ]
    keep = feature_cols + ["readmitted_30d", "age_band", "diag_group"]
    out = df[keep].dropna().reset_index(drop=True)
    return out, feature_cols, diag_map


def main() -> None:
    csv_path = download()
    raw = pd.read_csv(csv_path, low_memory=False)
    # Sample for faster local/docker builds while keeping class balance
    if len(raw) > 40000:
        pos = raw[raw["readmitted"] == "<30"]
        neg = raw[raw["readmitted"] != "<30"].sample(n=min(len(raw) - len(pos), 32000), random_state=42)
        raw = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

    prepared, feature_cols, diag_map = prepare(raw)
    PROC.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PROC / "readmission_features.csv", index=False)
    meta = {
        "feature_cols": feature_cols,
        "diag_map": diag_map,
        "n_rows": int(len(prepared)),
        "positive_rate": float(prepared["readmitted_30d"].mean()),
        "source": URL,
        "dataset": "UCI Diabetes 130-US hospitals for years 1999-2008",
    }
    import json
    (PROC / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Prepared {len(prepared)} rows | positive rate={meta['positive_rate']:.3f}")
    print(f"Wrote {PROC / 'readmission_features.csv'}")


if __name__ == "__main__":
    main()
