# How this project actually runs

This note is the “story of the code” — not a textbook chapter. If you clone the repo and wonder *what runs first*, start here.

---

## Big picture

There are two phases:

1. **Train offline** (once, or when you change features)  
2. **Serve online** (API + dashboard)

```text
UCI data  →  prepare_data.py  →  features CSV
                                      ↓
                              train_model.py
                                      ↓
                     models/readmission_model.joblib
                     models/metrics.json
                                      ↓
              start_services.py  →  FastAPI + Streamlit
                                      ↓
                         browser dashboard /predict
```

---

## Phase 1 — Data prep

**Script:** `scripts/prepare_data.py`

1. Downloads the UCI Diabetes 130-US hospitals zip (if `data/raw/diabetic_data.csv` is missing).  
2. Builds a 30-day label: readmitted as `<30` → 1, otherwise 0.  
3. Turns messy clinical fields into model-ready columns:
   - age band → midpoint (`age_mid`)
   - race / gender → small integer codes
   - primary ICD-9 → a diagnosis *group* (Circulatory, Respiratory, Diabetes, …) then `diag_enc`
   - counts like length of stay, labs, meds, prior inpatient/ER visits
   - A1C / glucose / diabetes meds / a few drug flags  
4. Writes:
   - `data/processed/readmission_features.csv`
   - `data/processed/meta.json` (feature list, diagnosis map, basic stats)

Raw and processed CSTs stay local (gitignored). The trained model in `models/` is what production loads.

---

## Phase 2 — Training

**Script:** `scripts/train_model.py`

1. Loads the feature table and `meta.json`.  
2. Stratified train/test split.  
3. Fits `HistGradientBoostingClassifier` with higher weight on the rarer readmitted class.  
4. Scores the test set (ROC-AUC, average precision, F1, accuracy, Brier).  
5. Runs the model over the full prepared table for **analytics** (mean risk by age band and by diagnosis group).  
6. Computes a global SHAP summary on a sample (mean absolute contribution per feature).  
7. Saves:
   - `models/readmission_model.joblib` — model, feature columns, diagnosis map, small background set for SHAP  
   - `models/metrics.json` — metrics + analytics + SHAP summary  

If SHAP fails for any reason, training still finishes; the API just may return empty explanations until SHAP works.

---

## Phase 3 — Serving

**Entry point:** `scripts/start_services.py`

This is intentional. A pure bash start script caused pain on Windows → Linux deploys (line endings, 502s). The Python starter is boring and reliable.

On boot it:

1. Sets `PYTHONPATH` to the repo root.  
2. Starts **Uvicorn** → `backend.main:app` on `127.0.0.1:8000` (or `API_HOST` / `API_PORT`).  
3. Waits a couple of seconds so the model can load.  
4. Starts **Streamlit** on `$PORT` (local default 8501), with `API_URL` pointing at the API.

### What FastAPI does on startup (`backend/main.py`)

- Loads `models/readmission_model.joblib`  
- Loads `models/metrics.json`  
- Builds a SHAP `Explainer` around `predict_proba` using the stored background sample  

### Main routes

- `GET /health` — is the model in memory?  
- `GET /metrics` — numbers for the dashboard header  
- `GET /analytics/age` / `GET /analytics/diagnosis` — chart data  
- `GET /meta/diag-map` — labels for the form  
- `POST /predict` — one patient JSON in → probability, risk band, top SHAP rows out  

Pydantic schemas in `backend/schemas.py` validate the payload before it ever touches the model.

### What happens on one prediction

```text
UI form
  → POST /predict
  → validate PatientFeatures
  → 1-row DataFrame in training column order
  → model.predict_proba → p
  → risk band: Low / Moderate / High
  → SHAP values for class 1 → top features by |contribution|
  → JSON back to Streamlit → metrics + bar chart
```

### Streamlit (`frontend/app.py`)

- Prefers the API when `API_URL` is set (default in full-stack mode).  
- If the API is down, it falls back to `backend.inference.predict_local` so the UI is not completely dead.  
- Dashboard tabs are just HTTP reads + Plotly. No training happens in the browser path.

---

## Phase 4 — Cloud (Railway)

Production URL:

https://hospital-readmission-predictor-production.up.railway.app

Roughly the same as local:

- Container/build installs `requirements.txt`  
- Start command runs `python scripts/start_services.py`  
- Railway injects `PORT` for the public URL  
- Streamlit listens on that port  
- FastAPI stays internal on 8000; the UI calls `http://127.0.0.1:8000`  

So from the outside you mostly see the Streamlit app. The API is still there, on the same machine, behind the UI.

Docker Compose is the same idea for a laptop: one service, both processes, ports 8501 (and 8000 if you map it).

---

## Mental model in one sentence

**Prep features → train once → ship the joblib → API scores and explains → Streamlit is the face.**

That’s the whole product loop.