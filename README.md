# Hospital Readmission Predictor

I built this to answer a simple clinical operations question:

> Right after discharge, how likely is this patient to come back within 30 days — and *why*?

The app scores **30-day readmission risk** for diabetes-related hospital stays using gradient boosting, explains each prediction with **SHAP**, and wraps everything in a small production-style stack: **FastAPI** for the model API and **Streamlit** for the dashboard.

**Live demo:** [hospital-readmission-predictor-production.up.railway.app](https://hospital-readmission-predictor-production.up.railway.app)

---

## What’s inside

| Piece | What it does |
|--------|----------------|
| Model | `HistGradientBoostingClassifier` (scikit-learn) |
| Explainability | SHAP values on each `/predict` call |
| API | FastAPI — health, metrics, analytics, predict |
| UI | Streamlit — risk by age & diagnosis, patient form |
| Data | [UCI Diabetes 130-US hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| Ship | Docker Compose + Railway (`scripts/start_services.py`) |

This is a portfolio / learning system on **public de-identified data**. It is **not** a medical device and should not be used for real care decisions without proper validation.

---

## Quick look at the UI

- **Population dashboard** — hold-out style metrics, mean risk by age band, mean risk by primary diagnosis group, global SHAP importance  
- **Patient risk** — enter a discharge-like profile, get probability + Low/Moderate/High band + top feature contributions  
- **About** — data and model notes  

---

## Run it locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# model files are already under models/
python scripts/start_services.py
```

Then open:

- App → http://localhost:8501  
- API docs → http://localhost:8000/docs  

### Retrain from the UCI file (optional)

```bash
python scripts/prepare_data.py
python scripts/train_model.py
```

### Docker

```bash
docker compose up --build
```

---

## API (short version)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + `model_loaded` |
| GET | `/metrics` | ROC-AUC, F1, SHAP summary, more |
| GET | `/analytics/age` | Risk breakdown by age |
| GET | `/analytics/diagnosis` | Risk breakdown by diagnosis group |
| POST | `/predict` | One patient → probability + SHAP |

How the pieces talk to each other is written out in plain language in [`flow.md`](./flow.md).

---

## Model notes (honest)

- **Label:** `readmitted == "<30"`  
- **Imbalance:** positive class weighted via `sample_weight`  
- **Split:** stratified train/test  
- **Ballpark test metrics** (see `models/metrics.json`): ROC-AUC ~0.65, average precision ~0.41  

Readmission is a hard, noisy problem. These numbers are a solid baseline for a demo product, not a claim of clinical-grade performance.

---

## Project layout

```text
backend/     FastAPI app + SHAP-aware inference helpers
frontend/    Streamlit dashboard
scripts/     data prep, training, process starter, tests
models/      trained joblib + metrics
flow.md      end-to-end execution story
```

---

## Author

**Sony Prashanthi**  
GitHub: [sonyprashanthi44/hospital-readmission-predictor](https://github.com/sonyprashanthi44/hospital-readmission-predictor)

MIT License