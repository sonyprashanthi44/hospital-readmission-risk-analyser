import json
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]

def test_artifacts_exist():
    assert (ROOT / 'models' / 'readmission_model.joblib').exists()
    assert (ROOT / 'models' / 'metrics.json').exists()

def test_metrics_reasonable():
    m = json.loads((ROOT / 'models' / 'metrics.json').read_text(encoding='utf-8'))
    assert m['roc_auc'] >= 0.60
    assert 0 < m['avg_precision'] <= 1

def test_api_predict():
    from backend.main import app
    with TestClient(app) as client:
        h = client.get('/health')
        assert h.status_code == 200 and h.json()['model_loaded'] is True
        body = {
            'age_mid': 75, 'race_enc': 0, 'gender_enc': 1, 'time_in_hospital': 6,
            'num_lab_procedures': 50, 'num_procedures': 2, 'num_medications': 20,
            'number_outpatient': 0, 'number_emergency': 1, 'number_inpatient': 2,
            'number_diagnoses': 9, 'a1c_enc': 2, 'glu_enc': 0, 'change_enc': 1,
            'diabetesMed_enc': 1, 'diag_enc': 0, 'insulin_enc': 1, 'metformin_enc': 0,
            'glipizide_enc': 0, 'glyburide_enc': 0, 'pioglitazone_enc': 0,
        }
        r = client.post('/predict', json=body)
        assert r.status_code == 200
        data = r.json()
        assert 0.0 <= data['readmission_probability'] <= 1.0
        assert data['risk_level'] in {'Low', 'Moderate', 'High'}
        a = client.get('/analytics/age')
        assert a.status_code == 200 and isinstance(a.json()['by_age'], list)
