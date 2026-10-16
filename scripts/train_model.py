"""Train HistGradientBoostingClassifier for 30-day readmission; save model, metrics, SHAP summary."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def main() -> None:
    csv = PROC / "readmission_features.csv"
    meta_path = PROC / "meta.json"
    if not csv.exists():
        raise SystemExit("Run scripts/prepare_data.py first")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    df = pd.read_csv(csv)

    X = df[feature_cols]
    y = df["readmitted_30d"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class weight via sample_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    w_pos = neg / max(pos, 1)
    sample_weight = np.where(y_train == 1, w_pos, 1.0)

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "brier": float(brier_score_loss(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
        "report": classification_report(y_test, pred, output_dict=True),
        "feature_cols": feature_cols,
        "diag_map": meta["diag_map"],
        "dataset": meta["dataset"],
    }

    # Analytics aggregates on full prepared set using model scores
    df_score = df.copy()
    df_score["risk"] = model.predict_proba(df[feature_cols])[:, 1]
    age_stats = (
        df_score.groupby("age_band")
        .agg(count=("risk", "size"), mean_risk=("risk", "mean"), readmit_rate=("readmitted_30d", "mean"))
        .reset_index()
        .sort_values("age_band")
        .to_dict(orient="records")
    )
    diag_stats = (
        df_score.groupby("diag_group")
        .agg(count=("risk", "size"), mean_risk=("risk", "mean"), readmit_rate=("readmitted_30d", "mean"))
        .reset_index()
        .sort_values("mean_risk", ascending=False)
        .to_dict(orient="records")
    )
    metrics["analytics"] = {"by_age": age_stats, "by_diagnosis": diag_stats}

    # SHAP on a sample for background + mean |shap|
    try:
        import shap

        bg = X_train.sample(n=min(200, len(X_train)), random_state=42)
        explainer = shap.Explainer(model.predict_proba, bg)
        sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
        sv = explainer(sample)
        # class 1 contributions
        values = sv.values
        if getattr(values, "ndim", 1) == 3:
            values = values[:, :, 1]
        mean_abs = np.abs(values).mean(axis=0)
        metrics["shap_mean_abs"] = {
            f: float(v) for f, v in sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1])
        }
    except Exception as exc:  # noqa: BLE001
        metrics["shap_mean_abs"] = {}
        metrics["shap_error"] = str(exc)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "diag_map": meta["diag_map"],
            "background": X_train.sample(n=min(100, len(X_train)), random_state=0),
        },
        MODELS / "readmission_model.joblib",
    )
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("roc_auc", "avg_precision", "f1", "accuracy", "brier")}, indent=2))
    print(f"Saved {MODELS / 'readmission_model.joblib'}")


if __name__ == "__main__":
    main()
