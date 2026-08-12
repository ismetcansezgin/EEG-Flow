"""
tests/test_model_api.py
======================
Integration tests for Phase 3 Model REST API endpoints:
  - POST /api/train-model
  - POST /api/cross-validate

Tests verify:
  1.  /api/train-model default params (SVM) returns 200 with metrics.
  2.  /api/train-model for Random Forest and XGBoost.
  3.  /api/train-model raises 400 for unknown model_name.
  4.  /api/train-model raises 400 for invalid test_size limits.
  5.  /api/train-model raises 400 for invalid window_size_sec or overlap_ratio.
  6.  /api/train-model raises 400 for non-CSV files.
  7.  /api/cross-validate default params (SVM, n_splits=3) returns 200.
  8.  /api/cross-validate for Random Forest and XGBoost.
  9.  /api/cross-validate raises 400 for n_splits < 2.
  10. /api/cross-validate raises 400 if subject_id column is missing from CSV.
  11. /api/cross-validate handles n_splits capping and returns proper keys.

Run with:  python -m pytest tests/test_model_api.py -v
"""

import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
SAMPLE_CSV_PATH = "data/sample_eeg.csv"


def _post_train(csv_path: str = SAMPLE_CSV_PATH, **form_kwargs):
    """Helper: POST to /api/train-model with a CSV file and optional form params."""
    defaults = {
        "model_name":      "svm",
        "window_size_sec": "2.0",
        "overlap_ratio":   "0.5",
        "test_size":       "0.2",
        "random_state":    "42",
    }
    defaults.update(form_kwargs)

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    return client.post(
        "/api/train-model",
        files={"file": ("sample_eeg.csv", io.BytesIO(csv_bytes), "text/csv")},
        data=defaults,
    )


def _post_cross_validate(csv_path: str = SAMPLE_CSV_PATH, **form_kwargs):
    """Helper: POST to /api/cross-validate with a CSV file and optional form params."""
    defaults = {
        "model_name":      "svm",
        "window_size_sec": "2.0",
        "overlap_ratio":   "0.5",
        "n_splits":        "3",
        "random_state":    "42",
    }
    defaults.update(form_kwargs)

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    return client.post(
        "/api/cross-validate",
        files={"file": ("sample_eeg.csv", io.BytesIO(csv_bytes), "text/csv")},
        data=defaults,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: train-model default (SVM)
# ══════════════════════════════════════════════════════════════════════════════
def test_train_model_svm_default_returns_200():
    """POST /api/train-model with SVM must return 200 with valid evaluation metrics."""
    response = _post_train(model_name="svm")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert data["model_name"] == "svm"
    assert 0.0 <= data["accuracy"] <= 1.0
    assert "precision_macro" in data
    assert "recall_macro" in data
    assert "f1_macro" in data
    assert "confusion_matrix" in data
    assert "classification_report" in data
    assert "roc_auc" in data
    assert "roc_curves" in data
    assert data["train_samples"] + data["test_samples"] == 59 # 59 epochs in sample data
    assert data["feature_matrix_shape"] == [59, 144]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: train-model RF & XGBoost
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model", ["random_forest", "xgboost"])
def test_train_model_classifiers_returns_200(model):
    """POST /api/train-model with RF/XGBoost must return 200 with valid metrics."""
    response = _post_train(model_name=model)
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert data["model_name"] == model
    assert 0.0 <= data["accuracy"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: train-model unknown model
# ══════════════════════════════════════════════════════════════════════════════
def test_train_model_unknown_model_raises_400():
    """POST /api/train-model with unsupported model must return 400 Bad Request."""
    response = _post_train(model_name="invalid_model")
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: train-model test_size boundary checks
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("test_size", ["0.05", "0.6", "-0.1"])
def test_train_model_invalid_test_size_raises_400(test_size):
    """POST /api/train-model with test_size outside [0.1, 0.5] must return 400."""
    response = _post_train(test_size=test_size)
    assert response.status_code == 400
    assert "test_size" in response.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: train-model invalid parameters
# ══════════════════════════════════════════════════════════════════════════════
def test_train_model_invalid_epoch_params_raises_400():
    """POST /api/train-model with negative window size or invalid overlap ratio must return 400."""
    response_win = _post_train(window_size_sec="-1.0")
    assert response_win.status_code == 400
    
    response_over = _post_train(overlap_ratio="1.2")
    assert response_over.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: train-model non-CSV files
# ══════════════════════════════════════════════════════════════════════════════
def test_train_model_non_csv_raises_400():
    """POST /api/train-model with non-CSV file must return 400."""
    response = client.post(
        "/api/train-model",
        files={"file": ("test.png", b"fake_bytes", "image/png")},
        data={"model_name": "svm"}
    )
    assert response.status_code == 400
    assert "Only CSV files are accepted" in response.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: cross-validate default (SVM, 3 splits)
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_validate_svm_returns_200():
    """POST /api/cross-validate with SVM and n_splits=3 must return 200 with CV metrics."""
    response = _post_cross_validate(model_name="svm", n_splits="3")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert data["model_name"] == "svm"
    assert data["n_splits"] == 3
    assert 0.0 <= data["mean_accuracy"] <= 1.0
    assert "std_accuracy" in data
    assert "mean_f1_macro" in data
    assert 0.0 <= data["mean_roc_auc"] <= 1.0
    assert "std_roc_auc" in data
    assert "accumulated_confusion_matrix" in data
    assert len(data["folds"]) == 3
    assert data["n_samples"] == 59
    
    # Check fold contents
    fold = data["folds"][0]
    assert fold["fold_index"] == 1
    assert "train_subjects" in fold
    assert "test_subjects" in fold
    assert "roc_auc" in fold
    assert "roc_curves" in fold
    assert fold["train_samples"] + fold["test_samples"] == 59


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: cross-validate RF & XGBoost
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model", ["random_forest", "xgboost"])
def test_cross_validate_classifiers_returns_200(model):
    """POST /api/cross-validate with RF/XGBoost must return 200 with CV metrics."""
    response = _post_cross_validate(model_name=model, n_splits="3")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert data["model_name"] == model
    assert 0.0 <= data["mean_accuracy"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: cross-validate invalid splits
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_validate_invalid_splits_raises_400():
    """POST /api/cross-validate with n_splits < 2 must return 400."""
    response = _post_cross_validate(n_splits="1")
    assert response.status_code == 400
    assert "n_splits" in response.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: cross-validate missing subject_id raises 400
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_validate_missing_subject_id_raises_400():
    """POST /api/cross-validate with CSV missing subject_id column must return 400."""
    # Create a temp CSV without subject_id column
    df = pd.read_csv(SAMPLE_CSV_PATH)
    if "subject_id" in df.columns:
        df = df.drop(columns=["subject_id"])
        
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    
    response = client.post(
        "/api/cross-validate",
        files={"file": ("no_subject_eeg.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"model_name": "svm", "n_splits": "3"}
    )
    assert response.status_code == 400
    assert "subject_id" in response.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: cross-validate capping splits
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_validate_splits_capping():
    """POST /api/cross-validate with n_splits > unique subjects must cap splits."""
    # sample_eeg.csv only contains subject_id [1, 2, 3] (3 unique subjects)
    response = _post_cross_validate(n_splits="10")
    assert response.status_code == 200
    
    data = response.json()
    assert data["n_splits"] == 3 # Capped from 10 to 3
    assert len(data["folds"]) == 3
