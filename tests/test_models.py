"""
tests/test_models.py
====================
Unit tests for EEGFlow backend/utils/models.py

Tests verify:
  1.  build_svm()           returns a valid sklearn Pipeline.
  2.  build_random_forest() returns a valid sklearn Pipeline.
  3.  build_xgboost()       returns a valid sklearn Pipeline.
  4.  train_model()         fits a pipeline without raising errors.
  5.  evaluate_model()      returns all required metric keys.
  6.  evaluate_model()      accuracy is in [0.0, 1.0].
  7.  evaluate_model()      confusion matrix shape == (n_classes, n_classes).
  8.  train_evaluate()      works end-to-end for SVM.
  9.  train_evaluate()      works end-to-end for Random Forest.
  10. train_evaluate()      works end-to-end for XGBoost.
  11. train_evaluate()      raises ValueError for unknown model_name.
  12. train_evaluate()      handles string class labels (LabelEncoder).

Run with:  python -m pytest tests/test_models.py -v
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from backend.utils.models import (
    CLASSIFIERS,
    build_random_forest,
    build_svm,
    build_xgboost,
    evaluate_model,
    train_evaluate,
    train_model,
    cross_validate_model,
)

# ── Shared synthetic dataset ────────────────────────────────────────────────
RNG   = np.random.default_rng(seed=42)
N_SAMPLES  = 100
N_FEATURES = 20
N_CLASSES  = 2

X_SYNTH = RNG.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)
Y_SYNTH = RNG.integers(0, N_CLASSES, size=N_SAMPLES)


def _split(X=X_SYNTH, y=Y_SYNTH, test_ratio=0.2):
    """Simple deterministic split helper."""
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: build_svm returns a Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def test_build_svm_returns_pipeline():
    """build_svm() must return an sklearn Pipeline with 2 steps."""
    pipeline = build_svm()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "scaler"
    assert pipeline.steps[1][0] == "clf"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: build_random_forest returns a Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def test_build_random_forest_returns_pipeline():
    """build_random_forest() must return an sklearn Pipeline with 2 steps."""
    pipeline = build_random_forest()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "scaler"
    assert pipeline.steps[1][0] == "clf"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: build_xgboost returns a Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def test_build_xgboost_returns_pipeline():
    """build_xgboost() must return an sklearn Pipeline with 2 steps."""
    pipeline = build_xgboost()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "scaler"
    assert pipeline.steps[1][0] == "clf"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: train_model fits without error
# ══════════════════════════════════════════════════════════════════════════════
def test_train_model_fits_without_error():
    """train_model() must fit an SVM pipeline without raising exceptions."""
    pipeline = build_svm()
    X_train, X_test, y_train, y_test = _split()
    fitted = train_model(pipeline, X_train, y_train)
    assert fitted is pipeline, "train_model must return the same pipeline object"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: evaluate_model returns all required keys
# ══════════════════════════════════════════════════════════════════════════════
def test_evaluate_model_required_keys():
    """evaluate_model() must return a dict containing all required metric keys."""
    required_keys = [
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "confusion_matrix",
        "classification_report", "n_test_samples", "n_classes",
        "roc_auc", "roc_curves",
    ]
    X_train, X_test, y_train, y_test = _split()
    pipeline = train_model(build_svm(), X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    for key in required_keys:
        assert key in metrics, f"Missing key: '{key}' in evaluate_model output"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: evaluate_model accuracy is in [0.0, 1.0]
# ══════════════════════════════════════════════════════════════════════════════
def test_evaluate_model_accuracy_range():
    """evaluate_model() accuracy must be a float in [0.0, 1.0]."""
    X_train, X_test, y_train, y_test = _split()
    pipeline = train_model(build_random_forest(), X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    acc = metrics["accuracy"]
    assert isinstance(acc, float), "accuracy must be a float"
    assert 0.0 <= acc <= 1.0, f"accuracy={acc} is out of [0, 1]"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: confusion matrix shape == (n_classes, n_classes)
# ══════════════════════════════════════════════════════════════════════════════
def test_evaluate_model_confusion_matrix_shape():
    """confusion_matrix shape must be (n_classes, n_classes)."""
    X_train, X_test, y_train, y_test = _split()
    pipeline = train_model(build_svm(), X_train, y_train)
    metrics  = evaluate_model(pipeline, X_test, y_test)
    cm = metrics["confusion_matrix"]
    n  = metrics["n_classes"]
    assert len(cm) == n, f"CM rows={len(cm)} != n_classes={n}"
    for row in cm:
        assert len(row) == n, f"CM col={len(row)} != n_classes={n}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: train_evaluate end-to-end for SVM
# ══════════════════════════════════════════════════════════════════════════════
def test_train_evaluate_svm_end_to_end():
    """train_evaluate() with model_name='svm' must return valid metrics."""
    metrics = train_evaluate(X_SYNTH, Y_SYNTH, model_name="svm", random_state=42)
    assert metrics["model_name"] == "svm"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["train_samples"] + metrics["test_samples"] == N_SAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: train_evaluate end-to-end for Random Forest
# ══════════════════════════════════════════════════════════════════════════════
def test_train_evaluate_random_forest_end_to_end():
    """train_evaluate() with model_name='random_forest' must return valid metrics."""
    metrics = train_evaluate(X_SYNTH, Y_SYNTH, model_name="random_forest", random_state=42)
    assert metrics["model_name"] == "random_forest"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["train_samples"] + metrics["test_samples"] == N_SAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: train_evaluate end-to-end for XGBoost
# ══════════════════════════════════════════════════════════════════════════════
def test_train_evaluate_xgboost_end_to_end():
    """train_evaluate() with model_name='xgboost' must return valid metrics."""
    metrics = train_evaluate(X_SYNTH, Y_SYNTH, model_name="xgboost", random_state=42)
    assert metrics["model_name"] == "xgboost"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["train_samples"] + metrics["test_samples"] == N_SAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: train_evaluate raises ValueError for unknown model
# ══════════════════════════════════════════════════════════════════════════════
def test_train_evaluate_unknown_model_raises():
    """train_evaluate() must raise ValueError for an unsupported model name."""
    with pytest.raises(ValueError, match="Unknown model"):
        train_evaluate(X_SYNTH, Y_SYNTH, model_name="neural_net")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: train_evaluate handles string labels (LabelEncoder)
# ══════════════════════════════════════════════════════════════════════════════
def test_train_evaluate_string_labels():
    """train_evaluate() must correctly handle string class labels via LabelEncoder."""
    y_str = np.where(Y_SYNTH == 0, "Relaxed", "Task")
    metrics = train_evaluate(X_SYNTH, y_str, model_name="random_forest", random_state=42)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_classes"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# TEST 13: cross_validate_model end-to-end for all classifiers
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model", ["svm", "random_forest", "xgboost"])
def test_cross_validate_classifiers_end_to_end(model):
    """cross_validate_model must run GroupKFold cross-validation for all classifiers."""
    # Create 5 synthetic subjects distributed across the samples
    groups = np.repeat([f"Sub_{i}" for i in range(5)], N_SAMPLES // 5)
    
    cv_results = cross_validate_model(
        X=X_SYNTH,
        y=Y_SYNTH,
        groups=groups,
        model_name=model,
        n_splits=5,
        random_state=42,
    )
    
    assert cv_results["model_name"] == model
    assert cv_results["n_splits"] == 5
    assert 0.0 <= cv_results["mean_accuracy"] <= 1.0
    assert 0.0 <= cv_results["mean_roc_auc"] <= 1.0
    assert "std_roc_auc" in cv_results
    assert len(cv_results["folds"]) == 5
    assert cv_results["n_samples"] == N_SAMPLES
    
    # Check fold-level keys
    first_fold = cv_results["folds"][0]
    assert "roc_auc" in first_fold
    assert "roc_curves" in first_fold

    # Check shape of accumulated confusion matrix
    cm = cv_results["accumulated_confusion_matrix"]
    assert len(cm) == cv_results["n_classes"]


# ══════════════════════════════════════════════════════════════════════════════
# TEST 14: GroupKFold subject isolation (Prevent Data Leakage)
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_validate_subject_isolation():
    """Subjects in train and test folds must be completely disjoint."""
    groups = np.repeat([f"Sub_{i}" for i in range(4)], N_SAMPLES // 4)
    
    cv_results = cross_validate_model(
        X=X_SYNTH,
        y=Y_SYNTH,
        groups=groups,
        model_name="svm",
        n_splits=4,
    )
    
    for fold in cv_results["folds"]:
        train_subs = set(fold["train_subjects"])
        test_subs  = set(fold["test_subjects"])
        
        # Train and test subject sets must have ZERO intersection
        intersection = train_subs.intersection(test_subs)
        assert len(intersection) == 0, f"Subject leakage detected in fold {fold['fold_index']}: {intersection}"
        
        # Test fold must contain exactly 1 subject (since n_splits=4 and unique_groups=4)
        assert len(test_subs) == 1


# ══════════════════════════════════════════════════════════════════════════════
# TEST 15: n_splits automatic capping
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_validate_n_splits_capping():
    """n_splits must be capped at the number of unique groups if n_splits is larger."""
    # Only 3 unique subjects
    groups = np.repeat(["Sub_A", "Sub_B", "Sub_C"], [30, 30, 40])
    
    cv_results = cross_validate_model(
        X=X_SYNTH,
        y=Y_SYNTH,
        groups=groups,
        model_name="random_forest",
        n_splits=10, # Requesting 10 splits, but only 3 subjects exist
    )
    
    # Must cap to 3 splits
    assert cv_results["n_splits"] == 3
    assert len(cv_results["folds"]) == 3


# ══════════════════════════════════════════════════════════════════════════════
# TEST 16: ValueError raised for single group
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_validate_single_group_raises():
    """cross_validate_model must raise ValueError if only 1 unique group is provided."""
    groups = np.repeat(["Single_Sub"], N_SAMPLES)
    
    with pytest.raises(ValueError, match="GroupKFold requires at least 2 unique groups"):
        cross_validate_model(
            X=X_SYNTH,
            y=Y_SYNTH,
            groups=groups,
            model_name="svm",
        )

