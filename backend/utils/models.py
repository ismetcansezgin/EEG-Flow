"""
backend/utils/models.py
=======================
EEGFlow — Machine Learning Classification Module (Phase 3)

Provides three EEG mental state classifiers:
  - Support Vector Machine (SVM)  with RBF kernel
  - Random Forest (RF)            with entropy criterion
  - XGBoost (XGB)                 with softmax objective

Each classifier is wrapped in a consistent interface:
  build_<model>()   → returns an unfitted sklearn-compatible Pipeline
  train_model()     → fits a pipeline on (X_train, y_train)
  evaluate_model()  → returns a MetricsDict on (X_test, y_test)
  train_evaluate()  → convenience wrapper: splits, trains and evaluates

Feature scaling is handled inside each Pipeline via StandardScaler so that
raw feature matrices can be passed directly without external preprocessing.

Usage:
    from backend.utils.models import train_evaluate, CLASSIFIERS

    results = train_evaluate(X, y, model_name="svm", test_size=0.2, random_state=42)
    print(results["accuracy"])
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ── Type aliases ────────────────────────────────────────────────────────────
ModelName = Literal["svm", "random_forest", "xgboost"]
MetricsDict = Dict[str, Any]

# ── Supported classifiers registry ─────────────────────────────────────────
CLASSIFIERS: List[ModelName] = ["svm", "random_forest", "xgboost"]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# Each builder returns a fresh, unfitted sklearn Pipeline.
# StandardScaler is always the first step so raw feature matrices are accepted.
# ══════════════════════════════════════════════════════════════════════════════

def build_svm(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = 42,
) -> Pipeline:
    """
    Build a Support Vector Machine (SVM) classification pipeline.

    Architecture:
        StandardScaler → SVC(kernel='rbf')

    Parameters
    ----------
    C : float
        Regularisation parameter. Higher values reduce margin violations.
    kernel : str
        Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
    gamma : str | float
        Kernel coefficient. 'scale' = 1 / (n_features × X.var()).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def build_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    criterion: str = "entropy",
    random_state: int = 42,
) -> Pipeline:
    """
    Build a Random Forest classification pipeline.

    Architecture:
        StandardScaler → RandomForestClassifier(criterion='entropy')

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int | None
        Maximum depth of each tree. None = expand until pure leaves.
    criterion : str
        Split quality measure ('entropy' or 'gini').
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def build_xgboost(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """
    Build an XGBoost classification pipeline.

    Architecture:
        StandardScaler → XGBClassifier(objective='multi:softmax')

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth per boosting round.
    learning_rate : float
        Step size shrinkage (eta) to prevent overfitting.
    subsample : float
        Fraction of samples used per boosting round.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            verbosity=0,
        )),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# CORE TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Fit a classifier pipeline on training data.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline (from any build_* function).
    X_train : np.ndarray, shape (n_samples, n_features)
        Training feature matrix.
    y_train : np.ndarray, shape (n_samples,)
        Integer or string class labels.

    Returns
    -------
    Pipeline
        Fitted pipeline ready for prediction.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> MetricsDict:
    """
    Evaluate a fitted pipeline and return a comprehensive metrics dictionary.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    X_test : np.ndarray, shape (n_samples, n_features)
        Test feature matrix.
    y_test : np.ndarray, shape (n_samples,)
        True class labels.
    class_names : list[str] | None
        Human-readable class label names for the classification report.

    Returns
    -------
    MetricsDict
        {
            "accuracy"         : float,          # overall accuracy [0, 1]
            "precision_macro"  : float,          # macro-averaged precision
            "recall_macro"     : float,          # macro-averaged recall
            "f1_macro"         : float,          # macro-averaged F1
            "f1_weighted"      : float,          # weighted F1
            "confusion_matrix" : list[list[int]],# rows=true, cols=pred
            "classification_report": str,        # per-class breakdown
            "n_test_samples"   : int,            # number of test samples
            "n_classes"        : int,            # number of unique classes
        }
    """
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy":              round(float(accuracy_score(y_test, y_pred)), 6),
        "precision_macro":       round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro":          round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro":              round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 6),
        "f1_weighted":           round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 6),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0,
        ),
        "n_test_samples":        int(len(y_test)),
        "n_classes":             int(len(np.unique(y_test))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def train_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: ModelName = "svm",
    test_size: float = 0.2,
    random_state: int = 42,
    class_names: Optional[List[str]] = None,
    **model_kwargs: Any,
) -> MetricsDict:
    """
    End-to-end convenience function: split → build → train → evaluate.

    Automatically encodes string labels to integers for XGBoost compatibility.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Full feature matrix (e.g., output of extract_all_features).
    y : np.ndarray, shape (n_samples,)
        Class labels (int or str).
    model_name : {"svm", "random_forest", "xgboost"}
        Which classifier to use.
    test_size : float
        Fraction of data reserved for testing (default 0.2 = 20%).
    random_state : int
        Seed for train/test split and model.
    class_names : list[str] | None
        Optional human-readable class names for the report.
    **model_kwargs
        Extra keyword arguments forwarded to the model builder.

    Returns
    -------
    MetricsDict
        Evaluation metrics dict (same as evaluate_model) plus:
        {
            "model_name"    : str,
            "train_samples" : int,
            "test_samples"  : int,
        }

    Raises
    ------
    ValueError
        If model_name is not one of the supported classifiers.
    """
    if model_name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {CLASSIFIERS}"
        )

    # Encode string labels → integers (required by XGBoost)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if class_names is None:
        class_names = [str(c) for c in le.classes_]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    # Build pipeline
    builders = {
        "svm":           build_svm,
        "random_forest": build_random_forest,
        "xgboost":       build_xgboost,
    }
    pipeline = builders[model_name](random_state=random_state, **model_kwargs)

    # Train
    trained = train_model(pipeline, X_train, y_train)

    # Evaluate
    metrics = evaluate_model(trained, X_test, y_test, class_names=class_names)
    metrics["model_name"]    = model_name
    metrics["train_samples"] = int(len(X_train))
    metrics["test_samples"]  = int(len(X_test))

    return metrics


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: ModelName = "svm",
    n_splits: int = 5,
    random_state: int = 42,
    class_names: Optional[List[str]] = None,
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform Group K-Fold Cross-Validation on the dataset.

    Groups (subjects) are guaranteed to be isolated between train and test
    folds, preventing subject-level data leakage.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        EEG feature matrix.
    y : np.ndarray, shape (n_samples,)
        Class labels (int or str).
    groups : np.ndarray, shape (n_samples,)
        Subject IDs/names to partition groups (denekler).
    model_name : {"svm", "random_forest", "xgboost"}
        Which classifier pipeline to run.
    n_splits : int
        Number of GroupKFold partitions. Automatically capped to the number
        of unique groups.
    random_state : int
        Seed for model reproducibility.
    class_names : list[str] | None
        Human-readable labels for confusion matrix and report.
    **model_kwargs
        Extra arguments forwarded to model builders.

    Returns
    -------
    dict
        Detailed cross-validation stats containing:
        {
            "model_name": str,
            "n_splits": int,
            "mean_accuracy": float,
            "std_accuracy": float,
            "mean_f1_macro": float,
            "mean_precision_macro": float,
            "mean_recall_macro": float,
            "folds": list[dict], # list of metrics for each fold
            "accumulated_confusion_matrix": list[list[int]],
            "class_names": list[str],
            "n_samples": int,
        }
    """
    if model_name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {CLASSIFIERS}"
        )

    # Encode string labels → integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if class_names is None:
        class_names = [str(c) for c in le.classes_]

    # Convert groups to numpy array
    groups = np.array(groups)
    unique_groups = np.unique(groups)
    num_unique_groups = len(unique_groups)

    if num_unique_groups < 2:
        raise ValueError(
            f"GroupKFold requires at least 2 unique groups (subjects). "
            f"Found only {num_unique_groups}: {unique_groups.tolist()}"
        )

    # Cap n_splits if there are fewer unique subjects than splits
    actual_splits = min(n_splits, num_unique_groups)

    gkf = GroupKFold(n_splits=actual_splits)
    builders = {
        "svm":           build_svm,
        "random_forest": build_random_forest,
        "xgboost":       build_xgboost,
    }

    fold_metrics_list = []
    accuracies = []
    f1_macros = []
    precisions = []
    recalls = []

    # Initialize accumulated confusion matrix (shape: n_classes x n_classes)
    n_classes = len(class_names)
    accum_cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y_enc, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        # Build & train fresh model pipeline
        pipeline = builders[model_name](random_state=random_state, **model_kwargs)
        trained_pipeline = train_model(pipeline, X_train, y_train)

        # Evaluate on test fold
        fold_metrics = evaluate_model(
            trained_pipeline, X_test, y_test, class_names=class_names
        )

        # Update lists and accumulated confusion matrix
        accuracies.append(fold_metrics["accuracy"])
        f1_macros.append(fold_metrics["f1_macro"])
        precisions.append(fold_metrics["precision_macro"])
        recalls.append(fold_metrics["recall_macro"])

        fold_cm = np.array(fold_metrics["confusion_matrix"])
        accum_cm += fold_cm

        fold_metrics_list.append({
            "fold_index":       int(i + 1),
            "accuracy":         fold_metrics["accuracy"],
            "precision_macro":  fold_metrics["precision_macro"],
            "recall_macro":     fold_metrics["recall_macro"],
            "f1_macro":         fold_metrics["f1_macro"],
            "f1_weighted":      fold_metrics["f1_weighted"],
            "confusion_matrix": fold_metrics["confusion_matrix"],
            "train_samples":    int(len(X_train)),
            "test_samples":     int(len(X_test)),
            "train_subjects":   [str(g) for g in np.unique(groups_train)],
            "test_subjects":    [str(g) for g in np.unique(groups_test)],
        })

    return {
        "model_name":                   model_name,
        "n_splits":                     actual_splits,
        "mean_accuracy":                round(float(np.mean(accuracies)), 6),
        "std_accuracy":                 round(float(np.std(accuracies)), 6),
        "mean_precision_macro":         round(float(np.mean(precisions)), 6),
        "mean_recall_macro":            round(float(np.mean(recalls)), 6),
        "mean_f1_macro":                round(float(np.mean(f1_macros)), 6),
        "folds":                        fold_metrics_list,
        "accumulated_confusion_matrix": accum_cm.tolist(),
        "n_samples":                    int(len(X)),
        "n_classes":                    int(n_classes),
        "class_names":                  class_names,
    }

