"""Model evaluation metrics and results persistence."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_test, y_pred, y_proba) -> dict:
    """Compute accuracy, precision/recall/F1 for the fraud class, ROC-AUC, and the confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_fraud": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_fraud": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def fit_and_score(models: dict[str, ClassifierMixin], X_train, X_test, y_train, y_test) -> dict[str, dict]:
    """Fit each model and evaluate it on the held-out split."""
    models_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        models_scores[name] = evaluate_model(y_test, y_pred, y_proba)
    return models_scores


def save_metrics(metrics: dict, path: str | os.PathLike) -> None:
    """Serialize a metrics dict to JSON at `path`, creating parent directories as needed."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def save_submission(trans_num: pd.Series, predictions: np.ndarray, path: str | os.PathLike) -> None:
    """Write final holdout predictions (trans_num, is_fraud) to a CSV at `path`."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = pd.DataFrame({"trans_num": trans_num, "is_fraud": predictions})
    response.to_csv(path, index=False)
