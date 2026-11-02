"""Shared metric helper for the Phishing Websites classification assignment."""
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# Target encoding used everywhere: 1 = Phishing (positive class), 0 = Legitimate.
CLASS_NAMES = ["Legitimate", "Phishing"]


def encode_target(result):
    """UCI 'Result' column: -1 = phishing, 1 = legitimate  ->  1 = phishing, 0 = legitimate."""
    return (np.asarray(result).astype(int) == -1).astype(int)


def compute_metrics(model, X, y):
    """Binary metrics with Phishing as the positive class."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:  # uploaded CSV contains only one class
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
