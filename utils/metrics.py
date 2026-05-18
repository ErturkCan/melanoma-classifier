import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall for melanoma class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc         = roc_auc_score(labels, probs)

    return {
        "auc":         round(float(auc), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
