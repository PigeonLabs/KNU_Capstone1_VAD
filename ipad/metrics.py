"""Offline, scene-wide scoring with explicit temporal support and label policies."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite score')
    extent = np.ptp(values)
    return np.zeros_like(values) if extent < 1e-12 else (values-values.min())/extent


def period_errors(phases, period_length, classes=200, window=5):
    """Centered window; references derive ONLY from normal training period length.

    Circular residuals handle phase wrap. Window boundaries are invalid, not padded.
    Predictions refer to clip starts, while results retain their reconstruction frame IDs.
    """
    phases = np.asarray(phases, dtype=float)
    out = np.full(len(phases), np.nan)
    half = window//2
    offsets = np.arange(-half, half+1)*classes/period_length
    for i in range(half, len(phases)-half):
        reference = phases[i]+offsets
        delta = (phases[i-half:i+half+1]-reference+classes/2)%classes-classes/2
        out[i] = np.abs(delta).mean()
    return out


def binary_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid aligned scores')
    if not set(np.unique(labels)).issubset({0,1}):
        raise ValueError('Labels must be binary')
    if len(np.unique(labels)) < 2:
        return {'auroc': None, 'auprc': None, 'reason': 'single_class', 'frames': len(labels)}
    return {'auroc': float(roc_auc_score(labels, scores)*100),
            'auprc': float(average_precision_score(labels, scores)*100), 'frames': len(labels)}
