import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib

def fit_isotonic_calibration(probs, y_true, out_path=None):
    """
    Fit isotonic regression to map probabilities to actual frequencies.
    probs: array of predicted probabilities
    y_true: binary labels
    Returns a callable calibrator.
    """
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(probs, y_true)
    if out_path:
        joblib.dump(iso, out_path)
    return iso

def apply_isotonic_calibration(iso, probs):
    return iso.predict(probs)

def fit_temperature_scaling(logits, y_true, out_path=None):
    """
    Fit logistic regression on logits (temperature scaling).
    logits: logit(prob) = log(prob/(1-prob))
    """
    lr = LogisticRegression()
    lr.fit(logits.reshape(-1,1), y_true)
    if out_path:
        joblib.dump(lr, out_path)
    return lr

def apply_temperature_scaling(lr, logits):
    return lr.predict_proba(logits.reshape(-1,1))[:,1]