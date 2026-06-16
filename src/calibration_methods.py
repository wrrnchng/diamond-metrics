import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration
import joblib

class PlattCalibration:
    def __init__(self):
        self.calibrator = None
    def fit(self, probs, y_true):
        X = probs.reshape(-1, 1)
        self.calibrator = LogisticRegression()
        self.calibrator.fit(X, y_true)
        return self
    def predict(self, probs):
        X = probs.reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]
    def save(self, path):
        joblib.dump(self.calibrator, path)
    def load(self, path):
        self.calibrator = joblib.load(path)

class IsotonicCalibration:
    def __init__(self):
        self.calibrator = None
    def fit(self, probs, y_true):
        self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.calibrator.fit(probs, y_true)
        return self
    def predict(self, probs):
        return self.calibrator.predict(probs)
    def save(self, path):
        joblib.dump(self.calibrator, path)
    def load(self, path):
        self.calibrator = joblib.load(path)

class BetaCalibrationWrapper:
    def __init__(self):
        self.calibrator = None
    def fit(self, probs, y_true):
        self.calibrator = BetaCalibration()
        self.calibrator.fit(probs, y_true)
        return self
    def predict(self, probs):
        return self.calibrator.predict(probs)
    def save(self, path):
        joblib.dump(self.calibrator, path)
    def load(self, path):
        self.calibrator = joblib.load(path)

class TemperatureScaling:
    def __init__(self):
        self.calibrator = None
    def fit(self, probs, y_true):
        logits = np.log(probs / (1 - probs + 1e-8))
        X = logits.reshape(-1, 1)
        self.calibrator = LogisticRegression()
        self.calibrator.fit(X, y_true)
        return self
    def predict(self, probs):
        logits = np.log(probs / (1 - probs + 1e-8))
        X = logits.reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]
    def save(self, path):
        joblib.dump(self.calibrator, path)
    def load(self, path):
        self.calibrator = joblib.load(path)

class EnsembleCalibration:
    def __init__(self, calibrators):
        self.calibrators = calibrators
    def fit(self, probs, y_true):
        for cal in self.calibrators:
            cal.fit(probs, y_true)
        return self
    def predict(self, probs):
        preds = [cal.predict(probs) for cal in self.calibrators]
        return np.mean(preds, axis=0)
    def save(self, path):
        for i, cal in enumerate(self.calibrators):
            cal.save(f"{path}_cal{i}.pkl")