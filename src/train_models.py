import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss, brier_score_loss
import xgboost as xgb
import joblib
import os
import warnings
from .calibration_methods import (
    PlattCalibration, IsotonicCalibration, BetaCalibrationWrapper,
    TemperatureScaling, EnsembleCalibration
)
warnings.filterwarnings('ignore')

def train_models(game_features, models_dir='../models', calibration_method='isotonic'):
    os.makedirs(models_dir, exist_ok=True)
    windows = [1, 3, 5, 7, 10, 15]
    base_features = []
    for w in windows:
        base_features.extend([
            f'home_team_avg_runs_last{w}', f'away_team_avg_runs_last{w}',
            f'home_team_form_rating_last{w}', f'away_team_form_rating_last{w}'
        ])
    base_features += ['home_team_pitching_strength', 'away_team_pitching_strength']
    runline_features = base_features + ['expected_margin']
    
    X_total = game_features[base_features].fillna(game_features[base_features].median())
    y_total = game_features['total_runs']
    X_winner = game_features[base_features].fillna(game_features[base_features].median())
    y_winner = game_features['home_win']
    X_runline = game_features[runline_features].fillna(game_features[runline_features].median())
    y_runline = game_features['run_line_cover']
    
    split_idx = int(len(X_winner) * 0.8)
    X_winner_train, X_winner_val = X_winner.iloc[:split_idx], X_winner.iloc[split_idx:]
    y_winner_train, y_winner_val = y_winner.iloc[:split_idx], y_winner.iloc[split_idx:]
    X_runline_train, X_runline_val = X_runline.iloc[:split_idx], X_runline.iloc[split_idx:]
    y_runline_train, y_runline_val = y_runline.iloc[:split_idx], y_runline.iloc[split_idx:]
    X_total_train, X_total_val = X_total.iloc[:split_idx], X_total.iloc[split_idx:]
    y_total_train, y_total_val = y_total.iloc[:split_idx], y_total.iloc[split_idx:]
    
    # Total Runs
    print("Training Total Runs model...")
    total_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    total_model.fit(X_total_train, y_total_train)
    joblib.dump(total_model, f"{models_dir}/total_model.pkl")
    
    # Winner base model (XGBoost)
    print("Training Winner base model...")
    n_neg = (y_winner_train == 0).sum()
    n_pos = (y_winner_train == 1).sum()
    scale_pos_weight = (n_neg / n_pos) * 2.5
    xgb_winner = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
    xgb_winner.fit(X_winner_train, y_winner_train)
    raw_proba_val = xgb_winner.predict_proba(X_winner_val)[:, 1]
    
    # Calibration
    if calibration_method == 'platt':
        calibrator = PlattCalibration()
    elif calibration_method == 'isotonic':
        calibrator = IsotonicCalibration()
    elif calibration_method == 'beta':
        calibrator = BetaCalibrationWrapper()
    elif calibration_method == 'temperature':
        calibrator = TemperatureScaling()
    elif calibration_method == 'ensemble':
        calibrator = EnsembleCalibration([PlattCalibration(), IsotonicCalibration(), BetaCalibrationWrapper()])
    else:
        raise ValueError(f"Unknown method: {calibration_method}")
    calibrator.fit(raw_proba_val, y_winner_val)
    calibrated_val = calibrator.predict(raw_proba_val)
    val_ll = log_loss(y_winner_val, calibrated_val, labels=[0,1])
    val_bs = brier_score_loss(y_winner_val, calibrated_val)
    print(f"Winner model ({calibration_method}): LogLoss={val_ll:.4f}, Brier={val_bs:.4f}")
    
    joblib.dump(xgb_winner, f"{models_dir}/winner_base.pkl")
    calibrator.save(f"{models_dir}/winner_calibrator.pkl")
    
    # Run Line model (simple isotonic for consistency)
    print("Training Run Line model...")
    n_neg_rl = (y_runline_train == 0).sum()
    n_pos_rl = (y_runline_train == 1).sum()
    scale_pos_weight_rl = n_neg_rl / n_pos_rl if n_pos_rl > 0 else 1.0
    xgb_runline = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight_rl,
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
    xgb_runline.fit(X_runline_train, y_runline_train)
    joblib.dump(xgb_runline, f"{models_dir}/runline_base.pkl")
    from sklearn.isotonic import IsotonicRegression
    rl_raw = xgb_runline.predict_proba(X_runline_val)[:, 1]
    rl_iso = IsotonicRegression(out_of_bounds='clip')
    rl_iso.fit(rl_raw, y_runline_val)
    joblib.dump(rl_iso, f"{models_dir}/runline_isotonic.pkl")
    
    feature_sets = {'total': base_features, 'winner': base_features, 'runline': runline_features}
    joblib.dump(feature_sets, f"{models_dir}/feature_sets.pkl")
    
    return {
        'total': total_model,
        'winner_base': xgb_winner,
        'winner_calibrator': calibrator,
        'runline_base': xgb_runline,
        'runline_calibrator': rl_iso
    }, feature_sets