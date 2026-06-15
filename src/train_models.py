import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_models(game_features, models_dir='../models'):
    os.makedirs(models_dir, exist_ok=True)
    
    windows = [1, 3, 5, 7, 10, 15]
    base_features = []
    for w in windows:
        base_features.extend([
            f'home_team_avg_runs_last{w}', f'away_team_avg_runs_last{w}',
            f'home_team_form_rating_last{w}', f'away_team_form_rating_last{w}',
            f'home_opp_strength_last{w}', f'away_opp_strength_last{w}'  # new opponent strength
        ])
    base_features += ['home_team_pitching_strength', 'away_team_pitching_strength']
    runline_features = base_features + ['expected_margin']
    
    # Prepare data
    X_total = game_features[base_features].fillna(game_features[base_features].median())
    y_total = game_features['total_runs']
    X_winner = game_features[base_features].fillna(game_features[base_features].median())
    y_winner = game_features['home_win']
    X_runline = game_features[runline_features].fillna(game_features[runline_features].median())
    y_runline = game_features['run_line_cover']
    
    # Temporal split (80/20)
    split_idx = int(len(X_winner) * 0.8)
    X_winner_train, X_winner_val = X_winner.iloc[:split_idx], X_winner.iloc[split_idx:]
    y_winner_train, y_winner_val = y_winner.iloc[:split_idx], y_winner.iloc[split_idx:]
    X_runline_train, X_runline_val = X_runline.iloc[:split_idx], X_runline.iloc[split_idx:]
    y_runline_train, y_runline_val = y_runline.iloc[:split_idx], y_runline.iloc[split_idx:]
    X_total_train, X_total_val = X_total.iloc[:split_idx], X_total.iloc[split_idx:]
    y_total_train, y_total_val = y_total.iloc[:split_idx], y_total.iloc[split_idx:]
    
    # --- Total Runs (RandomForest) ---
    print("Training Total Runs model...")
    total_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    total_model.fit(X_total_train, y_total_train)
    joblib.dump(total_model, f"{models_dir}/total_model.pkl")
    
    # --- Winner model: XGBoost with class weights + Platt scaling ---
    print("Training Winner model with XGBoost...")
    # Compute scale_pos_weight (ratio of negative to positive)
    n_neg = (y_winner_train == 0).sum()
    n_pos = (y_winner_train == 1).sum()
    scale_pos_weight = (n_neg / n_pos) * 2.5   # increased underdog weight
    
    xgb_winner = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    winner_calibrated = CalibratedClassifierCV(xgb_winner, method='sigmoid', cv=3)
    winner_calibrated.fit(X_winner_train, y_winner_train)
    
    # Bias correction on logits (logistic regression)
    raw_proba_train = winner_calibrated.predict_proba(X_winner_train)[:, 1]
    def logit(p): return np.log(p / (1 - p + 1e-8))
    logits_train = logit(raw_proba_train)
    bias_corrector = LogisticRegression()
    bias_corrector.fit(logits_train.reshape(-1,1), y_winner_train)
    joblib.dump(winner_calibrated, f"{models_dir}/winner_calibrated.pkl")
    joblib.dump(bias_corrector, f"{models_dir}/winner_bias_corrector.pkl")
    
    # Validation metrics
    raw_proba_val = winner_calibrated.predict_proba(X_winner_val)[:, 1]
    logits_val = logit(raw_proba_val)
    corrected_proba_val = bias_corrector.predict_proba(logits_val.reshape(-1,1))[:, 1]
    val_logloss = log_loss(y_winner_val, corrected_proba_val, labels=[0,1])
    val_brier = brier_score_loss(y_winner_val, corrected_proba_val)
    print(f"Winner model Validation LogLoss: {val_logloss:.4f}, Brier: {val_brier:.4f}")
    
    # --- Run Line model (XGBoost with calibration) ---
    print("Training Run Line model with XGBoost...")
    n_neg_rl = (y_runline_train == 0).sum()
    n_pos_rl = (y_runline_train == 1).sum()
    scale_pos_weight_rl = n_neg_rl / n_pos_rl if n_pos_rl > 0 else 1.0
    xgb_runline = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_rl,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    runline_model = CalibratedClassifierCV(xgb_runline, method='isotonic', cv=3)
    runline_model.fit(X_runline_train, y_runline_train)
    joblib.dump(runline_model, f"{models_dir}/runline_model.pkl")
    
    # Save feature sets
    feature_sets = {
        'total': base_features,
        'winner': base_features,
        'runline': runline_features
    }
    joblib.dump(feature_sets, f"{models_dir}/feature_sets.pkl")
    
    return {'total': total_model, 'winner': winner_calibrated, 'runline': runline_model}, feature_sets