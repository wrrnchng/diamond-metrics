import pandas as pd
import numpy as np
import joblib
import statsapi
from datetime import datetime, timedelta
from datetime import timezone
import warnings
import os

warnings.filterwarnings('ignore')

def get_todays_games():
    now_et = datetime.now(timezone(timedelta(hours=-4)))
    today_str = now_et.strftime("%Y-%m-%d")
    print(f"Fetching games for {today_str} (ET)...")
    games = statsapi.schedule(start_date=today_str, end_date=today_str)
    if not games:
        print(f"No games found for {today_str}. Trying tomorrow...")
        tomorrow_str = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")
        games = statsapi.schedule(start_date=tomorrow_str, end_date=tomorrow_str)
    if not games:
        print("No games found.")
        return []
    return [{'home_team': g['home_name'], 'away_team': g['away_name'], 'game_id': g['game_id']} for g in games]

def manual_odds_input(games):
    if not games:
        return {}
    print("\n" + "="*60)
    print("SELECT GAMES TO BET ON")
    print("="*60)
    for i, g in enumerate(games, 1):
        print(f"  {i}. {g['away_team']} @ {g['home_team']}")
    sel = input("\nYour selection (e.g., '1,3' or '1-4' or 'all'): ").strip()
    if sel.lower() == 'all':
        selected = games
    else:
        indices = set()
        for part in sel.replace(' ', '').split(','):
            if '-' in part:
                s, e = map(int, part.split('-'))
                indices.update(range(s, e+1))
            else:
                indices.add(int(part))
        selected = [games[i-1] for i in indices if 1 <= i <= len(games)]
    odds = {}
    print("\n" + "="*60)
    print("MANUAL ODDS ENTRY (Decimal format)")
    for g in selected:
        print(f"\n📌 {g['away_team']} @ {g['home_team']}")
        g_odds = {}
        ml_home = input("  Home Moneyline: ").strip()
        if ml_home: g_odds['moneyline_home'] = float(ml_home)
        ml_away = input("  Away Moneyline: ").strip()
        if ml_away: g_odds['moneyline_away'] = float(ml_away)
        rl_home = input("  Home Run Line -1.5: ").strip()
        if rl_home: g_odds['runline_home'] = float(rl_home)
        rl_away = input("  Away Run Line +1.5: ").strip()
        if rl_away: g_odds['runline_away'] = float(rl_away)
        # Over/Under lines (separate for over and under)
        over_line = input("  Over line (e.g., 8.5): ").strip()
        if over_line:
            g_odds['over_line'] = float(over_line)
            over_odds = input("  Over odds: ").strip()
            if over_odds: g_odds['over_odds'] = float(over_odds)
        under_line = input("  Under line (e.g., 8.5): ").strip()
        if under_line:
            g_odds['under_line'] = float(under_line)
            under_odds = input("  Under odds: ").strip()
            if under_odds: g_odds['under_odds'] = float(under_odds)
        # For backward compatibility, also allow a single line
        if 'over_line' not in g_odds and 'under_line' not in g_odds:
            ou_line = input("  Over/Under line (single line): ").strip()
            if ou_line:
                g_odds['over_under_line'] = float(ou_line)
                over_odds = input("  Over odds: ").strip()
                if over_odds: g_odds['over_odds'] = float(over_odds)
                under_odds = input("  Under odds: ").strip()
                if under_odds: g_odds['under_odds'] = float(under_odds)
        if g_odds:
            odds[f"{g['away_team']} @ {g['home_team']}"] = g_odds
    return odds

def compute_all_rolling_stats(historical_df, team_name, current_date, windows=[1,3,5,7,10,15]):
    df = historical_df[(historical_df['home_team'] == team_name) | (historical_df['away_team'] == team_name)]
    df = df[pd.to_datetime(df['date']) < current_date].copy()
    if df.empty:
        result = {}
        for w in windows:
            result[f'avg_runs_last{w}'] = 4.5
            result[f'avg_runs_allowed_last{w}'] = 4.5
            result[f'form_rating_last{w}'] = 0.5
        result['pitching_strength'] = 1.0 / 4.6
        return result
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    games_log = []
    for _, row in df.iterrows():
        if row['home_team'] == team_name:
            games_log.append({
                'runs_scored': row['home_team_runs'],
                'runs_allowed': row['away_team_runs'],
                'won': row['home_win']
            })
        else:
            games_log.append({
                'runs_scored': row['away_team_runs'],
                'runs_allowed': row['home_team_runs'],
                'won': 1 - row['home_win']
            })
    result = {}
    for w in windows:
        recent = games_log[:w]
        if len(recent) > 0:
            result[f'avg_runs_last{w}'] = np.mean([g['runs_scored'] for g in recent])
            result[f'avg_runs_allowed_last{w}'] = np.mean([g['runs_allowed'] for g in recent])
            result[f'form_rating_last{w}'] = np.mean([g['won'] for g in recent])
        else:
            result[f'avg_runs_last{w}'] = 4.5
            result[f'avg_runs_allowed_last{w}'] = 4.5
            result[f'form_rating_last{w}'] = 0.5
    w5 = windows[-3] if 5 in windows else 5
    result['pitching_strength'] = 1.0 / (result[f'avg_runs_allowed_last{w5}'] + 0.1)
    return result

def prepare_features_for_tomorrow(tomorrow_games, historical_df):
    current_date = datetime.now()
    windows = [1,3,5,7,10,15]
    features = []
    for g in tomorrow_games:
        home = g['home_team']
        away = g['away_team']
        home_stats = compute_all_rolling_stats(historical_df, home, current_date, windows)
        away_stats = compute_all_rolling_stats(historical_df, away, current_date, windows)
        row_dict = {'home_team': home, 'away_team': away}
        for w in windows:
            row_dict[f'home_team_avg_runs_last{w}'] = home_stats[f'avg_runs_last{w}']
            row_dict[f'away_team_avg_runs_last{w}'] = away_stats[f'avg_runs_last{w}']
            row_dict[f'home_team_form_rating_last{w}'] = home_stats[f'form_rating_last{w}']
            row_dict[f'away_team_form_rating_last{w}'] = away_stats[f'form_rating_last{w}']
        row_dict['home_team_pitching_strength'] = home_stats['pitching_strength']
        row_dict['away_team_pitching_strength'] = away_stats['pitching_strength']
        row_dict['expected_margin'] = (home_stats['avg_runs_last5'] + 0.5) - away_stats['avg_runs_last5']
        features.append(row_dict)
    return pd.DataFrame(features)

def calculate_ev_decimal(prob, odds):
    return (prob * odds) - 1

def kelly_fraction(prob, odds, kelly_factor=0.25, max_bankroll_fraction=0.05):
    b = odds - 1
    if b <= 0:
        return 0
    full_kelly = (prob * b - (1 - prob)) / b
    return max(0, min(full_kelly * kelly_factor, max_bankroll_fraction))

def get_recommendations(features_df, odds_data, models, feature_sets, bankroll=1000):
    winner_base = models['winner_base']
    calibrator = models['winner_calibrator']
    runline_base = models['runline_base']
    runline_cal = models['runline_calibrator']
    total_model = models['total']
    winner_feature_names = feature_sets['winner']
    runline_feature_names = feature_sets['runline']
    recs = []
    for _, row in features_df.iterrows():
        game_key = f"{row['away_team']} @ {row['home_team']}"
        game_odds = odds_data.get(game_key, {})
        if not game_odds:
            continue
        X_winner = np.array([[row[name] for name in winner_feature_names]])
        raw_proba = winner_base.predict_proba(X_winner)[0][1]
        home_win_prob = calibrator.predict(np.array([raw_proba]))[0]
        away_win_prob = 1 - home_win_prob
        X_runline = np.array([[row[name] for name in runline_feature_names]])
        raw_rl = runline_base.predict_proba(X_runline)[0][1]
        runline_home_prob = runline_cal.predict(np.array([raw_rl]))[0]
        runline_away_prob = 1 - runline_home_prob
        total_pred = total_model.predict(X_winner)[0]
        
        def add_bet(prop, prob, odds_val):
            ev = calculate_ev_decimal(prob, odds_val)
            if ev > 0:
                stake = kelly_fraction(prob, odds_val) * bankroll
                recs.append({
                    'game': game_key,
                    'prop': prop,
                    'model_prob': round(prob, 3),
                    'odds': round(odds_val, 2),
                    'ev': round(ev, 4),
                    'kelly_bet_$': round(stake, 2)
                })
        
        # Moneyline
        if 'moneyline_home' in game_odds: add_bet('Moneyline (Home)', home_win_prob, game_odds['moneyline_home'])
        if 'moneyline_away' in game_odds: add_bet('Moneyline (Away)', away_win_prob, game_odds['moneyline_away'])
        # Run Line
        if 'runline_home' in game_odds: add_bet('Run Line -1.5 (Home)', runline_home_prob, game_odds['runline_home'])
        if 'runline_away' in game_odds: add_bet('Run Line +1.5 (Away)', runline_away_prob, game_odds['runline_away'])
        # Totals: separate over and under lines
        if 'over_line' in game_odds and 'over_odds' in game_odds:
            line = game_odds['over_line']
            over_prob = np.clip(1 - (line - total_pred) / 10, 0.1, 0.9)
            add_bet(f"Over {line} runs", over_prob, game_odds['over_odds'])
        if 'under_line' in game_odds and 'under_odds' in game_odds:
            line = game_odds['under_line']
            under_prob = np.clip(1 - (line - total_pred) / 10, 0.1, 0.9)
            under_prob = 1 - under_prob  # because under probability is 1 - over_prob
            add_bet(f"Under {line} runs", under_prob, game_odds['under_odds'])
        # Backward compatibility: single line for both
        if 'over_under_line' in game_odds and 'over_odds' in game_odds and 'over_line' not in game_odds:
            line = game_odds['over_under_line']
            over_prob = np.clip(1 - (line - total_pred) / 10, 0.1, 0.9)
            add_bet(f"Over {line} runs", over_prob, game_odds['over_odds'])
        if 'over_under_line' in game_odds and 'under_odds' in game_odds and 'under_line' not in game_odds:
            line = game_odds['over_under_line']
            under_prob = np.clip(1 - (line - total_pred) / 10, 0.1, 0.9)
            under_prob = 1 - under_prob
            add_bet(f"Under {line} runs", under_prob, game_odds['under_odds'])
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs).sort_values('ev', ascending=False)