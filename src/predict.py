import pandas as pd
import numpy as np
import joblib
import statsapi
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Piecewise Calibration (updated deciles) ----------------------
def calibrate_proba_using_deciles(p):
    """
    Updated piecewise correction based on your decile analysis.
    Clamps extreme probabilities to reduce overconfidence.
    """
    # First clamp to reasonable range (0.05 – 0.95)
    p = max(0.05, min(0.95, p))
    
    # Apply mapping from your deciles
    if p <= 0.008:
        return 0.045
    elif p <= 0.160:
        # linear interpolation between 0.008→0.045 and 0.160→0.479
        return 0.045 + (p - 0.008) * (0.479 - 0.045) / (0.160 - 0.008)
    elif p <= 0.265:
        return 0.479 + (p - 0.160) * (0.470 - 0.479) / (0.265 - 0.160)
    elif p <= 0.350:
        return 0.470 + (p - 0.265) * (0.484 - 0.470) / (0.350 - 0.265)
    elif p <= 0.439:
        return 0.484 + (p - 0.350) * (0.518 - 0.484) / (0.439 - 0.350)
    elif p <= 0.564:
        return 0.518 + (p - 0.439) * (0.544 - 0.518) / (0.564 - 0.439)
    elif p <= 0.643:
        return 0.544 + (p - 0.564) * (0.537 - 0.544) / (0.643 - 0.564)
    elif p <= 0.748:
        return 0.537 + (p - 0.643) * (0.520 - 0.537) / (0.748 - 0.643)
    elif p <= 0.847:
        return 0.520 + (p - 0.748) * (0.578 - 0.520) / (0.847 - 0.748)
    else:
        return 0.578 + (p - 0.847) * (0.547 - 0.578) / (0.948 - 0.847)

# ---------------------- Game Retrieval (fixed for today's games) ----------------------
def get_todays_games():
    """
    Fetch games scheduled for today (using ET timezone).
    If no games found, try tomorrow.
    """
    # Use Eastern Time (ET) for MLB schedule
    from datetime import timezone
    now_et = datetime.now(timezone(timedelta(hours=-4)))  # EDT = UTC-4
    today_str = now_et.strftime("%Y-%m-%d")
    
    print(f"Fetching games for {today_str} (ET)...")
    games = statsapi.schedule(start_date=today_str, end_date=today_str)
    
    if not games:
        print(f"No games found for {today_str}. Trying tomorrow...")
        tomorrow_str = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")
        games = statsapi.schedule(start_date=tomorrow_str, end_date=tomorrow_str)
    
    if not games:
        print("No games found for today or tomorrow. Check API or date.")
        return []
    
    game_list = []
    for g in games:
        # Use 'game_date' from API, which is in ET
        game_list.append({
            'home_team': g.get('home_name'),
            'away_team': g.get('away_name'),
            'game_id': g.get('game_id'),
            'date': g.get('game_date')
        })
    print(f"Found {len(game_list)} games.")
    return game_list

def manual_odds_input(games):
    if not games:
        return {}
    print("\n" + "="*60)
    print("SELECT GAMES TO BET ON (Today's schedule)")
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
    print("Enter odds in DECIMAL format (e.g., 1.69, 2.10).")
    print("Press Enter to skip a bet type.\n")
    for g in selected:
        print(f"\n📌 {g['away_team']} @ {g['home_team']}")
        g_odds = {}
        ml_home = input("  Home Moneyline (decimal): ").strip()
        if ml_home:
            g_odds['moneyline_home'] = float(ml_home)
        ml_away = input("  Away Moneyline (decimal): ").strip()
        if ml_away:
            g_odds['moneyline_away'] = float(ml_away)
        rl_home = input("  Home Run Line -1.5 (decimal): ").strip()
        if rl_home:
            g_odds['runline_home'] = float(rl_home)
        rl_away = input("  Away Run Line +1.5 (decimal): ").strip()
        if rl_away:
            g_odds['runline_away'] = float(rl_away)
        ou_line = input("  Over/Under line (e.g., 8.5): ").strip()
        if ou_line:
            g_odds['over_under_line'] = float(ou_line)
            over_odds = input("  Over odds (decimal): ").strip()
            if over_odds:
                g_odds['over_odds'] = float(over_odds)
            under_odds = input("  Under odds (decimal): ").strip()
            if under_odds:
                g_odds['under_odds'] = float(under_odds)
        if g_odds:
            odds[f"{g['away_team']} @ {g['home_team']}"] = g_odds
    return odds

# ... (the rest of compute_all_rolling_stats, prepare_features_for_tomorrow, calculate_ev_decimal, kelly_fraction, get_recommendations remain exactly as in the previous version) ...

# The rest of the functions are unchanged; I'll include them for completeness.
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
    games = []
    for _, row in df.iterrows():
        if row['home_team'] == team_name:
            games.append({
                'runs_scored': row['home_team_runs'],
                'runs_allowed': row['away_team_runs'],
                'won': row['home_win']
            })
        else:
            games.append({
                'runs_scored': row['away_team_runs'],
                'runs_allowed': row['home_team_runs'],
                'won': 1 - row['home_win']
            })
    result = {}
    for w in windows:
        recent = games[:w]
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

def kelly_fraction(prob, odds, kelly_factor=0.25):
    b = odds - 1
    if b <= 0:
        return 0
    return max(0, ((prob * b - (1 - prob)) / b) * kelly_factor)

def get_recommendations(features_df, odds_data, models, feature_sets, bankroll=1000):
    import os
    bias_corrector = None
    if os.path.exists('models/winner_bias_corrector.pkl'):
        bias_corrector = joblib.load('models/winner_bias_corrector.pkl')
    winner_calibrated = models['winner']
    
    recs = []
    winner_feature_names = feature_sets['winner']
    runline_feature_names = feature_sets['runline']
    
    for _, row in features_df.iterrows():
        game_key = f"{row['away_team']} @ {row['home_team']}"
        game_odds = odds_data.get(game_key, {})
        if not game_odds:
            continue
        
        winner_features = [row[name] for name in winner_feature_names]
        winner_base = np.array([winner_features])
        runline_features = [row[name] for name in runline_feature_names]
        runline_base = np.array([runline_features])
        
        raw_proba = winner_calibrated.predict_proba(winner_base)[0][1]
        if bias_corrector:
            def logit(p): return np.log(p / (1 - p + 1e-8))
            logits = logit(raw_proba)
            home_win_prob = bias_corrector.predict_proba(np.array([[logits]]))[0][1]
        else:
            home_win_prob = raw_proba
        
        home_win_prob = calibrate_proba_using_deciles(home_win_prob)
        away_win_prob = 1 - home_win_prob
        
        raw_runline = models['runline'].predict_proba(runline_base)[0][1]
        runline_home_prob = calibrate_proba_using_deciles(raw_runline)
        runline_away_prob = 1 - runline_home_prob
        
        total_pred = models['total'].predict(winner_base)[0]
        
        def add_bet(prop, prob, odds_val):
            if prob < 0.25 or prob > 0.80:
                return
            ev = calculate_ev_decimal(prob, odds_val)
            if ev > 0.05:
                stake = kelly_fraction(prob, odds_val) * bankroll
                recs.append({
                    'game': game_key,
                    'prop': prop,
                    'model_prob': round(prob, 3),
                    'odds': round(odds_val, 2),
                    'ev': round(ev, 4),
                    'kelly_bet_$': round(stake, 2)
                })
        
        if 'moneyline_home' in game_odds:
            add_bet('Moneyline (Home)', home_win_prob, game_odds['moneyline_home'])
        if 'moneyline_away' in game_odds:
            add_bet('Moneyline (Away)', away_win_prob, game_odds['moneyline_away'])
        if 'runline_home' in game_odds:
            add_bet('Run Line -1.5 (Home)', runline_home_prob, game_odds['runline_home'])
        if 'runline_away' in game_odds:
            add_bet('Run Line +1.5 (Away)', runline_away_prob, game_odds['runline_away'])
        if 'over_under_line' in game_odds and 'over_odds' in game_odds:
            line = game_odds['over_under_line']
            over_prob = np.clip(1 - (line - total_pred) / 10, 0.1, 0.9)
            over_prob = calibrate_proba_using_deciles(over_prob)
            add_bet(f"Over {line} runs", over_prob, game_odds['over_odds'])
        if 'over_under_line' in game_odds and 'under_odds' in game_odds:
            line = game_odds['over_under_line']
            under_prob = 1 - np.clip(1 - (line - total_pred) / 10, 0.1, 0.9)
            under_prob = calibrate_proba_using_deciles(under_prob)
            add_bet(f"Under {line} runs", under_prob, game_odds['under_odds'])
    
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs).sort_values('ev', ascending=False)