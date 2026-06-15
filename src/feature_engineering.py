import pandas as pd
import numpy as np

def engineer_game_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    windows = [1, 3, 5, 7, 10, 15]
    
    # Initialize rolling columns
    for w in windows:
        df[f'home_team_avg_runs_last{w}'] = 0.0
        df[f'away_team_avg_runs_last{w}'] = 0.0
        df[f'home_team_avg_runs_allowed_last{w}'] = 0.0
        df[f'away_team_avg_runs_allowed_last{w}'] = 0.0
        df[f'home_team_form_rating_last{w}'] = 0.0
        df[f'away_team_form_rating_last{w}'] = 0.0
    
    # New feature columns
    df['home_team_sp_era_last5'] = 0.0
    df['away_team_sp_era_last5'] = 0.0
    df['home_bullpen_era_last5'] = 0.0
    df['away_bullpen_era_last5'] = 0.0
    df['home_park_factor'] = 1.0
    df['away_park_factor'] = 1.0
    
    team_logs = {team: [] for team in all_teams}
    
    for idx, game in df.iterrows():
        home, away = game['home_team'], game['away_team']
        home_log = team_logs[home]
        away_log = team_logs[away]
        
        # Rolling averages for runs and form
        for w in windows:
            home_recent = home_log[-w:] if len(home_log) >= w else home_log
            away_recent = away_log[-w:] if len(away_log) >= w else away_log
            
            if len(home_recent) > 0:
                df.at[idx, f'home_team_avg_runs_last{w}'] = np.mean([g['runs_scored'] for g in home_recent])
                df.at[idx, f'home_team_avg_runs_allowed_last{w}'] = np.mean([g['runs_allowed'] for g in home_recent])
                df.at[idx, f'home_team_form_rating_last{w}'] = np.mean([g['won'] for g in home_recent])
            else:
                df.at[idx, f'home_team_avg_runs_last{w}'] = 4.5
                df.at[idx, f'home_team_avg_runs_allowed_last{w}'] = 4.5
                df.at[idx, f'home_team_form_rating_last{w}'] = 0.5
            
            if len(away_recent) > 0:
                df.at[idx, f'away_team_avg_runs_last{w}'] = np.mean([g['runs_scored'] for g in away_recent])
                df.at[idx, f'away_team_avg_runs_allowed_last{w}'] = np.mean([g['runs_allowed'] for g in away_recent])
                df.at[idx, f'away_team_form_rating_last{w}'] = np.mean([g['won'] for g in away_recent])
            else:
                df.at[idx, f'away_team_avg_runs_last{w}'] = 4.5
                df.at[idx, f'away_team_avg_runs_allowed_last{w}'] = 4.5
                df.at[idx, f'away_team_form_rating_last{w}'] = 0.5
        
        # Starting pitcher ERA rolling average (last 5 games)
        sp_era_home = [g['sp_era'] for g in home_log[-5:]] if home_log else [game['home_team_sp_era']]
        sp_era_away = [g['sp_era'] for g in away_log[-5:]] if away_log else [game['away_team_sp_era']]
        df.at[idx, 'home_team_sp_era_last5'] = np.mean(sp_era_home) if sp_era_home else 4.0
        df.at[idx, 'away_team_sp_era_last5'] = np.mean(sp_era_away) if sp_era_away else 4.0
        
        # Bullpen ERA rolling average
        bullpen_home = [g['bullpen_era'] for g in home_log[-5:]] if home_log else [4.0]
        bullpen_away = [g['bullpen_era'] for g in away_log[-5:]] if away_log else [4.0]
        df.at[idx, 'home_bullpen_era_last5'] = np.mean(bullpen_home)
        df.at[idx, 'away_bullpen_era_last5'] = np.mean(bullpen_away)
        
        # Park factors (use game's actual value if available, else 1.0)
        df.at[idx, 'home_park_factor'] = game.get('home_park_factor', 1.0)
        df.at[idx, 'away_park_factor'] = game.get('away_park_factor', 1.0)
        
        # Pitching strength (inverse of runs allowed)
        df.at[idx, 'home_team_pitching_strength'] = 1.0 / (df.at[idx, 'home_team_avg_runs_allowed_last5'] + 0.1)
        df.at[idx, 'away_team_pitching_strength'] = 1.0 / (df.at[idx, 'away_team_avg_runs_allowed_last5'] + 0.1)
        
        # Expected margin
        df.at[idx, 'expected_margin'] = (df.at[idx, 'home_team_avg_runs_last5'] + 0.5) - df.at[idx, 'away_team_avg_runs_last5']
        
        # Update team logs with SP ERA and bullpen ERA
        team_logs[home].append({
            'runs_scored': game['home_team_runs'],
            'runs_allowed': game['away_team_runs'],
            'won': game['home_win'],
            'sp_era': game['home_team_sp_era'],
            'bullpen_era': game.get('home_bullpen_era', 4.0)
        })
        team_logs[away].append({
            'runs_scored': game['away_team_runs'],
            'runs_allowed': game['home_team_runs'],
            'won': 1 - game['home_win'],
            'sp_era': game['away_team_sp_era'],
            'bullpen_era': game.get('away_bullpen_era', 4.0)
        })
    
    # Target variables
    median_total = df['total_runs'].median()
    df['total_over'] = (df['total_runs'] > median_total).astype(int)
    df['home_win'] = (df['home_team_runs'] > df['away_team_runs']).astype(int)
    df['run_line_cover'] = (abs(df['home_team_runs'] - df['away_team_runs']) > 1.5).astype(int)
    
    df = df.dropna(subset=['home_team_avg_runs_last5', 'away_team_avg_runs_last5'])
    return df