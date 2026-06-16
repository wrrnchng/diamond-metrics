import pandas as pd
import numpy as np

def engineer_game_features(df, decay_factor=0.95):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    windows = [1, 3, 5, 7, 10, 15]
    for w in windows:
        df[f'home_team_avg_runs_last{w}'] = 0.0
        df[f'away_team_avg_runs_last{w}'] = 0.0
        df[f'home_team_avg_runs_allowed_last{w}'] = 0.0
        df[f'away_team_avg_runs_allowed_last{w}'] = 0.0
        df[f'home_team_form_rating_last{w}'] = 0.0
        df[f'away_team_form_rating_last{w}'] = 0.0
    team_logs = {team: [] for team in all_teams}
    for idx, game in df.iterrows():
        home, away = game['home_team'], game['away_team']
        home_log = team_logs[home][-15:]
        away_log = team_logs[away][-15:]
        def weighted_avg(series, weights):
            if not series:
                return 0.0
            weights = weights[:len(series)]
            return np.average(series, weights=weights)
        for w in windows:
            home_recent = home_log[-w:] if len(home_log) >= w else home_log
            away_recent = away_log[-w:] if len(away_log) >= w else away_log
            if len(home_recent) > 0:
                weights = [decay_factor ** i for i in range(len(home_recent)-1, -1, -1)]
                df.at[idx, f'home_team_avg_runs_last{w}'] = weighted_avg([g['runs_scored'] for g in home_recent], weights)
                df.at[idx, f'home_team_avg_runs_allowed_last{w}'] = weighted_avg([g['runs_allowed'] for g in home_recent], weights)
                df.at[idx, f'home_team_form_rating_last{w}'] = weighted_avg([g['won'] for g in home_recent], weights)
            else:
                df.at[idx, f'home_team_avg_runs_last{w}'] = 4.5
                df.at[idx, f'home_team_avg_runs_allowed_last{w}'] = 4.5
                df.at[idx, f'home_team_form_rating_last{w}'] = 0.5
            if len(away_recent) > 0:
                weights = [decay_factor ** i for i in range(len(away_recent)-1, -1, -1)]
                df.at[idx, f'away_team_avg_runs_last{w}'] = weighted_avg([g['runs_scored'] for g in away_recent], weights)
                df.at[idx, f'away_team_avg_runs_allowed_last{w}'] = weighted_avg([g['runs_allowed'] for g in away_recent], weights)
                df.at[idx, f'away_team_form_rating_last{w}'] = weighted_avg([g['won'] for g in away_recent], weights)
            else:
                df.at[idx, f'away_team_avg_runs_last{w}'] = 4.5
                df.at[idx, f'away_team_avg_runs_allowed_last{w}'] = 4.5
                df.at[idx, f'away_team_form_rating_last{w}'] = 0.5
        df.at[idx, 'home_team_pitching_strength'] = 1.0 / (df.at[idx, 'home_team_avg_runs_allowed_last5'] + 0.1)
        df.at[idx, 'away_team_pitching_strength'] = 1.0 / (df.at[idx, 'away_team_avg_runs_allowed_last5'] + 0.1)
        df.at[idx, 'expected_margin'] = (df.at[idx, 'home_team_avg_runs_last5'] + 0.5) - df.at[idx, 'away_team_avg_runs_last5']
        team_logs[home].append({
            'runs_scored': game['home_team_runs'],
            'runs_allowed': game['away_team_runs'],
            'won': game['home_win']
        })
        team_logs[away].append({
            'runs_scored': game['away_team_runs'],
            'runs_allowed': game['home_team_runs'],
            'won': 1 - game['home_win']
        })
    median_total = df['total_runs'].median()
    df['total_over'] = (df['total_runs'] > median_total).astype(int)
    df['home_win'] = (df['home_team_runs'] > df['away_team_runs']).astype(int)
    df['run_line_cover'] = (abs(df['home_team_runs'] - df['away_team_runs']) > 1.5).astype(int)
    df = df.dropna(subset=['home_team_avg_runs_last5', 'away_team_avg_runs_last5'])
    return df