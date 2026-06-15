#!/usr/bin/env python
import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import functools

# Force all print statements to flush immediately
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_fetcher import MLBDataFetcher
from src.feature_engineering import engineer_game_features
from src.train_models import train_models
from src.predict import get_todays_games, manual_odds_input, prepare_features_for_tomorrow, get_recommendations
import joblib

def parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d').date()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Update cache and retrain models')
    parser.add_argument('--predict', action='store_true', help='Run prediction for tomorrow')
    parser.add_argument('--start', type=parse_date, help='Start date for fetching (YYYY-MM-DD)')
    parser.add_argument('--end', type=parse_date, help='End date for fetching (YYYY-MM-DD)')
    parser.add_argument('--append', action='store_true', help='Append fetched range to existing cache')
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Bankroll amount (default: 1000)')
    args = parser.parse_args()
    
    do_train = args.train or (not args.predict)
    do_predict = args.predict or (not args.train)
    
    if do_train:
        print("\n=== Training Mode ===")
        fetcher = MLBDataFetcher(cache_dir='data')
        
        if args.start and args.end:
            print(f"Fetching custom range: {args.start} to {args.end}")
            historical = fetcher.fetch_custom_range(args.start, args.end, append_to_cache=args.append)
        else:
            historical = fetcher.update_cache_incremental()
        
        if historical.empty:
            print("No historical data available. Exiting.")
            return
        
        print("Engineering features...")
        game_features = engineer_game_features(historical)
        print("Training models...")
        models, feature_sets = train_models(game_features, models_dir='models')
        print("Training complete.")
    
    if do_predict:
        print("\n=== Prediction Mode ===")
        if not os.path.exists('models/total_model.pkl'):
            print("No trained models found. Run with --train first.")
            return
        models = {
            'total': joblib.load('models/total_model.pkl'),
            'winner': joblib.load('models/winner_calibrated.pkl'),
            'runline': joblib.load('models/runline_model.pkl')
        }
        feature_sets = joblib.load('models/feature_sets.pkl')
        
        fetcher = MLBDataFetcher(cache_dir='data')
        cache_file = 'data/master_games.csv'
        if not os.path.exists(cache_file):
            print("Historical data not found. Run --train first.")
            return
        historical = pd.read_csv(cache_file)
        
        tomorrow_games = get_todays_games()   # changed from get_tomorrows_games
        if not tomorrow_games:
            print("No games scheduled for today.")
            return
        print(f"\nFound {len(tomorrow_games)} games today.")
        odds_data = manual_odds_input(tomorrow_games)
        if not odds_data:
            print("No odds entered. Exiting.")
            return
        features_df = prepare_features_for_tomorrow(tomorrow_games, historical)
        recs = get_recommendations(features_df, odds_data, models, feature_sets, bankroll=args.bankroll)
        if recs.empty:
            print("\nNo positive EV bets found.")
        else:
            print("\n" + "="*80)
            print(f"📈 POSITIVE EXPECTED VALUE BETS (Bankroll: {args.bankroll:.2f} PHP)")
            print("="*80)
            print(recs.to_string(index=False))

if __name__ == '__main__':
    main()