#!/usr/bin/env python
import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import functools
import re
import numpy as np

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_fetcher import MLBDataFetcher
from src.feature_engineering import engineer_game_features
from src.train_models import train_models
from src.predict import get_todays_games, manual_odds_input, prepare_features_for_tomorrow, get_recommendations
import joblib

def format_currency(value):
    return f"₱{value:.2f}"

def format_probability(p):
    return f"{p:.1%}"

def format_ev(ev):
    if ev is None:
        return ""
    sign = "+" if ev > 0 else ""
    return f"{sign}{ev:.4f}"

def normalize_team(team):
    """Normalize team name for consistent matching."""
    return team.strip().lower()

def parlay_kelly_fraction(prob_product, odds_product, kelly_factor=0.25, max_bankroll_fraction=0.05):
    """Calculate Kelly stake fraction for a parlay."""
    b = odds_product - 1
    if b <= 0:
        return 0
    full_kelly = (prob_product * b - (1 - prob_product)) / b
    fractional = full_kelly * kelly_factor
    return max(0, min(fractional, max_bankroll_fraction))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Update cache and retrain models')
    parser.add_argument('--predict', action='store_true', help='Run prediction for today')
    parser.add_argument('--start', type=str, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='End date YYYY-MM-DD')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--bankroll', type=float, default=1000.0)
    args = parser.parse_args()
    
    do_train = args.train or (not args.predict)
    do_predict = args.predict or (not args.train)
    
    if do_train:
        print("\n=== Training Mode ===")
        fetcher = MLBDataFetcher(cache_dir='data')
        if args.start and args.end:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
            historical = fetcher.fetch_custom_range(start_date, end_date, append_to_cache=args.append)
        else:
            historical = fetcher.update_cache_incremental()
        if historical.empty:
            print("No data.")
            return
        print("Engineering features...")
        game_features = engineer_game_features(historical)
        print("Training models...")
        models, feature_sets = train_models(game_features, models_dir='models')
        print("Training complete.")
    
    if do_predict:
        print("\n=== Prediction Mode ===")
        if not os.path.exists('models/total_model.pkl'):
            print("No models. Run --train first.")
            return
        models = {
            'total': joblib.load('models/total_model.pkl'),
            'winner': joblib.load('models/winner_calibrated.pkl'),
            'runline': joblib.load('models/runline_model.pkl')
        }
        feature_sets = joblib.load('models/feature_sets.pkl')
        historical = pd.read_csv('data/master_games.csv')
        
        games = get_todays_games()
        if not games:
            print("No games today.")
            return
        print(f"\nFound {len(games)} games today.")
        odds_data = manual_odds_input(games)
        if not odds_data:
            print("No odds entered.")
            return
        features_df = prepare_features_for_tomorrow(games, historical)
        recs = get_recommendations(features_df, odds_data, models, feature_sets, bankroll=args.bankroll)
        
        # Build mapping of best bet per game using normalized team names
        best_by_game = {}
        if not recs.empty:
            for _, row in recs.iterrows():
                if ' @ ' not in row['game']:
                    continue
                parts = row['game'].split(' @ ')
                away = normalize_team(parts[0])
                home = normalize_team(parts[1])
                key = (away, home)
                if key not in best_by_game or row['ev'] > best_by_game[key]['ev']:
                    best_by_game[key] = row.to_dict()
        
        # Build output rows in original game order
        output_rows = []
        for g in games:
            away = normalize_team(g['away_team'])
            home = normalize_team(g['home_team'])
            key = (away, home)
            original_game_str = f"{g['away_team']} @ {g['home_team']}"
            if key in best_by_game:
                row = best_by_game[key]
                output_rows.append({
                    'game': original_game_str,
                    'prop': row['prop'],
                    'model_prob': row['model_prob'],
                    'odds': row['odds'],
                    'ev': row['ev'],
                    'kelly': row['kelly_bet_$'],
                    'has_positive_ev': row['ev'] > 0,
                    'ev_value': row['ev']
                })
            else:
                output_rows.append({
                    'game': original_game_str,
                    'prop': 'No positive EV bet',
                    'model_prob': None,
                    'odds': None,
                    'ev': None,
                    'kelly': None,
                    'has_positive_ev': False,
                    'ev_value': -999
                })
        
        # Separate positive EV games for parlays
        positive_games = [r for r in output_rows if r['has_positive_ev']]
        positive_games.sort(key=lambda x: -x['ev_value'])
        
        # Display per‑game table (positive first, then negative/none)
        pos_rows = [r for r in output_rows if r['has_positive_ev']]
        neg_rows = [r for r in output_rows if not r['has_positive_ev'] and r['ev'] is not None]
        no_bet_rows = [r for r in output_rows if r['ev'] is None]
        pos_rows.sort(key=lambda x: -x['ev_value'])
        neg_rows.sort(key=lambda x: -x['ev_value'])
        sorted_rows = pos_rows + neg_rows + no_bet_rows
        
        print("\n" + "="*130)
        print(f"📈 BEST BET PER GAME (Bankroll: {format_currency(args.bankroll)})")
        print(f"   Positive EV bets: {len(pos_rows)}  |  Negative EV bets: {len(neg_rows)}  |  No bet available: {len(no_bet_rows)}")
        print("="*130)
        print(f"{'#':<3} {'Game':<42} {'Prop':<32} {'Win Prob':<10} {'Odds':<8} {'EV':<12} {'Kelly Bet':<12}")
        print("-"*130)
        for i, row in enumerate(sorted_rows, 1):
            game_str = row['game'][:40] + ".." if len(row['game']) > 40 else row['game']
            if row['prop'] != 'No positive EV bet':
                prop_str = row['prop'][:30] + ".." if len(row['prop']) > 30 else row['prop']
                win_prob = format_probability(row['model_prob']) if row['model_prob'] else ""
                odds_str = f"{row['odds']:.2f}" if row['odds'] else ""
                ev_str = format_ev(row['ev'])
                kelly_str = format_currency(row['kelly']) if row['kelly'] else ""
                if not row['has_positive_ev'] and row['ev'] is not None:
                    ev_str = f"⚠️ {ev_str}"
                print(f"{i:<3} {game_str:<42} {prop_str:<32} {win_prob:<10} {odds_str:<8} {ev_str:<12} {kelly_str:<12}")
            else:
                print(f"{i:<3} {game_str:<42} {'No positive EV bet':<32} {'':<10} {'':<8} {'':<12} {'':<12}")
        print("="*130)
        
        # Top recommendation (first positive EV or highest negative)
        if pos_rows:
            best = pos_rows[0]
            print(f"\n🎯 Top pick (highest positive EV):")
            print(f"   {best['game']} - {best['prop']}")
            print(f"   Model win probability: {format_probability(best['model_prob'])}")
            print(f"   Odds: {best['odds']:.2f} | EV: +{best['ev']:.4f} | Suggested bet: {format_currency(best['kelly'])}")
        elif neg_rows:
            best = neg_rows[0]
            print(f"\n⚠️ No positive EV bets. Highest EV (negative):")
            print(f"   {best['game']} - {best['prop']}")
            print(f"   Model win probability: {format_probability(best['model_prob'])}")
            print(f"   Odds: {best['odds']:.2f} | EV: {best['ev']:.4f} | Suggested bet: {format_currency(best['kelly'])}")
            print("   WARNING: Negative expected value – not recommended.")
        else:
            print("\n❌ No bets available for any game (all filtered out).")
        
        # ==================== PARLAY BUILDER ====================
        if len(positive_games) >= 2:
            print("\n" + "="*130)
            print("🎲 AUTOMATIC PARLAY SUGGESTIONS (using top positive EV bets)")
            print("   Note: Parlays assume independence; actual variance is higher. Bet smaller stakes.")
            print("="*130)
            
            max_legs = min(5, len(positive_games))
            for k in range(2, max_legs + 1):
                legs = positive_games[:k]
                
                parlay_odds = np.prod([leg['odds'] for leg in legs])
                parlay_prob = np.prod([leg['model_prob'] for leg in legs])
                parlay_ev = (parlay_prob * parlay_odds) - 1
                parlay_kelly_frac = parlay_kelly_fraction(parlay_prob, parlay_odds)
                parlay_stake = parlay_kelly_frac * args.bankroll
                
                print(f"\n📋 {k}-Leg Parlay (Top {k} EV bets)")
                for idx, leg in enumerate(legs, 1):
                    print(f"   {idx}. {leg['game']} - {leg['prop']} (Win% {format_probability(leg['model_prob'])}, Odds {leg['odds']:.2f})")
                print(f"   → Combined decimal odds: {parlay_odds:.2f}")
                print(f"   → Combined model probability: {format_probability(parlay_prob)}")
                print(f"   → Parlay Expected Value: {format_ev(parlay_ev)}")
                if parlay_ev > 0:
                    print(f"   → Recommended stake (quarter-Kelly): {format_currency(parlay_stake)} ({parlay_kelly_frac:.2%} of bankroll)")
                else:
                    print(f"   → ⚠️ Negative EV – not recommended.")
                print("   " + "-"*80)
        else:
            print("\n❌ Not enough positive EV games to build a parlay (need at least 2).")
        
        print("\n💡 Kelly bet sizes capped at 5% of bankroll. For parlays, consider betting less than the suggested amount due to higher variance.")

if __name__ == '__main__':
    main()