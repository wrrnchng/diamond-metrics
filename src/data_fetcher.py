import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

class MLBDataFetcher:
    def __init__(self, cache_dir='../data'):
        self.mlb_api = "https://statsapi.mlb.com/api/v1"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.master_file = f"{self.cache_dir}/master_games.csv"
        self.last_fetch_file = f"{self.cache_dir}/last_fetch_date.txt"
        self.park_factors = self._fallback_park_factors()
        
        self.team_name_to_abbr = {
            'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
            'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
            'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
            'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
            'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
            'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
            'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
            'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
            'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
            'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
            'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
            'San Diego Padres': 'SDP', 'Seattle Mariners': 'SEA',
            'San Francisco Giants': 'SFG', 'St. Louis Cardinals': 'STL',
            'Tampa Bay Rays': 'TBR', 'Texas Rangers': 'TEX',
            'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN'
        }
    
    def _get_all_teams(self):
        return list(self.team_name_to_abbr.values())
    
    def compute_park_factors_from_data(self, historical_df=None, window_years=3):
        if historical_df is None:
            if os.path.exists(self.master_file):
                historical_df = pd.read_csv(self.master_file)
                historical_df['date'] = pd.to_datetime(historical_df['date'])
                # Remove future games
                today = datetime.now().date()
                historical_df = historical_df[historical_df['date'].dt.date <= today]
            else:
                print("No cached data found; park factors set to neutral 1.0.")
                return {team: 1.0 for team in self._get_all_teams()}
        else:
            if 'date' in historical_df.columns and not pd.api.types.is_datetime64_any_dtype(historical_df['date']):
                historical_df['date'] = pd.to_datetime(historical_df['date'])
            # Remove future games
            today = datetime.now().date()
            historical_df = historical_df[historical_df['date'].dt.date <= today]
        
        if historical_df.empty:
            return {team: 1.0 for team in self._get_all_teams()}
        
        latest_date = historical_df['date'].max()
        cutoff_date = latest_date - pd.DateOffset(years=window_years)
        recent_df = historical_df[historical_df['date'] >= cutoff_date]
        all_teams = set(recent_df['home_team']).union(set(recent_df['away_team']))
        park_factors = {}
        for team in all_teams:
            home_mask = (recent_df['home_team'] == team)
            home_runs_scored = recent_df.loc[home_mask, 'home_team_runs'].sum()
            home_runs_allowed = recent_df.loc[home_mask, 'away_team_runs'].sum()
            home_total = home_runs_scored + home_runs_allowed
            away_mask = (recent_df['away_team'] == team)
            away_runs_scored = recent_df.loc[away_mask, 'away_team_runs'].sum()
            away_runs_allowed = recent_df.loc[away_mask, 'home_team_runs'].sum()
            away_total = away_runs_scored + away_runs_allowed
            if away_total > 0:
                pf = home_total / away_total
            else:
                pf = 1.0
            park_factors[team] = max(0.85, min(1.15, pf))
        print(f"Computed park factors for {len(park_factors)} teams")
        return park_factors
    
    def _fallback_park_factors(self):
        return {
            'ARI': 0.98, 'ATL': 0.99, 'BAL': 0.97, 'BOS': 1.02,
            'CHC': 0.98, 'CHW': 0.99, 'CIN': 1.01, 'CLE': 0.96,
            'COL': 1.10, 'DET': 0.97, 'HOU': 0.99, 'KCR': 0.99,
            'LAA': 1.00, 'LAD': 1.01, 'MIA': 0.97, 'MIL': 0.98,
            'MIN': 1.01, 'NYM': 0.99, 'NYY': 1.02, 'OAK': 0.98,
            'PHI': 1.00, 'PIT': 0.98, 'SDP': 0.99, 'SEA': 0.95,
            'SFG': 0.96, 'STL': 0.99, 'TBR': 0.97, 'TEX': 1.03,
            'TOR': 1.00, 'WSN': 0.99
        }
    
    def get_pitcher_era(self, pitcher_id, season=2024):
        if not pitcher_id:
            return 4.0
        url = f"{self.mlb_api}/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                stats = data.get('stats', [])
                if stats and stats[0].get('splits'):
                    era = stats[0]['splits'][0].get('stat', {}).get('era')
                    if era:
                        return float(era)
        except:
            pass
        return 4.0
    
    def get_team_name(self, team_id):
        if not team_id:
            return None
        url = f"{self.mlb_api}/teams/{team_id}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                full_name = data.get('teams', [{}])[0].get('name')
                if full_name:
                    return self.team_name_to_abbr.get(full_name)
        except:
            pass
        return None
    
    def get_game_data(self, game_pk):
        url = f"{self.mlb_api}/game/{game_pk}/linescore"
        resp = requests.get(url)
        if resp.status_code != 200:
            return None
        data = resp.json()
        home_runs = data.get('teams', {}).get('home', {}).get('runs', 0)
        away_runs = data.get('teams', {}).get('away', {}).get('runs', 0)
        game_info_url = f"{self.mlb_api}/game/{game_pk}/contextual"
        game_info_resp = requests.get(game_info_url)
        home_sp_era = 4.0
        away_sp_era = 4.0
        if game_info_resp.status_code == 200:
            game_info = game_info_resp.json()
            home_sp = game_info.get('probablePitchers', {}).get('home', {})
            away_sp = game_info.get('probablePitchers', {}).get('away', {})
            if home_sp:
                home_sp_era = self.get_pitcher_era(home_sp.get('id'))
            if away_sp:
                away_sp_era = self.get_pitcher_era(away_sp.get('id'))
        home_team_id = data.get('teams', {}).get('home', {}).get('team', {}).get('id')
        away_team_id = data.get('teams', {}).get('away', {}).get('team', {}).get('id')
        home_team = self.get_team_name(home_team_id)
        away_team = self.get_team_name(away_team_id)
        home_park_factor = self.park_factors.get(home_team, 1.0)
        away_park_factor = self.park_factors.get(away_team, 1.0)
        return {
            'game_id': game_pk,
            'date': data.get('gameDate', ''),
            'home_team': home_team,
            'away_team': away_team,
            'home_team_runs': int(home_runs),
            'away_team_runs': int(away_runs),
            'total_runs': int(home_runs) + int(away_runs),
            'home_win': 1 if home_runs > away_runs else 0,
            'run_line_cover': 1 if abs(home_runs - away_runs) > 1.5 else 0,
            'home_team_sp_era': home_sp_era,
            'away_team_sp_era': away_sp_era,
            'home_bullpen_era': 4.0,
            'away_bullpen_era': 4.0,
            'home_park_factor': home_park_factor,
            'away_park_factor': away_park_factor
        }
    
    def fetch_season_games(self, year):
        url = f"{self.mlb_api}/schedule?sportId=1&season={year}&hydrate=linescore"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"    ❌ Failed to fetch {year} (status {resp.status_code})")
            return []
        data = resp.json()
        total_games = data.get('totalGames', 0)
        print(f"    API reports {total_games} total games for {year}")
        dates = data.get('dates', [])
        if not dates:
            print(f"    No 'dates' in response. Keys: {list(data.keys())}")
            return []
        games = []
        for date in dates:
            for game in date.get('games', []):
                teams = game.get('teams', {})
                home_full = teams.get('home', {}).get('team', {}).get('name')
                away_full = teams.get('away', {}).get('team', {}).get('name')
                if not home_full or not away_full:
                    linescore = game.get('linescore', {})
                    ls_teams = linescore.get('teams', {})
                    home_full = ls_teams.get('home', {}).get('team', {}).get('name')
                    away_full = ls_teams.get('away', {}).get('team', {}).get('name')
                if not home_full or not away_full:
                    continue
                home_abbr = self.team_name_to_abbr.get(home_full)
                away_abbr = self.team_name_to_abbr.get(away_full)
                if not home_abbr or not away_abbr:
                    continue
                linescore = game.get('linescore', {})
                home_runs = linescore.get('teams', {}).get('home', {}).get('runs', 0)
                away_runs = linescore.get('teams', {}).get('away', {}).get('runs', 0)
                if home_runs is None or away_runs is None:
                    continue
                games.append({
                    'game_id': game.get('gamePk'),
                    'date': date.get('date'),
                    'home_team': home_abbr,
                    'away_team': away_abbr,
                    'home_team_runs': int(home_runs),
                    'away_team_runs': int(away_runs),
                    'total_runs': int(home_runs) + int(away_runs),
                    'home_win': 1 if home_runs > away_runs else 0,
                    'run_line_cover': 1 if abs(home_runs - away_runs) > 1.5 else 0,
                    'home_team_sp_era': 4.0,
                    'away_team_sp_era': 4.0,
                    'home_bullpen_era': 4.0,
                    'away_bullpen_era': 4.0,
                    'home_park_factor': 1.0,
                    'away_park_factor': 1.0
                })
        print(f"    Parsed {len(games)} games for {year}")
        return games
    
    def enrich_games_with_pitcher_data(self, games_df):
        print("  Enrichment is disabled by default for speed.")
        return games_df
    
    def fetch_games_in_date_range(self, start_date, end_date, enrich=False):
        start_year = start_date.year
        end_year = end_date.year
        all_games = []
        for year in range(start_year, end_year + 1):
            print(f"  Fetching {year} season...")
            season_games = self.fetch_season_games(year)
            if not season_games:
                print(f"    No games for {year}")
                continue
            filtered = []
            for g in season_games:
                game_date = datetime.strptime(g['date'], '%Y-%m-%d').date()
                if start_date <= game_date <= end_date:
                    filtered.append(g)
            all_games.extend(filtered)
            print(f"    Kept {len(filtered)} games in range.")
            time.sleep(0.2)
        if not all_games:
            return pd.DataFrame()
        df = pd.DataFrame(all_games)
        df['date'] = pd.to_datetime(df['date'])  # ensure datetime
        if enrich:
            print("  Enriching games with pitcher/park data (may take a long time)...")
            df = self.enrich_games_with_pitcher_data(df)
        else:
            print("  Skipping enrichment. Using default values for pitcher ERA and park factors.")
        return df
    
    def load_cached_data(self):
        if os.path.exists(self.master_file):
            df = pd.read_csv(self.master_file)
            df['date'] = pd.to_datetime(df['date'])
            # Remove any future games (data may have been corrupted)
            today = datetime.now().date()
            df = df[df['date'].dt.date <= today]
            print(f"Loaded {len(df)} cached games from {self.master_file}")
            return df
        return pd.DataFrame()
    
    def get_last_cache_date(self, df):
        if df.empty:
            return None
        # Return the maximum date that is not in the future (should be already filtered)
        return df['date'].max().date()
    
    def fetch_new_games_since(self, since_date, enrich=False):
        today = datetime.now().date()
        if since_date >= today:
            return pd.DataFrame()
        print(f"Fetching new games from {since_date} to {today}...")
        return self.fetch_games_in_date_range(since_date, today, enrich=enrich)
    
    def full_initial_fetch(self, start_year=2018, end_year=None, enrich=False):
        if end_year is None:
            end_year = datetime.now().year
        start_date = datetime(start_year, 1, 1).date()
        end_date = datetime(end_year, 12, 31).date()
        print(f"Fetching {start_year} to {end_year}.")
        df = self.fetch_games_in_date_range(start_date, end_date, enrich=enrich)
        if not df.empty:
            # Filter future dates again to be safe
            today = datetime.now().date()
            df = df[df['date'].dt.date <= today]
            self.park_factors = self.compute_park_factors_from_data(df)
        return df
    
    def update_cache_incremental(self, enrich=False):
        cached_df = self.load_cached_data()
        if cached_df.empty:
            print("No cache found. Performing initial fetch...")
            new_df = self.full_initial_fetch(enrich=enrich)
            if new_df.empty:
                print("No games fetched. Check your internet connection.")
                return new_df
            new_df.to_csv(self.master_file, index=False)
            with open(self.last_fetch_file, 'w') as f:
                f.write(datetime.now().isoformat())
            print(f"Initial cache saved: {len(new_df)} games.")
            self.park_factors = self.compute_park_factors_from_data(new_df)
            return new_df
        
        last_date = self.get_last_cache_date(cached_df)
        print(f"Last cached game date: {last_date}")
        new_df = self.fetch_new_games_since(last_date, enrich=enrich)
        if new_df.empty:
            print("No new games found.")
            return cached_df
        
        # Ensure both DataFrames have datetime date
        combined = pd.concat([cached_df, new_df], ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.drop_duplicates(subset=['date', 'home_team', 'away_team'])
        combined = combined.sort_values('date').reset_index(drop=True)
        # Filter future dates again
        today = datetime.now().date()
        combined = combined[combined['date'].dt.date <= today]
        combined.to_csv(self.master_file, index=False)
        with open(self.last_fetch_file, 'w') as f:
            f.write(datetime.now().isoformat())
        print(f"Added {len(new_df)} new games. Total: {len(combined)}")
        self.park_factors = self.compute_park_factors_from_data(combined)
        return combined
    
    def fetch_custom_range(self, start_date, end_date, append_to_cache=False, enrich=False):
        df = self.fetch_games_in_date_range(start_date, end_date, enrich=enrich)
        if df.empty:
            return df
        # Filter future dates
        today = datetime.now().date()
        df = df[df['date'].dt.date <= today]
        if append_to_cache:
            cached = self.load_cached_data()
            if not cached.empty:
                combined = pd.concat([cached, df], ignore_index=True)
                combined['date'] = pd.to_datetime(combined['date'])
                combined = combined.drop_duplicates(subset=['date', 'home_team', 'away_team'])
                combined = combined.sort_values('date').reset_index(drop=True)
                combined = combined[combined['date'].dt.date <= today]
                combined.to_csv(self.master_file, index=False)
                self.park_factors = self.compute_park_factors_from_data(combined)
            else:
                df.to_csv(self.master_file, index=False)
                self.park_factors = self.compute_park_factors_from_data(df)
        return df