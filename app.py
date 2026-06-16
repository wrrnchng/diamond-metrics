import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsapi
from datetime import datetime, timedelta
from datetime import timezone
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import get_todays_games, prepare_features_for_tomorrow, get_recommendations

st.set_page_config(page_title="MLB Betting Predictor", layout="wide")
st.title("⚾ MLB Betting Predictor")

def fetch_games_for_date(date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    try:
        games_data = statsapi.schedule(start_date=date_str, end_date=date_str)
        if not games_data:
            return []
        games = []
        for game in games_data:
            games.append({
                'home_team': game['home_name'],
                'away_team': game['away_name'],
                'game_id': game['game_id'],
                'date': date_str
            })
        return games
    except Exception as e:
        st.error(f"Error fetching games: {e}")
        return []

st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode:", ["Predict", "Train"])

# ========== TRAINING MODE ==========
if mode == "Train":
    st.subheader("🔄 Train Models")
    st.markdown("Fetch new games and retrain models (may take several minutes).")
    if st.button("Start Training", type="primary"):
        progress_bar = st.progress(0, text="Initializing...")
        log_area = st.empty()
        status_placeholder = st.empty()
        cmd = [sys.executable, "run.py", "--train"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output_lines = []
        step = 0
        max_steps = 9
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output_lines.append(line)
            if len(output_lines) > 30:
                output_lines = output_lines[-30:]
            log_area.code("".join(output_lines), language="bash")
            if "Fetching" in line and "season" in line:
                step += 1
                progress = min(step / max_steps, 0.9)
                progress_bar.progress(progress, text=line.strip())
            elif "Engineering features" in line:
                progress_bar.progress(0.92, text="Engineering features...")
            elif "Training models" in line:
                progress_bar.progress(0.95, text="Training models...")
            elif "Training complete" in line:
                progress_bar.progress(1.0, text="Done!")
        process.wait()
        if process.returncode == 0:
            status_placeholder.success("✅ Training completed!")
            st.balloons()
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()
        else:
            status_placeholder.error("❌ Training failed.")
    st.stop()

# ========== PREDICTION MODE ==========
@st.cache_resource
def load_models_and_data():
    models_dir = "models"
    data_dir = "data"
    required = ['total_model.pkl', 'winner_base.pkl', 'winner_calibrator.pkl',
                'runline_base.pkl', 'runline_isotonic.pkl', 'feature_sets.pkl']
    for f in required:
        if not os.path.exists(f"{models_dir}/{f}"):
            st.error(f"Missing {f}. Switch to Train mode first.")
            return None, None, None
    total = joblib.load(f"{models_dir}/total_model.pkl")
    w_base = joblib.load(f"{models_dir}/winner_base.pkl")
    w_cal = joblib.load(f"{models_dir}/winner_calibrator.pkl")
    r_base = joblib.load(f"{models_dir}/runline_base.pkl")
    r_cal = joblib.load(f"{models_dir}/runline_isotonic.pkl")
    feat = joblib.load(f"{models_dir}/feature_sets.pkl")
    hist = pd.read_csv(f"{data_dir}/master_games.csv")
    models = {
        'total': total,
        'winner_base': w_base,
        'winner_calibrator': w_cal,
        'runline_base': r_base,
        'runline_calibrator': r_cal
    }
    return models, feat, hist

models, feature_sets, historical = load_models_and_data()
if models is None:
    st.stop()

now_et = datetime.now(timezone(timedelta(hours=-4)))
default_date = now_et.date()
st.sidebar.header("Game Date")
selected_date = st.sidebar.date_input(
    "Select date",
    value=default_date,
    min_value=default_date,
    max_value=default_date + timedelta(days=7)
)
fetch_button = st.sidebar.button("Fetch Games", use_container_width=True)

if 'games_df' not in st.session_state:
    st.session_state.games_df = pd.DataFrame()
    st.session_state.selected_date = None

if fetch_button or st.session_state.selected_date != selected_date:
    st.session_state.selected_date = selected_date
    with st.spinner(f"Fetching games for {selected_date}..."):
        games = fetch_games_for_date(selected_date)
        st.session_state.games_df = pd.DataFrame(games)

games_df = st.session_state.games_df
if games_df.empty:
    st.warning(f"No games for {selected_date}.")
    st.stop()

st.subheader(f"📅 Games on {selected_date}")
selected_indices = []
for i, game in games_df.iterrows():
    if st.checkbox(f"{game['away_team']} @ {game['home_team']}", key=f"game_{i}"):
        selected_indices.append(i)

if not selected_indices:
    st.info("Select at least one game.")
    st.stop()

st.subheader("💰 Enter Odds (Decimal)")
odds_data = {}
with st.form("odds_form"):
    for idx in selected_indices:
        game = games_df.loc[idx]
        home, away = game['home_team'], game['away_team']
        st.markdown(f"### {away} @ {home}")
        col1, col2, col3 = st.columns(3)
        with col1:
            ml_h = st.text_input(f"{home} Moneyline", key=f"ml_h_{idx}", placeholder="1.68")
            ml_a = st.text_input(f"{away} Moneyline", key=f"ml_a_{idx}", placeholder="2.20")
        with col2:
            rl_h = st.text_input(f"{home} Run Line -1.5", key=f"rl_h_{idx}", placeholder="2.37")
            rl_a = st.text_input(f"{away} Run Line +1.5", key=f"rl_a_{idx}", placeholder="1.59")
        with col3:
            ou = st.text_input("Over/Under Line", key=f"ou_{idx}", placeholder="8.5")
            over = st.text_input("Over odds", key=f"over_{idx}", placeholder="1.97")
            under = st.text_input("Under odds", key=f"under_{idx}", placeholder="1.85")
        game_odds = {}
        if ml_h: game_odds['moneyline_home'] = float(ml_h)
        if ml_a: game_odds['moneyline_away'] = float(ml_a)
        if rl_h: game_odds['runline_home'] = float(rl_h)
        if rl_a: game_odds['runline_away'] = float(rl_a)
        if ou and over:
            game_odds['over_under_line'] = float(ou)
            game_odds['over_odds'] = float(over)
        if ou and under:
            game_odds['over_under_line'] = float(ou)
            game_odds['under_odds'] = float(under)
        if game_odds:
            odds_data[f"{away} @ {home}"] = game_odds
    bankroll = st.number_input("Bankroll (₱)", min_value=100.0, value=1000.0, step=100.0)
    submitted = st.form_submit_button("Get Predictions")

# Store predictions in session state to survive widget changes
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame()

if submitted and odds_data:
    selected_games = games_df.iloc[selected_indices].to_dict('records')
    features_df = prepare_features_for_tomorrow(selected_games, historical)
    recs = get_recommendations(features_df, odds_data, models, feature_sets, bankroll=bankroll)
    st.session_state.predictions = recs

recs = st.session_state.predictions

if not recs.empty:
    recs_sorted = recs.sort_values('ev', ascending=False)
    st.subheader("📈 Predicted Bets (Sorted by EV)")
    st.dataframe(
        recs_sorted[['game', 'prop', 'model_prob', 'odds', 'ev', 'kelly_bet_$']].style.format({
            'model_prob': '{:.1%}',
            'odds': '{:.2f}',
            'ev': '{:.4f}',
            'kelly_bet_$': '₱{:.2f}'
        }),
        use_container_width=True
    )
    
    # ========== FIXED PARLAY BUILDER (no slider) ==========
    st.subheader("📋 Automatic Parlays")
    positive_ev = recs[recs['ev'] > 0]
    if len(positive_ev) < 2:
        st.info("Need at least 2 positive EV bets to build a parlay.")
    else:
        top_legs = positive_ev.head(5)  # use top 5 EV bets
        for k in range(2, 6):  # 2-leg, 3-leg, 4-leg, 5-leg
            if k > len(top_legs):
                continue
            legs = top_legs.head(k)
            parlay_odds = legs['odds'].product()
            combined_prob = legs['model_prob'].product()
            ev_parlay = (combined_prob * parlay_odds) - 1
            with st.expander(f"🎯 {k}-Leg Parlay (top {k} EV bets)"):
                for j, (_, row) in enumerate(legs.iterrows(), 1):
                    st.write(f"{j}. {row['game']} - {row['prop']} (Odds: {row['odds']:.2f}, Win%: {row['model_prob']:.1%})")
                st.write(f"**Combined odds:** {parlay_odds:.2f}")
                st.write(f"**Model win probability:** {combined_prob:.2%}")
                if ev_parlay > 0:
                    st.success(f"✅ Positive EV: {ev_parlay:.4f}")
                    # Kelly stake for parlay (capped at 5%)
                    b = parlay_odds - 1
                    if b > 0:
                        kelly_full = (combined_prob * b - (1 - combined_prob)) / b
                        kelly_stake = min(kelly_full * 0.25, 0.05) * bankroll
                        st.write(f"**Recommended stake (quarter‑Kelly):** ₱{kelly_stake:.2f}")
                else:
                    st.warning(f"⚠️ Negative EV: {ev_parlay:.4f}")
                st.caption("Parlays assume independence; bet smaller than recommended.")
    
    st.subheader("Expected Value by Bet")
    st.bar_chart(recs_sorted.set_index('game')['ev'])
else:
    if submitted:
        st.warning("No positive EV bets found.")