"""Predict the current 2025-26 season MVP race using the trained model."""

import os
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scrape_data import fetch_page, parse_table, clean_player_name, CACHE_DIR

DATA_DIR = "data"
CURRENT_YEAR = 2026  # Basketball Reference convention: 2026 = 2025-26 season

FEATURES = [
    "Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK",
    "TOV", "PF", "PTS",
    "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%",
    "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP",
    "WinPct", "TeamWins", "ConfSeed",
    "PriorMVPs", "YearsSinceLastMVP", "WinImprovement",
    "BestRecordConf", "BestRecordLeague", "AgePrime",
]

TARGET = "vote_share"

# Team abbreviation mapping for standings
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def scrape_current_season():
    """Scrape current season per-game stats, advanced stats, and standings."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Per-game stats
    url = f"https://www.basketball-reference.com/leagues/NBA_{CURRENT_YEAR}_per_game.html"
    cache = os.path.join(CACHE_DIR, f"per_game_{CURRENT_YEAR}_live.html")
    # Always re-fetch current season (delete cache for live data)
    if os.path.exists(cache):
        os.remove(cache)
    html = fetch_page(url, cache)
    pg = parse_table(html, "per_game_stats")
    pg["Year"] = CURRENT_YEAR
    if "Player" in pg.columns:
        pg["Player"] = pg["Player"].apply(clean_player_name)

    # Advanced stats
    url = f"https://www.basketball-reference.com/leagues/NBA_{CURRENT_YEAR}_advanced.html"
    cache = os.path.join(CACHE_DIR, f"advanced_{CURRENT_YEAR}_live.html")
    if os.path.exists(cache):
        os.remove(cache)
    html = fetch_page(url, cache)
    adv = parse_table(html, "advanced")
    if adv.empty:
        adv = parse_table(html, "advanced_stats")
    adv["Year"] = CURRENT_YEAR
    if "Player" in adv.columns:
        adv["Player"] = adv["Player"].apply(clean_player_name)

    # Normalize "Team" -> "Tm"
    for df in [pg, adv]:
        if "Team" in df.columns and "Tm" not in df.columns:
            df.rename(columns={"Team": "Tm"}, inplace=True)

    # Handle traded players — keep combined row (TOT or 2TM/3TM/etc.)
    for df in [pg, adv]:
        if "Tm" in df.columns:
            combined_mask = df["Tm"].str.match(r"^(TOT|\dTM)$", na=False)
            combined_players = df.loc[combined_mask, ["Player", "Year"]].drop_duplicates()
            remove_mask = pd.Series(False, index=df.index)
            for _, row in combined_players.iterrows():
                player_year_mask = (df["Player"] == row["Player"]) & (df["Year"] == row["Year"])
                non_combined = player_year_mask & ~combined_mask
                remove_mask = remove_mask | non_combined
            df.drop(df.index[remove_mask], inplace=True)

    # Drop any remaining duplicates
    pg.drop_duplicates(subset=["Player", "Year"], keep="first", inplace=True)
    adv.drop_duplicates(subset=["Player", "Year"], keep="first", inplace=True)

    # Merge per-game + advanced
    pg_cols = set(pg.columns)
    adv_cols = set(adv.columns)
    shared = pg_cols & adv_cols - {"Player", "Year", "Tm", "Pos", "Age", "G", "Rk"}
    adv_drop = [c for c in shared if c in adv.columns and c not in ("Player", "Year")]
    adv_clean = adv.drop(columns=adv_drop, errors="ignore")
    current = pg.merge(adv_clean, on=["Player", "Year"], how="left", suffixes=("", "_adv"))
    drop_cols = [c for c in current.columns if c.endswith("_adv")]
    current.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Standings for win%
    url = f"https://www.basketball-reference.com/leagues/NBA_{CURRENT_YEAR}_standings.html"
    cache = os.path.join(CACHE_DIR, f"standings_{CURRENT_YEAR}_live.html")
    if os.path.exists(cache):
        os.remove(cache)
    html = fetch_page(url, cache)

    # Parse standings into East/West with wins and seed
    east_dfs = []
    west_dfs = []
    for table_id in ["divs_standings_E", "confs_standings_E"]:
        sdf = parse_table(html, table_id)
        if not sdf.empty:
            east_dfs.append(sdf)
            break
    for table_id in ["divs_standings_W", "confs_standings_W"]:
        sdf = parse_table(html, table_id)
        if not sdf.empty:
            west_dfs.append(sdf)
            break

    standings_parts = []
    for dfs, conf in [(east_dfs, "East"), (west_dfs, "West")]:
        if dfs:
            sdf = dfs[0].copy()
            cols = sdf.columns.tolist()
            team_col = cols[0]
            sdf["Team"] = sdf[team_col].str.replace(r"[*†]", "", regex=True).str.replace(r"\(\d+\)", "", regex=True).str.strip()
            sdf["Tm"] = sdf["Team"].map(TEAM_NAME_TO_ABBR)
            if "W/L%" in cols:
                sdf["WinPct"] = pd.to_numeric(sdf["W/L%"], errors="coerce")
            if "W" in cols:
                sdf["TeamWins"] = pd.to_numeric(sdf["W"], errors="coerce")
            sdf["Conference"] = conf
            sdf = sdf.dropna(subset=["Tm", "WinPct"])
            standings_parts.append(sdf)

    if standings_parts:
        standings = pd.concat(standings_parts, ignore_index=True)
        standings = standings.drop_duplicates(subset=["Tm"], keep="first")
        # Compute conference seed
        standings["ConfSeed"] = (
            standings.groupby("Conference")["WinPct"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        current = current.merge(standings[["Tm", "WinPct", "TeamWins", "ConfSeed"]], on="Tm", how="left")
        for col in ["WinPct", "TeamWins", "ConfSeed"]:
            if col in current.columns:
                current[col] = current[col].fillna(current[col].mean())
    else:
        current["WinPct"] = 0.5
        current["TeamWins"] = 41
        current["ConfSeed"] = 8

    # --- Narrative features for current season ---
    # PriorMVPs and YearsSinceLastMVP from historical data
    hist = pd.read_csv(os.path.join(DATA_DIR, "model_dataset.csv"))
    hist["vote_share"] = pd.to_numeric(hist["vote_share"], errors="coerce").fillna(0)
    mvp_history = {}
    for yr in sorted(hist["Year"].unique()):
        yr_df = hist[hist["Year"] == yr]
        if yr_df.empty:
            continue
        winner = yr_df.loc[yr_df["vote_share"].idxmax()]
        if winner["vote_share"] > 0:
            mvp_history.setdefault(winner["Player"], []).append(yr)

    prior_mvps = []
    years_since = []
    for _, row in current.iterrows():
        player = row.get("Player", "")
        wins = [y for y in mvp_history.get(player, []) if y < CURRENT_YEAR]
        prior_mvps.append(len(wins))
        years_since.append(CURRENT_YEAR - max(wins) if wins else 99)
    current["PriorMVPs"] = prior_mvps
    current["YearsSinceLastMVP"] = years_since

    # WinImprovement: current wins minus last year's wins
    prev_standings = hist[hist["Year"] == CURRENT_YEAR - 1][["Tm", "TeamWins"]].drop_duplicates("Tm")
    if not prev_standings.empty and "TeamWins" in current.columns:
        prev_map = dict(zip(prev_standings["Tm"], prev_standings["TeamWins"]))
        current["WinImprovement"] = current.apply(
            lambda r: r["TeamWins"] - prev_map.get(r["Tm"], r["TeamWins"]), axis=1)
    else:
        current["WinImprovement"] = 0.0

    # BestRecordConf and BestRecordLeague
    if "ConfSeed" in current.columns:
        current["BestRecordConf"] = (current["ConfSeed"] == 1).astype(int)
    else:
        current["BestRecordConf"] = 0
    if "WinPct" in current.columns:
        current["BestRecordLeague"] = (
            current["WinPct"] == current["WinPct"].max()
        ).astype(int)
    else:
        current["BestRecordLeague"] = 0

    # AgePrime
    if "Age" in current.columns:
        current["Age"] = pd.to_numeric(current["Age"], errors="coerce").fillna(27)
        current["AgePrime"] = np.maximum(0, 1 - np.abs(current["Age"] - 27.5) / 5)
    else:
        current["AgePrime"] = 0.5

    # Convert numeric
    for f in FEATURES:
        if f in current.columns:
            current[f] = pd.to_numeric(current[f], errors="coerce").fillna(0)

    # Filter: significant playing time
    if "G" in current.columns and "MP" in current.columns:
        current = current[(current["G"] >= 30) & (current["MP"] >= 25.0)].copy()

    return current


def main():
    # Load historical data and train on all of it
    hist = pd.read_csv(os.path.join(DATA_DIR, "model_dataset.csv"))
    features = [f for f in FEATURES if f in hist.columns]

    for f in features:
        hist[f] = pd.to_numeric(hist[f], errors="coerce").fillna(0)
    hist[TARGET] = pd.to_numeric(hist[TARGET], errors="coerce").fillna(0)

    print(f"Training on {len(hist)} historical player-seasons ({int(hist['Year'].min())}-{int(hist['Year'].max())})")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(hist[features].values, hist[TARGET].values)

    # Scrape current season
    print(f"\nScraping {CURRENT_YEAR - 1}-{str(CURRENT_YEAR)[2:]} season data...")
    current = scrape_current_season()

    # Ensure features exist
    for f in features:
        if f not in current.columns:
            current[f] = 0.0

    # Predict
    X = current[features].values
    raw_preds = np.clip(model.predict(X), 0, 1)

    # Normalize so predictions sum to ~2.6 (realistic vote share distribution)
    pred_sum = raw_preds.sum()
    if pred_sum > 0:
        preds = raw_preds * (2.6 / pred_sum)
        preds = np.clip(preds, 0, 1)
    else:
        preds = raw_preds
    current["predicted_share"] = preds

    # Rank
    current = current.sort_values("predicted_share", ascending=False)

    # Display top 15
    print(f"\n{'='*65}")
    print(f"  2025-26 NBA MVP Predictions (as of current stats)")
    print(f"{'='*65}")
    print(f"{'Rank':<6}{'Player':<25}{'Team':<6}{'PTS':<7}{'Pred Share':<12}")
    print(f"{'-'*65}")

    display_cols = ["Player", "Tm", "PTS", "predicted_share"]
    available = [c for c in display_cols if c in current.columns]

    for i, (_, row) in enumerate(current.head(15).iterrows(), 1):
        player = row.get("Player", "?")
        team = row.get("Tm", "?")
        pts = row.get("PTS", 0)
        share = row["predicted_share"]
        print(f"{i:<6}{player:<25}{team:<6}{pts:<7.1f}{share:<12.4f}")

    # Save full predictions
    out_path = os.path.join(DATA_DIR, "predictions_2026.csv")
    current.to_csv(out_path, index=False)
    print(f"\nFull predictions saved to {out_path}")


if __name__ == "__main__":
    main()
