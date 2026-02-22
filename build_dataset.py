"""Merge scraped data into a model-ready dataset."""

import os
import pandas as pd
import numpy as np

DATA_DIR = "data"

# Basketball Reference team abbreviation mapping (handles common variations)
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Charlotte Bobcats": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New Orleans Hornets": "NOH",
    "New Orleans/Oklahoma City Hornets": "NOK",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Seattle SuperSonics": "SEA",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    "Washington Bullets": "WSB", "Vancouver Grizzlies": "VAN",
    "New Jersey Nets": "NJN", "Kansas City Kings": "KCK",
    "San Diego Clippers": "SDC",
}


def load_standings():
    """Load standings and extract team win%, team wins, and conference seed."""
    path = os.path.join(DATA_DIR, "standings.csv")
    if not os.path.exists(path):
        print("WARNING: standings.csv not found, skipping team context features")
        return pd.DataFrame()

    df = pd.read_csv(path)
    cols = df.columns.tolist()

    if "Eastern Conference" not in cols or "Western Conference" not in cols:
        print("WARNING: Unexpected standings format")
        return pd.DataFrame()

    parts = []

    # Eastern teams
    east = df[df["Eastern Conference"].notna()][["Eastern Conference", "W", "L", "W/L%", "Year"]].copy()
    east.columns = ["Team", "W", "L", "WinPct", "Year"]
    east["Conference"] = "East"

    # Western teams
    west = df[df["Western Conference"].notna()][["Western Conference", "W", "L", "W/L%", "Year"]].copy()
    west.columns = ["Team", "W", "L", "WinPct", "Year"]
    west["Conference"] = "West"

    df_out = pd.concat([east, west], ignore_index=True)
    df_out["WinPct"] = pd.to_numeric(df_out["WinPct"], errors="coerce")
    df_out["W"] = pd.to_numeric(df_out["W"], errors="coerce")
    df_out["L"] = pd.to_numeric(df_out["L"], errors="coerce")

    # Clean team names
    df_out["Team"] = df_out["Team"].str.replace(r"[*†]", "", regex=True).str.strip()
    df_out["Tm"] = df_out["Team"].map(TEAM_NAME_TO_ABBR)

    # Drop unmapped rows (division headers, etc.)
    df_out = df_out.dropna(subset=["Tm", "WinPct"])
    df_out = df_out.drop_duplicates(subset=["Tm", "Year"], keep="first")

    # Compute conference seed: rank by W/L% within conference+year (1 = best)
    df_out["ConfSeed"] = (
        df_out.groupby(["Conference", "Year"])["WinPct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Rename W to TeamWins
    df_out = df_out.rename(columns={"W": "TeamWins"})

    return df_out[["Tm", "Year", "WinPct", "TeamWins", "ConfSeed"]]


def add_narrative_features(df, standings):
    """Add narrative-proxy features that capture voter behavior patterns."""
    # --- PriorMVPs and YearsSinceLastMVP ---
    # Determine MVP winner per year (highest vote_share)
    years = sorted(df["Year"].unique())
    mvp_history = {}  # player -> list of years they won MVP
    for year in years:
        year_df = df[df["Year"] == year]
        if year_df.empty or "vote_share" not in year_df.columns:
            continue
        winner_idx = year_df["vote_share"].idxmax()
        winner = year_df.loc[winner_idx, "Player"]
        if year_df.loc[winner_idx, "vote_share"] > 0:
            mvp_history.setdefault(winner, []).append(year)

    prior_mvps = []
    years_since = []
    for _, row in df.iterrows():
        player = row["Player"]
        year = row["Year"]
        wins_before = [y for y in mvp_history.get(player, []) if y < year]
        prior_mvps.append(len(wins_before))
        if wins_before:
            years_since.append(year - max(wins_before))
        else:
            years_since.append(99)
    df["PriorMVPs"] = prior_mvps
    df["YearsSinceLastMVP"] = years_since

    # --- WinImprovement ---
    # Team wins this year minus team wins last year
    if not standings.empty and "TeamWins" in standings.columns:
        standings_lookup = standings.set_index(["Tm", "Year"])["TeamWins"].to_dict()
        improvements = []
        for _, row in df.iterrows():
            tm = row.get("Tm", "")
            year = row["Year"]
            curr_wins = standings_lookup.get((tm, year), np.nan)
            prev_wins = standings_lookup.get((tm, year - 1), np.nan)
            if pd.notna(curr_wins) and pd.notna(prev_wins):
                improvements.append(curr_wins - prev_wins)
            else:
                improvements.append(0.0)
        df["WinImprovement"] = improvements
    else:
        df["WinImprovement"] = 0.0

    # --- BestRecordConf ---
    if "ConfSeed" in df.columns:
        df["BestRecordConf"] = (df["ConfSeed"] == 1).astype(int)
    else:
        df["BestRecordConf"] = 0

    # --- BestRecordLeague ---
    if "WinPct" in df.columns:
        max_winpct = df.groupby("Year")["WinPct"].transform("max")
        df["BestRecordLeague"] = (df["WinPct"] == max_winpct).astype(int)
    else:
        df["BestRecordLeague"] = 0

    # --- AgePrime ---
    if "Age" in df.columns:
        df["AgePrime"] = np.maximum(0, 1 - np.abs(df["Age"] - 27.5) / 5)
    else:
        df["AgePrime"] = 0.0

    return df


def main():
    # Load per-game stats
    pg = pd.read_csv(os.path.join(DATA_DIR, "player_stats.csv"))
    if "Team" in pg.columns and "Tm" not in pg.columns:
        pg = pg.rename(columns={"Team": "Tm"})
    print(f"Per-game stats: {len(pg)} rows, columns: {list(pg.columns[:10])}...")

    # Load advanced stats
    adv = pd.read_csv(os.path.join(DATA_DIR, "advanced_stats.csv"))
    if "Team" in adv.columns and "Tm" not in adv.columns:
        adv = adv.rename(columns={"Team": "Tm"})
    print(f"Advanced stats: {len(adv)} rows")

    # Load MVP voting
    mvp = pd.read_csv(os.path.join(DATA_DIR, "mvp_voting.csv"))
    print(f"MVP voting: {len(mvp)} rows")

    # Clean up: for traded players, keep only the combined row (TOT or 2TM/3TM/etc.)
    for df in [pg, adv]:
        if "Tm" in df.columns:
            # Combined rows are TOT (old format) or 2TM/3TM/4TM/5TM (new format)
            combined_mask = df["Tm"].str.match(r"^(TOT|\dTM)$", na=False)
            combined_players = df.loc[combined_mask, ["Player", "Year"]].drop_duplicates()

            # Remove individual team rows for players who have a combined row
            remove_mask = pd.Series(False, index=df.index)
            for _, row in combined_players.iterrows():
                player_year_mask = (df["Player"] == row["Player"]) & (df["Year"] == row["Year"])
                non_combined = player_year_mask & ~combined_mask
                remove_mask = remove_mask | non_combined

            df.drop(df.index[remove_mask], inplace=True)

    # Drop any remaining duplicates (edge cases)
    pg.drop_duplicates(subset=["Player", "Year"], keep="first", inplace=True)
    adv.drop_duplicates(subset=["Player", "Year"], keep="first", inplace=True)

    # Identify overlapping columns to avoid duplication in merge
    pg_cols = set(pg.columns)
    adv_cols = set(adv.columns)
    shared = pg_cols & adv_cols - {"Player", "Year", "Tm", "Pos", "Age", "G", "Rk"}
    # Drop shared non-key columns from advanced to avoid _x/_y
    adv_drop = [c for c in shared if c in adv.columns and c not in ("Player", "Year")]
    adv_clean = adv.drop(columns=adv_drop, errors="ignore")

    # Merge per-game + advanced on Player, Year
    merged = pg.merge(adv_clean, on=["Player", "Year"], how="left", suffixes=("", "_adv"))

    # Clean up any remaining duplicate columns
    drop_cols = [c for c in merged.columns if c.endswith("_adv")]
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    print(f"After merge (pg+adv): {len(merged)} rows, {len(merged.columns)} cols")

    # Normalize MVP voting column names
    mvp_cols_map = {}
    for c in mvp.columns:
        cl = c.lower().strip()
        if cl == "share" or cl == "vote share":
            mvp_cols_map[c] = "vote_share"
        elif cl in ("pts won", "points won"):
            mvp_cols_map[c] = "pts_won"
        elif cl in ("pts max", "points max"):
            mvp_cols_map[c] = "pts_max"
        elif cl in ("first", "1st"):
            mvp_cols_map[c] = "first_place_votes"
    mvp = mvp.rename(columns=mvp_cols_map)

    # Keep only needed MVP columns
    mvp_keep = ["Player", "Year"]
    for c in ["vote_share", "pts_won", "pts_max", "first_place_votes"]:
        if c in mvp.columns:
            mvp_keep.append(c)
    mvp_slim = mvp[mvp_keep].copy()

    if "vote_share" in mvp_slim.columns:
        mvp_slim["vote_share"] = pd.to_numeric(mvp_slim["vote_share"], errors="coerce")

    # Merge MVP voting — left join so non-MVP-candidates get NaN (then fill 0)
    dataset = merged.merge(mvp_slim, on=["Player", "Year"], how="left")
    if "vote_share" in dataset.columns:
        dataset["vote_share"] = dataset["vote_share"].fillna(0.0)
    else:
        dataset["vote_share"] = 0.0

    for c in ["pts_won", "pts_max", "first_place_votes"]:
        if c in dataset.columns:
            dataset[c] = pd.to_numeric(dataset[c], errors="coerce").fillna(0)

    # Add team context: win%, team wins, conference seed
    standings = load_standings()
    if not standings.empty:
        dataset = dataset.merge(standings, on=["Tm", "Year"], how="left")
        # Fill missing values (TOT/multi-team players) with year averages
        for col in ["WinPct", "TeamWins", "ConfSeed"]:
            if col in dataset.columns:
                avg = dataset.groupby("Year")[col].transform("mean")
                dataset[col] = dataset[col].fillna(avg)
    else:
        dataset["WinPct"] = np.nan
        dataset["TeamWins"] = np.nan
        dataset["ConfSeed"] = np.nan

    # --- Narrative-proxy features ---
    dataset = add_narrative_features(dataset, standings)
    print(f"Added narrative features: PriorMVPs, YearsSinceLastMVP, WinImprovement, BestRecordConf, BestRecordLeague, AgePrime")

    # Convert numeric columns
    numeric_cols = [
        "Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
        "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK",
        "TOV", "PF", "PTS",
        "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%",
        "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48",
        "OBPM", "DBPM", "BPM", "VORP",
        "WinPct", "TeamWins", "ConfSeed",
        "PriorMVPs", "YearsSinceLastMVP", "WinImprovement",
        "BestRecordConf", "BestRecordLeague", "AgePrime",
        "vote_share",
    ]
    for c in numeric_cols:
        if c in dataset.columns:
            dataset[c] = pd.to_numeric(dataset[c], errors="coerce")

    # Filter: GP >= 50 and MPG >= 25
    pre_filter = len(dataset)
    if "G" in dataset.columns and "MP" in dataset.columns:
        dataset = dataset[(dataset["G"] >= 50) & (dataset["MP"] >= 25.0)].copy()
    print(f"Filtered {pre_filter} -> {len(dataset)} rows (G>=50, MPG>=25)")

    # Fill remaining NaN in numeric features with 0
    for c in numeric_cols:
        if c in dataset.columns:
            dataset[c] = dataset[c].fillna(0.0)

    # Save
    out_path = os.path.join(DATA_DIR, "model_dataset.csv")
    dataset.to_csv(out_path, index=False)
    print(f"Saved {out_path}: {len(dataset)} rows, {len(dataset.columns)} columns")

    # Summary
    years = sorted(dataset["Year"].unique())
    print(f"Seasons covered: {int(min(years))}-{int(max(years))} ({len(years)} seasons)")
    mvp_winners = dataset[dataset["vote_share"] > 0]
    print(f"Players with MVP votes: {len(mvp_winners)}")


if __name__ == "__main__":
    main()
