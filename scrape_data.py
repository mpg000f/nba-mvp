"""Scrape Basketball Reference for NBA per-game stats, advanced stats, and MVP voting."""

import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR = "cache"
DATA_DIR = "data"
SEASONS = list(range(1980, 2026))  # 1980 means the 1979-80 season
DELAY = 3.5  # seconds between requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def fetch_page(url, cache_path):
    """Fetch a page, using cached HTML if available."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"  Fetching {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    time.sleep(DELAY)
    return resp.text


def parse_table(html, table_id):
    """Parse an HTML table by its id, handling Basketball Reference's commented-out tables."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": table_id})

    # Some tables are inside HTML comments — extract and re-parse
    if table is None:
        for comment in soup.find_all(string=lambda t: isinstance(t, type(soup.new_string(''))) and table_id in str(t)):
            pass
        # Try pulling from comments
        import re
        comments = soup.find_all(string=lambda text: isinstance(text, str) and table_id in text)
        for c in comments:
            comment_soup = BeautifulSoup(str(c), "lxml")
            table = comment_soup.find("table", {"id": table_id})
            if table is not None:
                break

    if table is None:
        return pd.DataFrame()

    # Extract header
    thead = table.find("thead")
    header_row = thead.find_all("tr")[-1]  # last header row
    cols = [th.get_text(strip=True) for th in header_row.find_all("th")]

    # Extract rows
    rows = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr["class"]:
            continue  # skip mid-table header repeats
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if len(cells) == len(cols):
            rows.append(cells)

    df = pd.DataFrame(rows, columns=cols)
    # Drop empty-string rows
    df = df[df.iloc[:, 0] != ""].reset_index(drop=True)
    return df


def scrape_per_game(year):
    """Scrape per-game stats for a season."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    cache = os.path.join(CACHE_DIR, f"per_game_{year}.html")
    html = fetch_page(url, cache)
    df = parse_table(html, "per_game_stats")
    if df.empty:
        print(f"  WARNING: No per-game table found for {year}")
        return df
    df["Year"] = year
    return df


def scrape_advanced(year):
    """Scrape advanced stats for a season."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    cache = os.path.join(CACHE_DIR, f"advanced_{year}.html")
    html = fetch_page(url, cache)
    # Table id is "advanced" (not "advanced_stats")
    df = parse_table(html, "advanced")
    if df.empty:
        df = parse_table(html, "advanced_stats")
    if df.empty:
        print(f"  WARNING: No advanced table found for {year}")
        return df
    df["Year"] = year
    return df


def scrape_mvp_voting(year):
    """Scrape MVP voting results for a season."""
    url = f"https://www.basketball-reference.com/awards/awards_{year}.html"
    cache = os.path.join(CACHE_DIR, f"awards_{year}.html")
    html = fetch_page(url, cache)
    df = parse_table(html, "mvp")
    if df.empty:
        # Try alternate table id
        df = parse_table(html, "mvp_NBA")
        if df.empty:
            print(f"  WARNING: No MVP table found for {year}")
            return df
    df["Year"] = year
    return df


def scrape_standings(year):
    """Scrape team standings to get win percentages."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
    cache = os.path.join(CACHE_DIR, f"standings_{year}.html")
    html = fetch_page(url, cache)

    # Try both conference tables
    dfs = []
    for table_id in ["divs_standings_E", "divs_standings_W",
                      "confs_standings_E", "confs_standings_W"]:
        df = parse_table(html, table_id)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        # Fallback: try expanded standings
        df = parse_table(html, "expanded_standings")
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print(f"  WARNING: No standings found for {year}")
        return pd.DataFrame()

    standings = pd.concat(dfs, ignore_index=True)
    standings["Year"] = year
    return standings


def clean_player_name(name):
    """Remove asterisks and other markers from player names."""
    return name.rstrip("*").strip()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    all_per_game = []
    all_advanced = []
    all_mvp = []
    all_standings = []

    for year in SEASONS:
        print(f"Scraping {year-1}-{str(year)[2:]} season...")

        pg = scrape_per_game(year)
        if not pg.empty:
            all_per_game.append(pg)

        adv = scrape_advanced(year)
        if not adv.empty:
            all_advanced.append(adv)

        mvp = scrape_mvp_voting(year)
        if not mvp.empty:
            all_mvp.append(mvp)

        standings = scrape_standings(year)
        if not standings.empty:
            all_standings.append(standings)

    # Concatenate and save
    if all_per_game:
        df_pg = pd.concat(all_per_game, ignore_index=True)
        if "Player" in df_pg.columns:
            df_pg["Player"] = df_pg["Player"].apply(clean_player_name)
        # Normalize team column: "Team" -> "Tm"
        if "Team" in df_pg.columns and "Tm" not in df_pg.columns:
            df_pg = df_pg.rename(columns={"Team": "Tm"})
        df_pg.to_csv(os.path.join(DATA_DIR, "player_stats.csv"), index=False)
        print(f"Saved player_stats.csv: {len(df_pg)} rows")

    if all_advanced:
        df_adv = pd.concat(all_advanced, ignore_index=True)
        if "Player" in df_adv.columns:
            df_adv["Player"] = df_adv["Player"].apply(clean_player_name)
        if "Team" in df_adv.columns and "Tm" not in df_adv.columns:
            df_adv = df_adv.rename(columns={"Team": "Tm"})
        df_adv.to_csv(os.path.join(DATA_DIR, "advanced_stats.csv"), index=False)
        print(f"Saved advanced_stats.csv: {len(df_adv)} rows")

    if all_mvp:
        df_mvp = pd.concat(all_mvp, ignore_index=True)
        if "Player" in df_mvp.columns:
            df_mvp["Player"] = df_mvp["Player"].apply(clean_player_name)
        df_mvp.to_csv(os.path.join(DATA_DIR, "mvp_voting.csv"), index=False)
        print(f"Saved mvp_voting.csv: {len(df_mvp)} rows")

    if all_standings:
        df_stand = pd.concat(all_standings, ignore_index=True)
        df_stand.to_csv(os.path.join(DATA_DIR, "standings.csv"), index=False)
        print(f"Saved standings.csv: {len(df_stand)} rows")

    print("Done!")


if __name__ == "__main__":
    main()
