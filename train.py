"""Train MVP prediction models with leave-one-year-out cross-validation."""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBRanker, XGBClassifier
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"

FEATURES = [
    "Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK",
    "TOV", "PF", "PTS",
    "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%",
    "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP",
    "WinPct", "TeamWins", "ConfSeed",
]

NARRATIVE_FEATURES = [
    "PriorMVPs", "YearsSinceLastMVP", "WinImprovement",
    "BestRecordConf", "BestRecordLeague", "AgePrime",
]

FEATURES_NARR = FEATURES + NARRATIVE_FEATURES
TARGET = "vote_share"

XGB_KWARGS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def log(msg):
    print(msg, flush=True)


def get_available(df, feat_list):
    return [f for f in feat_list if f in df.columns]


def year_result(test_df, pred_col="pred_share"):
    """Extract accuracy metrics from a test year DataFrame."""
    actual_top5 = test_df.nlargest(5, TARGET)
    pred_top5 = test_df.nlargest(5, pred_col)
    actual_mvp = actual_top5.iloc[0]["Player"]
    pred_mvp = pred_top5.iloc[0]["Player"]
    pred_top3 = pred_top5.head(3)["Player"].tolist()
    pred_top5_names = pred_top5["Player"].tolist()
    return {
        "actual_mvp": actual_mvp,
        "actual_top5": actual_top5[["Player", TARGET]].values.tolist(),
        "predicted_mvp": pred_mvp,
        "predicted_top5": pred_top5[["Player", pred_col]].values.tolist(),
        "top1_correct": actual_mvp == pred_mvp,
        "top3_correct": actual_mvp in pred_top3,
        "top5_correct": actual_mvp in pred_top5_names,
    }


# ---------------------------------------------------------------------------
# LOOCV runners — each returns (results_list, {year: test_df_with_pred_share})
# ---------------------------------------------------------------------------

def loocv_regressor(df, features, model_class=XGBRegressor, model_kwargs=None,
                    scale=False, compute_shap=False):
    model_kwargs = model_kwargs or XGB_KWARGS
    years = sorted(df["Year"].unique())
    results, year_preds = [], {}
    shap_rows = []  # list of (Player, Year, {feat: shap_val})

    for year in years:
        train, test = df[df["Year"] != year], df[df["Year"] == year]
        if test.empty:
            continue
        X_tr, y_tr = train[features].values, train[TARGET].values
        X_te = test[features].values

        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        m = model_class(**model_kwargs)
        m.fit(X_tr, y_tr)
        preds = np.clip(m.predict(X_te), 0, 1)

        tdf = test[["Player", "Tm", TARGET]].copy()
        tdf["pred_share"] = preds
        year_preds[year] = tdf

        if compute_shap and not scale:
            explainer = shap.TreeExplainer(m)
            sv = explainer.shap_values(X_te)  # shape: (n_players, n_features)
            for i, (_, row) in enumerate(test.iterrows()):
                shap_dict = {features[j]: float(sv[i, j]) for j in range(len(features))}
                shap_rows.append((row["Player"], int(year), shap_dict))

        r = year_result(tdf)
        r["Year"] = int(year)
        results.append(r)

    return results, year_preds, shap_rows


def loocv_ranker(df, features):
    years = sorted(df["Year"].unique())
    results, year_preds = [], {}

    for year in years:
        train, test = df[df["Year"] != year], df[df["Year"] == year]
        if test.empty:
            continue
        X_tr, y_tr = train[features].values, train[TARGET].values
        X_te = test[features].values
        groups = train.groupby("Year").size().sort_index().values

        m = XGBRanker(
            objective="rank:pairwise", n_estimators=200, max_depth=5,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, group=groups)
        scores = m.predict(X_te)

        tdf = test[["Player", "Tm", TARGET]].copy()
        tdf["pred_share"] = scores
        year_preds[year] = tdf

        r = year_result(tdf)
        r["Year"] = int(year)
        results.append(r)

    return results, year_preds


def loocv_classifier(df, features, top_n=1):
    years = sorted(df["Year"].unique())
    results, year_preds = [], {}

    for year in years:
        train, test = df[df["Year"] != year], df[df["Year"] == year]
        if test.empty:
            continue
        X_tr, X_te = train[features].values, test[features].values

        # Build binary labels: top_n vote-getters per training year
        labels = []
        for y in sorted(train["Year"].unique()):
            mask = train["Year"] == y
            shares = train.loc[mask, TARGET]
            thresh = shares.nlargest(top_n).min()
            labels.append((shares >= thresh).astype(int))
        y_tr = pd.concat(labels).values

        n_pos = y_tr.sum()
        spw = (len(y_tr) - n_pos) / n_pos if n_pos > 0 else 1.0

        m = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, scale_pos_weight=spw, random_state=42,
            n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_te)[:, 1]

        tdf = test[["Player", "Tm", TARGET]].copy()
        tdf["pred_share"] = probs
        year_preds[year] = tdf

        r = year_result(tdf)
        r["Year"] = int(year)
        results.append(r)

    return results, year_preds


# ---------------------------------------------------------------------------
# Post-processing heuristics
# ---------------------------------------------------------------------------

def post_process_results(results, year_preds, df):
    """Apply seed-tiebreaker and voter-fatigue adjustments."""
    new_results = []
    new_preds = {}

    for r in results:
        year = r["Year"]
        tdf = year_preds[year].copy()
        tdf = tdf.sort_values("pred_share", ascending=False).reset_index(drop=True)
        year_full = df[df["Year"] == year]

        if len(tdf) >= 2:
            p1, p2 = tdf.iloc[0]["Player"], tdf.iloc[1]["Player"]
            gap = tdf.iloc[0]["pred_share"] - tdf.iloc[1]["pred_share"]

            swapped = False
            # Seed tiebreaker
            if gap < 0.03:
                s1 = year_full.loc[year_full["Player"] == p1, "ConfSeed"]
                s2 = year_full.loc[year_full["Player"] == p2, "ConfSeed"]
                if len(s1) > 0 and len(s2) > 0 and s2.values[0] < s1.values[0]:
                    tdf.iloc[0, tdf.columns.get_loc("pred_share")] -= 0.001
                    tdf.iloc[1, tdf.columns.get_loc("pred_share")] += 0.001
                    tdf = tdf.sort_values("pred_share", ascending=False).reset_index(drop=True)
                    swapped = True

            # Voter fatigue
            if not swapped and gap < 0.05 and "PriorMVPs" in year_full.columns:
                p1, p2 = tdf.iloc[0]["Player"], tdf.iloc[1]["Player"]
                pm1 = year_full.loc[year_full["Player"] == p1, "PriorMVPs"]
                pm2 = year_full.loc[year_full["Player"] == p2, "PriorMVPs"]
                if len(pm1) > 0 and len(pm2) > 0:
                    if pm1.values[0] >= 2 and pm2.values[0] == 0:
                        tdf.iloc[0, tdf.columns.get_loc("pred_share")] -= 0.001
                        tdf.iloc[1, tdf.columns.get_loc("pred_share")] += 0.001
                        tdf = tdf.sort_values("pred_share", ascending=False).reset_index(drop=True)

        new_preds[year] = tdf
        nr = year_result(tdf)
        nr["Year"] = year
        new_results.append(nr)

    return new_results, new_preds


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def ensemble_majority_vote(models_with_preds):
    """Majority vote from model predictions. Returns results list."""
    names = list(models_with_preds.keys())
    # Model accuracies for tiebreaking
    accs = {}
    for name, (res, _) in models_with_preds.items():
        accs[name] = sum(r["top1_correct"] for r in res)

    first_res = models_with_preds[names[0]][0]
    results = []
    for i, ref in enumerate(first_res):
        year = ref["Year"]
        votes = {}
        for name in names:
            _, ypreds = models_with_preds[name]
            if year not in ypreds:
                continue
            pick = ypreds[year].nlargest(1, "pred_share").iloc[0]["Player"]
            votes.setdefault(pick, []).append(name)

        # Most votes, tiebreak by best model accuracy
        best = sorted(votes.items(),
                      key=lambda kv: (len(kv[1]), max(accs[m] for m in kv[1])),
                      reverse=True)
        winner = best[0][0]

        results.append({
            "Year": year,
            "actual_mvp": ref["actual_mvp"],
            "actual_top5": ref["actual_top5"],
            "predicted_mvp": winner,
            "predicted_top5": [],
            "top1_correct": ref["actual_mvp"] == winner,
            "top3_correct": False,
            "top5_correct": False,
        })
    return results


def ensemble_rank_avg(models_with_preds):
    """Average rank across models. Returns results list."""
    names = list(models_with_preds.keys())
    first_res = models_with_preds[names[0]][0]
    results = []

    for i, ref in enumerate(first_res):
        year = ref["Year"]
        player_ranks = {}
        for name in names:
            _, ypreds = models_with_preds[name]
            if year not in ypreds:
                continue
            tdf = ypreds[year].copy()
            tdf["rank"] = tdf["pred_share"].rank(ascending=False)
            for _, row in tdf.iterrows():
                player_ranks.setdefault(row["Player"], []).append(row["rank"])

        avg_ranks = {p: np.mean(rs) for p, rs in player_ranks.items()}
        sorted_p = sorted(avg_ranks, key=lambda p: avg_ranks[p])

        pred_mvp = sorted_p[0] if sorted_p else ""
        pred_top3 = sorted_p[:3]
        pred_top5 = sorted_p[:5]

        results.append({
            "Year": year,
            "actual_mvp": ref["actual_mvp"],
            "actual_top5": ref["actual_top5"],
            "predicted_mvp": pred_mvp,
            "predicted_top5": [[p, 1.0 / avg_ranks[p]] for p in pred_top5],
            "top1_correct": ref["actual_mvp"] == pred_mvp,
            "top3_correct": ref["actual_mvp"] in pred_top3,
            "top5_correct": ref["actual_mvp"] in pred_top5,
        })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def scores(results):
    n = len(results)
    return (sum(r["top1_correct"] for r in results),
            sum(r["top3_correct"] for r in results),
            sum(r["top5_correct"] for r in results), n)


def print_comparison(all_models):
    n = len(list(all_models.values())[0])
    log(f"\n{'='*60}")
    log(f"  Model Comparison — LOOCV ({n} seasons)")
    log(f"{'='*60}")
    log(f"  {'Model':<28}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}")
    log(f"  {'-'*52}")
    best_name, best_t1 = None, -1
    for name, res in all_models.items():
        t1, t3, t5, n = scores(res)
        t3s = f"{t3}/{n}" if t3 > 0 else "  -  "
        t5s = f"{t5}/{n}" if t5 > 0 else "  -  "
        log(f"  {name:<28}{t1}/{n:>5}{t3s:>8}{t5s:>8}")
        if t1 > best_t1:
            best_t1, best_name = t1, name
    log(f"\n  Best: {best_name} ({best_t1}/{len(all_models[best_name])})")
    return best_name


def print_verification(results, name):
    log(f"\n{'='*90}")
    log(f"  {name} — Year-by-Year")
    log(f"{'='*90}")
    log(f"{'Year':<6}{'Actual MVP':<28}{'Predicted #1':<28}{'Match':>5}")
    log(f"{'-'*90}")
    for r in results:
        m = "Y" if r["top1_correct"] else ("~" if r["top3_correct"] else "X")
        log(f"{r['Year']:<6}{r['actual_mvp']:<28}{r['predicted_mvp']:<28}{m:>5}")
    log(f"\n  Y = correct, ~ = in top 3, X = missed")


def print_misses(results, name):
    misses = [r for r in results if not r["top1_correct"]]
    if not misses:
        return
    log(f"\n{'='*90}")
    log(f"  {name} — Misses ({len(misses)})")
    log(f"{'='*90}")
    for r in misses:
        tag = " [top 3]" if r["top3_correct"] else " [MISSED]"
        log(f"\n  {r['Year']}{tag}")
        log(f"    {'Actual Top 5':<40}{'Predicted Top 5':<40}")
        log(f"    {'-'*38}  {'-'*38}")
        a, p = r["actual_top5"], r["predicted_top5"]
        for i in range(min(5, max(len(a), len(p)))):
            as_ = f"{i+1}. {a[i][0]} ({a[i][1]:.3f})" if i < len(a) else ""
            ps_ = f"{i+1}. {p[i][0]} ({p[i][1]:.3f})" if i < len(p) else ""
            log(f"    {as_:<40}{ps_:<40}")


def plot_importance(model, features, path):
    imp = model.feature_importances_
    idx = np.argsort(imp)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), imp[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx])
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost+Narrative — Top 20 Features")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    log(f"\nFeature importance saved to {path}")


# ---------------------------------------------------------------------------
# Website data export
# ---------------------------------------------------------------------------

def _safe_float(prow, col):
    """Extract a float from a single-row DataFrame slice, or 0."""
    if len(prow) == 0 or col not in prow.columns:
        return 0.0
    return round(float(prow[col].values[0]), 1)


def _player_stats(prow):
    """Pull stat fields from a matched row in the full-year DataFrame."""
    return {
        "trb": _safe_float(prow, "TRB"),
        "ast": _safe_float(prow, "AST"),
        "stl": _safe_float(prow, "STL"),
        "blk": _safe_float(prow, "BLK"),
        "fg_pct": _safe_float(prow, "FG%"),
        "three_pct": _safe_float(prow, "3P%"),
        "bpm": _safe_float(prow, "BPM"),
        "obpm": _safe_float(prow, "OBPM"),
        "ws": _safe_float(prow, "WS"),
        "ws48": _safe_float(prow, "WS/48"),
        "ows": _safe_float(prow, "OWS"),
        "per": _safe_float(prow, "PER"),
        "vorp": _safe_float(prow, "VORP"),
        "win_pct": _safe_float(prow, "WinPct"),
        "conf_seed": _safe_float(prow, "ConfSeed"),
        "team_wins": _safe_float(prow, "TeamWins"),
        "best_record_conf": _safe_float(prow, "BestRecordConf"),
    }


def export_website_data(all_models, best_name, best_year_preds, df, shap_rows=None):
    """Export JSON data for the GitHub Pages site."""
    os.makedirs("docs", exist_ok=True)
    from scipy.stats import percentileofscore

    # Load player code -> headshot URL mapping
    codes_path = os.path.join(DATA_DIR, "player_codes.json")
    player_codes = {}
    if os.path.exists(codes_path):
        with open(codes_path) as f:
            player_codes = json.load(f)
        log(f"Loaded {len(player_codes)} player headshot codes")

    # Per-year data from the best model
    years_data = []
    results = all_models[best_name]
    for r in results:
        year = r["Year"]
        tdf = best_year_preds[year]
        actual_top5 = tdf.nlargest(5, TARGET)
        pred_top5 = tdf.nlargest(5, "pred_share")
        year_full = df[df["Year"] == year]

        # Normalized predicted shares within this year (top 5 sum to 1)
        pred_top5_total = pred_top5["pred_share"].sum()

        def player_info(row, share_col, norm_total=None):
            p = row["Player"]
            prow = year_full[year_full["Player"] == p]
            pts = float(prow["PTS"].values[0]) if len(prow) > 0 else 0
            tm = str(prow["Tm"].values[0]) if len(prow) > 0 else row.get("Tm", "")
            info = {
                "player": p,
                "team": tm,
                "share": round(float(row[share_col]), 4),
                "pts": pts,
                "headshot": player_codes.get(p, ""),
            }
            info.update(_player_stats(prow))
            if norm_total and norm_total > 0:
                info["norm_share"] = round(float(row[share_col]) / norm_total, 4)
            return info

        actual_list = [player_info(row, TARGET) for _, row in actual_top5.iterrows()]
        pred_list = [player_info(row, "pred_share", pred_top5_total) for _, row in pred_top5.iterrows()]

        match = "correct" if r["top1_correct"] else ("top3" if r["top3_correct"] else "miss")
        years_data.append({
            "year": year,
            "actual_mvp": r["actual_mvp"],
            "predicted_mvp": r["predicted_mvp"],
            "match": match,
            "actual_top5": actual_list,
            "predicted_top5": pred_list,
        })

    # Deserving lists: all players with both actual and predicted share
    deserving = []
    for year, tdf in best_year_preds.items():
        year_full = df[df["Year"] == year]
        actual_mvp = tdf.nlargest(1, TARGET).iloc[0]["Player"]
        for _, row in tdf.iterrows():
            prow = year_full[year_full["Player"] == row["Player"]]
            pts = float(prow["PTS"].values[0]) if len(prow) > 0 else 0
            tm = str(prow["Tm"].values[0]) if len(prow) > 0 else str(row.get("Tm", ""))
            entry = {
                "player": row["Player"],
                "year": int(year),
                "team": tm,
                "pts": pts,
                "actual_share": round(float(row[TARGET]), 4),
                "predicted_share": round(float(row["pred_share"]), 4),
                "was_mvp": bool(row["Player"] == actual_mvp),
                "actual_mvp": actual_mvp,
                "headshot": player_codes.get(row["Player"], ""),
            }
            entry.update(_player_stats(prow))
            deserving.append(entry)

    # --- Add model rank and snub classification ---
    for year, tdf in best_year_preds.items():
        ranked = tdf.nlargest(len(tdf), "pred_share")
        rank_map = {row["Player"]: i + 1 for i, (_, row) in enumerate(ranked.iterrows())}
        actual_mvp = tdf.nlargest(1, TARGET).iloc[0]["Player"]
        mvp_pred_share = float(tdf[tdf["Player"] == actual_mvp]["pred_share"].values[0])
        for d in deserving:
            if d["year"] == int(year):
                d["model_rank"] = rank_map.get(d["player"], 999)
                if d["was_mvp"]:
                    d["snub_type"] = "winner"
                elif d["model_rank"] == 1:
                    # Model's #1 pick didn't win — true snub
                    d["snub_type"] = "snub"
                elif d["predicted_share"] > mvp_pred_share:
                    # Model predicted them higher than actual winner
                    d["snub_type"] = "snub"
                else:
                    d["snub_type"] = "deserving"

    # --- SHAP-based scatter plot axes ---
    # Split features into individual performance vs team context / winning
    TEAM_FEATURES = {
        "WinPct", "TeamWins", "ConfSeed", "OWS", "DWS", "WS", "WS/48",
    }
    # Narrative features: good for predicting votes, but don't reflect season quality
    EXCLUDE_FEATURES = {
        "PriorMVPs", "YearsSinceLastMVP", "AgePrime",
        "WinImprovement", "BestRecordConf", "BestRecordLeague",
    }
    # Everything else = individual performance (PTS, TRB, AST, PER, BPM, etc.)

    # Build lookup: (player, year) -> (team_shap, indiv_shap, top_features)
    shap_lookup = {}
    if shap_rows:
        for player, year, shap_dict in shap_rows:
            # Filter out narrative features that don't measure season quality
            filtered = {k: v for k, v in shap_dict.items() if k not in EXCLUDE_FEATURES}
            team_shap = sum(v for k, v in filtered.items() if k in TEAM_FEATURES)
            indiv_shap = sum(v for k, v in filtered.items() if k not in TEAM_FEATURES)
            # Top features by absolute magnitude
            sorted_feats = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)
            top = [[k, round(v, 4)] for k, v in sorted_feats[:8]]
            shap_lookup[(player, year)] = (team_shap, indiv_shap, top)
        log(f"SHAP values computed for {len(shap_lookup)} player-seasons")

    # Assign raw SHAP sums + top features
    for d in deserving:
        key = (d["player"], d["year"])
        if key in shap_lookup:
            t, i, top = shap_lookup[key]
        else:
            t, i, top = 0.0, 0.0, []
        d["shap_team"] = round(float(t), 5)
        d["shap_indiv"] = round(float(i), 5)
        d["shap_top"] = top

    t1, t3, t5, n = scores(results)
    data = {
        "best_model": best_name,
        "summary": {"top1": t1, "top3": t3, "top5": t5, "n": n},
        "years": years_data,
        "deserving": deserving,
    }

    path = os.path.join("docs", "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, 'item') else o)
    log(f"\nWebsite data exported to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset.csv"))
    f45 = get_available(df, FEATURES)
    f51 = get_available(df, FEATURES_NARR)
    log(f"Dataset: {len(df)} rows, {len(f45)} base features, {len(f51)} with narrative")
    log(f"Years: {int(df['Year'].min())}-{int(df['Year'].max())}")

    for f in f51:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0)

    all_models = {}
    all_preds = {}  # name -> (results, year_preds)

    # 1. Lasso (45)
    log("\n[1/7] Lasso (45)...")
    r, yp, _ = loocv_regressor(df, f45, Lasso, {"alpha": 0.001, "max_iter": 10000}, scale=True)
    all_models["Lasso (45)"] = r
    all_preds["Lasso (45)"] = (r, yp)

    # 2. XGB-Regress (45)
    log("[2/7] XGB-Regress (45)...")
    r, yp, _ = loocv_regressor(df, f45)
    all_models["XGB-Regress (45)"] = r
    all_preds["XGB-Regress (45)"] = (r, yp)

    # 3. XGB-Regress+Narr (51)
    log("[3/7] XGB-Regress+Narr (51)...")
    r_narr, yp_narr, shap_rows = loocv_regressor(df, f51, compute_shap=True)
    all_models["XGB-Regress+Narr (51)"] = r_narr
    all_preds["XGB-Regress+Narr (51)"] = (r_narr, yp_narr)

    # 4. XGB-Ranker (51)
    log("[4/7] XGB-Ranker (51)...")
    r, yp = loocv_ranker(df, f51)
    all_models["XGB-Ranker (51)"] = r
    all_preds["XGB-Ranker (51)"] = (r, yp)

    # 5. XGB-Binary (51)
    log("[5/7] XGB-Binary (51)...")
    r, yp = loocv_classifier(df, f51, top_n=1)
    all_models["XGB-Binary (51)"] = r
    all_preds["XGB-Binary (51)"] = (r, yp)

    # 6. XGB-Top3 (51)
    log("[6/7] XGB-Top3 (51)...")
    r, yp = loocv_classifier(df, f51, top_n=3)
    all_models["XGB-Top3 (51)"] = r
    all_preds["XGB-Top3 (51)"] = (r, yp)

    # 7. Post-processing on XGB-Regress+Narr
    log("[7/7] XGB+PostProcess (51)...")
    r_pp, yp_pp = post_process_results(r_narr, yp_narr, df)
    all_models["XGB+PostProcess (51)"] = r_pp

    # Ensembles (use 4 narrative models)
    log("\nBuilding ensembles...")
    ens_models = {k: all_preds[k] for k in [
        "XGB-Regress+Narr (51)", "XGB-Ranker (51)",
        "XGB-Binary (51)", "XGB-Top3 (51)",
    ]}
    all_models["Ensemble-Vote"] = ensemble_majority_vote(ens_models)
    all_models["Ensemble-AvgRank"] = ensemble_rank_avg(ens_models)

    # Comparison table
    best_name = print_comparison(all_models)

    # Detail for best model
    print_verification(all_models[best_name], best_name)
    print_misses(all_models[best_name], best_name)

    # Feature importance
    final = XGBRegressor(**XGB_KWARGS)
    final.fit(df[f51].values, df[TARGET].values)
    plot_importance(final, f51, "feature_importance.png")

    # Save comparison
    rows = []
    for name, res in all_models.items():
        t1, t3, t5, n = scores(res)
        rows.append({"Model": name, "Top1": t1, "Top3": t3, "Top5": t5, "N": n})
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "model_comparison.csv"), index=False)
    log(f"Comparison saved to {os.path.join(DATA_DIR, 'model_comparison.csv')}")

    # Export website data using best model's predictions
    best_preds_map = {
        "XGB-Regress+Narr (51)": yp_narr,
        "XGB+PostProcess (51)": yp_pp,
        "Lasso (45)": all_preds["Lasso (45)"][1],
        "XGB-Regress (45)": all_preds["XGB-Regress (45)"][1],
        "XGB-Ranker (51)": all_preds["XGB-Ranker (51)"][1],
        "XGB-Binary (51)": all_preds["XGB-Binary (51)"][1],
        "XGB-Top3 (51)": all_preds["XGB-Top3 (51)"][1],
    }
    best_yp = best_preds_map.get(best_name, yp_narr)
    export_website_data(all_models, best_name, best_yp, df, shap_rows)


if __name__ == "__main__":
    main()
