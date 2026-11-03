import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from cbb_predictor.db import get_connection

RESULT_COLUMNS = ["GameID", "TeamID", "Win", "Score", "OppScore"]


def latest_predictions(log: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent logged prediction for each team in each game."""
    return log.sort_values("PredictedAtUTC").drop_duplicates(["GameID", "TeamID"], keep="last")


def score_games(latest: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join each final game's result to its latest prediction."""
    scored = latest.merge(
        results[RESULT_COLUMNS],
        on=["GameID", "TeamID"],
        how="inner",
        validate="one_to_one",
    )

    scored["Win"] = scored["Win"].astype(int)
    return scored


def summarize(scored: pd.DataFrame) -> dict:
    """Games, accuracy, log loss and Brier score for a set of scored rows."""
    games = scored["GameID"].nunique()
    accuracy = ((scored["WinProb"] >= 0.5) == scored["Win"]).mean()
    logloss = log_loss(scored["Win"], scored["WinProb"])
    brier = brier_score_loss(scored["Win"], scored["WinProb"])

    return {
        "Games": games,
        "Accuracy": accuracy,
        "Log Loss": logloss,
        "Brier": brier,
    }


def main() -> None:
    with get_connection(read_only=True) as con:
        prediction_log = con.sql("SELECT * FROM prediction_log").df()
        live_team_games = con.sql("SELECT * FROM live_team_games").df()
        teams = con.sql("SELECT * FROM teams").df()

    results = live_team_games[live_team_games["Status"] == "final"]
    latest = latest_predictions(prediction_log)
    scored = score_games(latest, results)

    if scored.empty:
        print("No finished games to score yet")
        return

    missing = set(results["GameID"]) - set(scored["GameID"])
    print(f"Games not predicted: {len(missing)}")

    last_night = scored["Date"].max()

    night = scored[scored["Date"] == last_night]

    summary = pd.DataFrame({
        "Last night": summarize(night),
        "Season so far": summarize(scored),
    }).T

    summary["Games"] = summary["Games"].astype(int)
    summary["Accuracy"] = summary["Accuracy"].map("{:.1%}".format)
    summary["Log Loss"] = summary["Log Loss"].map("{:.3f}".format)
    summary["Brier"] = summary["Brier"].map("{:.3f}".format)

    print(f"\n{last_night:%A, %B} {last_night.day}: summary of results")
    print(summary.to_string())

    winners = night[night["Win"] == 1]
    upsets = winners[winners["WinProb"] < 0.5]
    upsets = upsets.sort_values("WinProb").head()

    names = teams.set_index("TeamID")["TeamName"]
    upsets_table = pd.DataFrame({
        "Winner": upsets["TeamID"].map(names).fillna("New team"),
        "": upsets["Loc"].map({"H": "vs", "A": "at", "N": "vs (N)"}),
        "Loser": upsets["OppTeamID"].map(names).fillna("New team"),
        "WinProb": upsets["WinProb"].map("{:.1%}".format),
        "Score": upsets["Score"].astype(str) + "-" + upsets["OppScore"].astype(str),
    })

    if upsets.empty:
        print("No upsets")
    else:
        print(f"\n{last_night:%A, %B} {last_night.day}: top {len(upsets_table)} upsets")
        print(upsets_table.to_string(index=False))


if __name__ == "__main__":
    main()