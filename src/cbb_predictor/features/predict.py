import pandas as pd

from cbb_predictor.config import CURRENT_SEASON
from cbb_predictor.db import append_table, get_connection, write_table
from cbb_predictor.features.elo import (
    HOME_ADV,
    START_RATING,
    current_ratings,
    expected_score,
)

ELO_COLUMNS = ["GameID", "Season", "Date", "TeamID", "OppTeamID", "Loc", "Win"]
LOG_COLUMNS = ["GameID", "Date", "TeamID", "OppTeamID", "Loc",
               "TeamElo", "OppElo", "WinProb", "PredictedAtUTC"]

def predict_games(upcoming: pd.DataFrame, ratings: dict, home_adv: float = HOME_ADV) -> pd.DataFrame:
    """Add TeamElo, OppElo and WinProb to upcoming team-game rows."""
    predictions = upcoming.copy()

    team_elo = upcoming["TeamID"].map(ratings).fillna(START_RATING)
    opp_elo = upcoming["OppTeamID"].map(ratings).fillna(START_RATING)

    adv = upcoming["Loc"].map({"H": home_adv, "A": -home_adv, "N": 0})
    win_prob = expected_score(team_elo, opp_elo, adv)

    predictions["TeamElo"] = team_elo
    predictions["OppElo"] = opp_elo
    predictions["WinProb"] = win_prob

    return predictions


def main() -> None:
    """Predict every scheduled game this season and save the predictions to DuckDB."""
    with get_connection(read_only=True) as con:
        history = con.sql("SELECT * FROM team_games").df()
        live = con.sql("SELECT * FROM live_team_games").df()
        teams = con.sql("SELECT TeamID, TeamName FROM teams").df()
    
    finished = live[live["Status"] == "final"]
    scheduled = live[live["Status"] == "scheduled"]

    combined = pd.concat(
        [history[ELO_COLUMNS], finished[ELO_COLUMNS]],
        ignore_index=True,
    )

    ratings = current_ratings(combined, CURRENT_SEASON)
    predictions = predict_games(scheduled, ratings)
    predictions["PredictedAtUTC"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

    with get_connection() as con:
        write_table(con, "predictions", predictions)
        append_table(con, "prediction_log", predictions[LOG_COLUMNS])

    print_preview(predictions, teams)



def preview_table(predictions, teams):
    """One row per game on the next game day, from the favorite's side, sorted by tip-off."""
    # move everything from print_preview here, EXCEPT:
    #   - keep WinProb as a NUMBER (don't format it as "87.3%" yet)
    #   - return (next_day, table) instead of printing
    next_day = predictions["Date"].min()
    day = predictions[predictions["Date"] == next_day]

    favorites = day[day["WinProb"] >= 0.5].drop_duplicates("GameID")
    names = teams.set_index("TeamID")["TeamName"]

    tip = favorites["StartET"].dt.strftime("%-I:%M %p")
    tip = tip.where(~favorites["TimeTBD"], "TBD")

    table = pd.DataFrame({
        "Tip Off (ET)": tip,
        "Favorite": favorites["TeamID"].map(names).fillna("New team"),
        "": favorites["Loc"].map({"H": "vs", "A": "at", "N": "vs (N)"}),
        "Underdog": favorites["OppTeamID"].map(names).fillna("New team"),
        "WinProb": favorites["WinProb"],
    })
    table = table.loc[favorites.sort_values(["TimeTBD", "StartET", "WinProb"], ascending=[True, True, False]).index]

    return (next_day, table)


def print_preview(predictions, teams):
    next_day, table = preview_table(predictions, teams)

    table["WinProb"] = table["WinProb"].map("{:.1%}".format)

    print(f"\n{next_day:%A, %B %-d}: {len(table)} games")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()