import numpy as np
import pandas as pd

from cbb_predictor.config import DB_PATH
from cbb_predictor.db import get_connection, write_table

LIVE_GAME_ID_OFFSET = 1_000_000


def to_team_games(cbbd_games: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Convert CBBD games into two rows per D1-vs-D1 game, matching team_games' layout."""
    ids = team_map[["cbbd_id", "TeamID"]].dropna()

    games = cbbd_games.merge(
        ids.rename(columns={"cbbd_id": "homeTeamId", "TeamID": "HomeTeamID"}),
        on="homeTeamId", how="left", validate="many_to_one",
    )
    games = games.merge(
        ids.rename(columns={"cbbd_id": "awayTeamId", "TeamID": "AwayTeamID"}),
        on="awayTeamId", how="left", validate="many_to_one",
    )
    games = games.dropna(subset=["HomeTeamID", "AwayTeamID"])

    shared = pd.DataFrame({
        "GameID": LIVE_GAME_ID_OFFSET + games["id"],
        "Season": games["season"],
        "Date": pd.to_datetime(games["startDate"], utc=True)
                  .dt.tz_convert("America/New_York")
                  .dt.tz_localize(None)
                  .dt.normalize(),
        "GameType": np.where(games["tournament"].eq("NCAA").fillna(False), "T", "R"),
        "Status": games["status"],
    })

    home = shared.assign(
        TeamID=games["HomeTeamID"],
        OppTeamID=games["AwayTeamID"],
        Score=games["homePoints"],
        OppScore=games["awayPoints"],
        Loc=np.where(games["neutralSite"], "N", "H"),
    )
    away = shared.assign(
        TeamID=games["AwayTeamID"],
        OppTeamID=games["HomeTeamID"],
        Score=games["awayPoints"],
        OppScore=games["homePoints"],
        Loc=np.where(games["neutralSite"], "N", "A"),
    )

    team_games = pd.concat([home, away], ignore_index=True)

    team_games["Win"] = (team_games["Score"] > team_games["OppScore"]).astype("Int64")

    not_final = team_games["Status"] != "final"
    team_games.loc[not_final, ["Win", "Score", "OppScore"]] = pd.NA

    for col in ["TeamID", "OppTeamID", "Score", "OppScore"]:
        team_games[col] = team_games[col].astype("Int64")
    
    return team_games


def main() -> None:
    """Build live_team_games from cbbd_games and team_map and save it to DuckDB."""
    with get_connection(read_only=True) as con:
        cbbd_games = con.sql("SELECT * FROM cbbd_games").df()
        team_map = con.sql("SELECT * FROM team_map").df()

    live = to_team_games(cbbd_games, team_map)

    with get_connection() as con:
        write_table(con, "live_team_games", live)

    print(f"Saved {len(live)} live team-game rows to {DB_PATH}.")
    print(live["Status"].value_counts().to_string())


if __name__ == "__main__":
    main()