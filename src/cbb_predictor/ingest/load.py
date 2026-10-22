# src/cbb_predictor/ingest/load.py
import pandas as pd

from cbb_predictor.config import RAW_DIR, DB_PATH
from cbb_predictor.db import get_connection, write_table
from cbb_predictor.ingest.kaggle import build_team_games


def main() -> None:
    """Build all tables from the raw Kaggle files and save them to DuckDB."""
    team_games = build_team_games(RAW_DIR)
    seasons = pd.read_csv(RAW_DIR / "MSeasons.csv")
    teams = pd.read_csv(RAW_DIR / "MTeams.csv")
    with get_connection() as con:
        write_table(con, "team_games", team_games)
        write_table(con, "seasons", seasons)
        write_table(con, "teams", teams)
    print(f"Saved {len(team_games)} team games to {DB_PATH}.")
    print(f"Saved {len(seasons)} seasons and {len(teams)} teams to the same database.")


if __name__ == "__main__":
    main()