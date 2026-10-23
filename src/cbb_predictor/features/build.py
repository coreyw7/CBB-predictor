from cbb_predictor.config import DB_PATH
from cbb_predictor.db import get_connection, write_table
from cbb_predictor.features.elo import compute_elo


def main() -> None:
    """Build feature tables from the cleaned data in DuckDB."""
    with get_connection(read_only=True) as con:
        team_games = con.sql("SELECT * FROM team_games").df()

    elo = compute_elo(team_games)
    with get_connection() as con:
        write_table(con, "elo", elo)
    print(f"Saved {len(elo)} Elo ratings to {DB_PATH}.")


if __name__ == "__main__":
    main()