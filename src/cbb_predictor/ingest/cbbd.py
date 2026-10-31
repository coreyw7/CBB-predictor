import pandas as pd
import requests

from cbb_predictor.config import CBBD_API_KEY, CBBD_BASE_URL, DB_PATH
from cbb_predictor.db import get_connection, write_table

SEASON = 2027
MONTHS = [("2026-10-01T00:00:00Z", "2026-11-01T00:00:00Z"),
          ("2026-11-01T00:00:00Z", "2026-12-01T00:00:00Z"),
          ("2026-12-01T00:00:00Z", "2027-01-01T00:00:00Z"),
          ("2027-01-01T00:00:00Z", "2027-02-01T00:00:00Z"),
          ("2027-02-01T00:00:00Z", "2027-03-01T00:00:00Z"),
          ("2027-03-01T00:00:00Z", "2027-04-01T00:00:00Z"),
          ("2027-04-01T00:00:00Z", "2027-05-01T00:00:00Z")]



def fetch_season_games(season: int = SEASON) -> pd.DataFrame:
    """Fetch every game for a season, one month at a time to stay under the API's 3,000-game limit."""
    frames = []

    for start, end in MONTHS:
        response = requests.get(
            f"{CBBD_BASE_URL}/games",
            headers={"Authorization": f"Bearer {CBBD_API_KEY}"},
            params={"season": season, "startDateRange": start, "endDateRange": end},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if len(data) >= 3000:
            raise ValueError(f"{start} to {end} hit the 3,000-game limit; split it into smaller ranges")
        frames.append(pd.json_normalize(data))

    games = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")
    return games


def main() -> None:
    """Fetch this season's games from CBBD and save them to DuckDB."""
    games = fetch_season_games()
    games["FetchedAtUTC"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

    with get_connection() as con:
        write_table(con, "cbbd_games", games)

    print(f"Saved {len(games)} games for season {SEASON} to {DB_PATH}.")


if __name__ == "__main__":
    main()