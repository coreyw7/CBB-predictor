import unicodedata

import pandas as pd

from cbb_predictor.config import DB_PATH, RAW_DIR
from cbb_predictor.db import get_connection, write_table

CURRENT_SEASON = 2027

# Renamed D1 teams: normalized CBBD name -> Kaggle TeamID
TEAM_OVERRIDES = {
    "ualbany": 1107,
    "app state": 1111,
    "miami": 1274,               # CBBD's plain "Miami" is Miami (FL)
    "ut rio grande valley": 1410,
    "ul monroe": 1419,
    "st. thomas-minnesota": 1472,
    "queens university": 1474,
}

# Teams new to D1 this season, not in Kaggle: IDs above Kaggle's range
NEW_TEAMS = {"west florida": 1600}

# Current D1 teams with no CBBD games this season
EXPECTED_UNMAPPED = {1384}   # St. Francis (PA)

def normalize_name(name: str) -> str:
    """Lowercase, trim, and strip accents so names can be compared."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return name.lower().strip()


def map_team_ids(cbbd_teams: pd.DataFrame, spellings: pd.DataFrame,
                 kaggle_teams: pd.DataFrame) -> pd.DataFrame:
    """Return cbbd_id, cbbd_name, TeamID; TeamID is missing for non-D1 teams."""
    mapped = cbbd_teams.copy()
    mapped["normalized"] = mapped["cbbd_name"].map(normalize_name)

    current_d1 = kaggle_teams.loc[kaggle_teams["LastD1Season"] >= CURRENT_SEASON - 1, "TeamID"]

    # 1. Match against Kaggle's spellings
    mapped = mapped.merge(
        spellings, left_on="normalized", right_on="TeamNameSpelling",
        how="left", validate="many_to_one",
    ).drop(columns="TeamNameSpelling")

    # 2. discard matches to teams that are no longer D1
    mapped.loc[~mapped["TeamID"].isin(current_d1), "TeamID"] = pd.NA

    # 3. fill gaps with manual overrides
    mapped["TeamID"] = mapped["TeamID"].fillna(mapped["normalized"].map(TEAM_OVERRIDES))

    # 4. assign IDs to new D1 teams
    mapped["TeamID"] = mapped["TeamID"].fillna(mapped["normalized"].map(NEW_TEAMS))

    mapped["TeamID"] = mapped["TeamID"].astype("Int64")

    return mapped[["cbbd_id", "cbbd_name", "TeamID"]]


def check_mapping(mapped: pd.DataFrame, kaggle_teams: pd.DataFrame) -> None:
    """Raise if any D1 team is unmapped or any TeamID is used twice."""
    current_d1 = set(kaggle_teams.loc[kaggle_teams["LastD1Season"] >= CURRENT_SEASON - 1, "TeamID"])
    mapped_ids = set(mapped["TeamID"].dropna())

    unmapped = current_d1 - mapped_ids
    unexpected = unmapped - EXPECTED_UNMAPPED
    if unexpected:
        raise ValueError(f"Unmapped D1 teams: {sorted(unexpected)}")

    # No TeamID assigned to two CBBD teams
    d1 = mapped.dropna(subset=["TeamID"])
    dupes = d1[d1["TeamID"].duplicated(keep=False)]
    if not dupes.empty:
        raise ValueError(f"TeamIDs mapped to more than one CBBD team:\n{dupes}")


def main() -> None:
    """Map CBBD teams to Kaggle TeamIDs and save the result to DuckDB."""
    with get_connection(read_only=True) as con:
        cbbd_teams = con.sql("""
            SELECT homeTeamId AS cbbd_id, homeTeam AS cbbd_name FROM cbbd_games
            UNION
            SELECT awayTeamId, awayTeam FROM cbbd_games
        """).df()

    spellings = pd.read_csv(RAW_DIR / "MTeamSpellings.csv")
    kaggle_teams = pd.read_csv(RAW_DIR / "MTeams.csv")

    mapped = map_team_ids(cbbd_teams, spellings, kaggle_teams)
    check_mapping(mapped, kaggle_teams)

    # write mapped to DuckDB as "team_map", then print how many teams were mapped
    with get_connection() as con:
        write_table(con, "team_map", mapped)

    print(f"Saved {mapped['TeamID'].notna().sum()} teams for season {CURRENT_SEASON} to {DB_PATH}.")


if __name__ == "__main__":
    main()