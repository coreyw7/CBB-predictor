import pandas as pd

from cbb_predictor.ingest.live import LIVE_GAME_ID_OFFSET, to_team_games

GAME_COLUMNS = [
    "id", "season", "startDate", "tournament", "status", "neutralSite",
    "homeTeamId", "awayTeamId", "homePoints", "awayPoints",
]


def make_cbbd_games(rows):
    """rows: list of tuples in GAME_COLUMNS order, like the raw cbbd_games table."""
    return pd.DataFrame(rows, columns=GAME_COLUMNS)


def make_team_map(rows):
    """rows: list of (cbbd_id, TeamID or None)."""
    df = pd.DataFrame(rows, columns=["cbbd_id", "TeamID"])
    df.insert(1, "cbbd_name", [f"Team {i}" for i in df["cbbd_id"]])
    df["TeamID"] = df["TeamID"].astype("Int64")
    return df


# CBBD 10 -> Duke, 20 -> UNC, 30 -> Kansas, 99 -> a non-D1 school
TEAM_MAP = make_team_map([(10, 1181), (20, 1314), (30, 1242), (99, None)])


def row(result, game_id, team_id):
    """The single row for one team in one game."""
    match = result[(result["GameID"] == LIVE_GAME_ID_OFFSET + game_id) & (result["TeamID"] == team_id)]
    assert len(match) == 1
    return match.iloc[0]


def final_game(game_id=1, home=10, away=20, home_points=80, away_points=70, neutral=False,
               start="2026-11-10T00:00:00.000Z", tournament=None):
    return (game_id, 2027, start, tournament, "final", neutral, home, away, home_points, away_points)


def scheduled_game(game_id=1, home=10, away=20, start="2026-11-10T00:00:00.000Z"):
    # CBBD stores 0-0 for games that haven't been played
    return (game_id, 2027, start, None, "scheduled", False, home, away, 0, 0)


def test_two_rows_per_game_with_offset_ids():
    result = to_team_games(make_cbbd_games([final_game(game_id=5)]), TEAM_MAP)
    assert len(result) == 2
    assert set(result["GameID"]) == {LIVE_GAME_ID_OFFSET + 5}


def test_uses_kaggle_ids_mirrored():
    result = to_team_games(make_cbbd_games([final_game()]), TEAM_MAP)
    assert row(result, 1, 1181)["OppTeamID"] == 1314
    assert row(result, 1, 1314)["OppTeamID"] == 1181


def test_games_with_non_d1_team_are_dropped():
    games = make_cbbd_games([
        final_game(game_id=1, home=10, away=99),   # non-D1 away team
        final_game(game_id=2, home=99, away=20),   # non-D1 home team
        final_game(game_id=3, home=10, away=30),   # D1 vs D1
    ])
    result = to_team_games(games, TEAM_MAP)
    assert set(result["GameID"]) == {LIVE_GAME_ID_OFFSET + 3}


def test_home_and_away_locations():
    result = to_team_games(make_cbbd_games([final_game()]), TEAM_MAP)
    assert row(result, 1, 1181)["Loc"] == "H"
    assert row(result, 1, 1314)["Loc"] == "A"


def test_neutral_site_is_neutral_for_both_teams():
    result = to_team_games(make_cbbd_games([final_game(neutral=True)]), TEAM_MAP)
    assert set(result["Loc"]) == {"N"}


def test_final_game_win_and_scores():
    result = to_team_games(make_cbbd_games([final_game(home_points=80, away_points=70)]), TEAM_MAP)
    home, away = row(result, 1, 1181), row(result, 1, 1314)
    assert (home["Score"], home["OppScore"], home["Win"]) == (80, 70, 1)
    assert (away["Score"], away["OppScore"], away["Win"]) == (70, 80, 0)


def test_away_team_can_win():
    result = to_team_games(make_cbbd_games([final_game(home_points=60, away_points=75)]), TEAM_MAP)
    assert row(result, 1, 1314)["Win"] == 1
    assert row(result, 1, 1181)["Win"] == 0


def test_scheduled_game_has_no_score_or_result():
    # The 0-0 placeholder must not turn into a result for either team
    result = to_team_games(make_cbbd_games([scheduled_game()]), TEAM_MAP)
    assert result["Win"].isna().all()
    assert result["Score"].isna().all()
    assert result["OppScore"].isna().all()


def test_in_progress_game_has_no_result():
    game = (1, 2027, "2026-11-10T00:00:00.000Z", None, "in_progress", False, 10, 20, 41, 38)
    result = to_team_games(make_cbbd_games([game]), TEAM_MAP)
    assert result["Win"].isna().all()
    assert result["Score"].isna().all()


def test_evening_game_uses_eastern_date():
    # 00:30 UTC on Nov 11 is 7:30 PM Eastern on Nov 10
    result = to_team_games(make_cbbd_games([final_game(start="2026-11-11T00:30:00.000Z")]), TEAM_MAP)
    assert set(result["Date"]) == {pd.Timestamp("2026-11-10")}


def test_eastern_date_handles_daylight_saving_time():
    # March 15, 2027 is daylight time (UTC-4): 03:30 UTC is 11:30 PM on March 14
    result = to_team_games(make_cbbd_games([final_game(start="2027-03-15T03:30:00.000Z")]), TEAM_MAP)
    assert set(result["Date"]) == {pd.Timestamp("2027-03-14")}


def test_date_is_plain_date_without_time_zone():
    result = to_team_games(make_cbbd_games([final_game(start="2026-11-11T00:30:00.000Z")]), TEAM_MAP)
    assert result["Date"].dt.tz is None
    assert (result["Date"] == result["Date"].dt.normalize()).all()


def test_ncaa_tournament_game_type():
    games = make_cbbd_games([
        final_game(game_id=1, tournament="NCAA"),
        final_game(game_id=2, tournament="NIT"),
        final_game(game_id=3, tournament=None),
    ])
    result = to_team_games(games, TEAM_MAP)
    assert row(result, 1, 1181)["GameType"] == "T"
    assert row(result, 2, 1181)["GameType"] == "R"
    assert row(result, 3, 1181)["GameType"] == "R"


def test_all_missing_tournament_column_does_not_crash():
    # DuckDB returns an all-empty column as nullable integers, which broke == "NCAA"
    games = make_cbbd_games([final_game()])
    games["tournament"] = pd.array([pd.NA], dtype="Int64")
    result = to_team_games(games, TEAM_MAP)
    assert set(result["GameType"]) == {"R"}


def test_integer_column_types():
    result = to_team_games(make_cbbd_games([final_game(), scheduled_game(game_id=2)]), TEAM_MAP)
    for col in ["TeamID", "OppTeamID", "Score", "OppScore", "Win"]:
        assert str(result[col].dtype) == "Int64", col


def test_home_and_away_counts_match():
    games = make_cbbd_games([
        final_game(game_id=1),
        final_game(game_id=2, home=20, away=30),
        final_game(game_id=3, neutral=True),
    ])
    counts = to_team_games(games, TEAM_MAP)["Loc"].value_counts()
    assert counts["H"] == counts["A"] == 2
    assert counts["N"] == 2
