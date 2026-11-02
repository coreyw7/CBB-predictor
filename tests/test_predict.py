import pandas as pd
import pytest

from cbb_predictor import db
from cbb_predictor.features import predict
from cbb_predictor.features.elo import START_RATING
from cbb_predictor.features.predict import LOG_COLUMNS, main, predict_games, print_preview


def make_upcoming(rows):
    """rows: list of (GameID, Date, StartET, TimeTBD, TeamID, OppTeamID, Loc), one row per team."""
    df = pd.DataFrame(rows, columns=["GameID", "Date", "StartET", "TimeTBD", "TeamID", "OppTeamID", "Loc"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["StartET"] = pd.to_datetime(df["StartET"])
    return df


def game(game_id, team, opp, loc="H", date="2026-11-10", start="2026-11-10 19:00", tbd=False):
    """Both rows of one game; loc is from `team`'s point of view."""
    flip = {"H": "A", "A": "H", "N": "N"}[loc]
    return [
        (game_id, date, start, tbd, team, opp, loc),
        (game_id, date, start, tbd, opp, team, flip),
    ]


def prob(predictions, game_id, team_id):
    match = predictions[(predictions["GameID"] == game_id) & (predictions["TeamID"] == team_id)]
    assert len(match) == 1
    return match["WinProb"].iloc[0]


TEAMS = pd.DataFrame({"TeamID": [1, 2, 3, 4], "TeamName": ["Duke", "Army", "BYU", "Ohio St"]})


def test_adds_rating_and_probability_columns():
    upcoming = make_upcoming(game(1, 1, 2))
    predictions = predict_games(upcoming, {1: 1600, 2: 1500})
    assert list(predictions.columns[-3:]) == ["TeamElo", "OppElo", "WinProb"]
    assert len(predictions) == len(upcoming)


def test_input_is_not_modified():
    upcoming = make_upcoming(game(1, 1, 2))
    columns = list(upcoming.columns)
    predict_games(upcoming, {1: 1600, 2: 1500})
    assert list(upcoming.columns) == columns


def test_uses_team_and_opponent_ratings():
    predictions = predict_games(make_upcoming(game(1, 1, 2, loc="N")), {1: 1600, 2: 1500})
    row = predictions[predictions["TeamID"] == 1].iloc[0]
    assert (row["TeamElo"], row["OppElo"]) == (1600, 1500)
    assert prob(predictions, 1, 1) == pytest.approx(1 / (1 + 10 ** (-100 / 400)))


def test_unrated_team_gets_start_rating():
    predictions = predict_games(make_upcoming(game(1, 1, 1600, loc="N")), {1: 1600})
    new_team = predictions[predictions["TeamID"] == 1600].iloc[0]
    assert new_team["TeamElo"] == START_RATING


def test_mirrored_rows_sum_to_one():
    upcoming = make_upcoming(game(1, 1, 2) + game(2, 3, 4, loc="A"))
    predictions = predict_games(upcoming, {1: 1700, 2: 1450, 3: 1520, 4: 1600})
    for game_id, team, opp in [(1, 1, 2), (2, 3, 4)]:
        assert prob(predictions, game_id, team) + prob(predictions, game_id, opp) == pytest.approx(1)


def test_home_advantage_favors_home_team():
    predictions = predict_games(make_upcoming(game(1, 1, 2, loc="H")), {1: 1500, 2: 1500})
    assert prob(predictions, 1, 1) > 0.5
    assert prob(predictions, 1, 2) < 0.5


def test_neutral_site_equal_teams_is_even():
    predictions = predict_games(make_upcoming(game(1, 1, 2, loc="N")), {1: 1500, 2: 1500})
    assert prob(predictions, 1, 1) == pytest.approx(0.5)


def test_home_adv_parameter_is_used():
    predictions = predict_games(make_upcoming(game(1, 1, 2, loc="H")), {1: 1500, 2: 1500}, home_adv=0)
    assert prob(predictions, 1, 1) == pytest.approx(0.5)


def preview_output(capsys, rows, ratings):
    print_preview(predict_games(make_upcoming(rows), ratings), TEAMS)
    return capsys.readouterr().out


def test_preview_shows_one_line_per_game_from_favorite_side(capsys):
    out = preview_output(capsys, game(1, 1, 2), {1: 1800, 2: 1400})
    assert "1 games" in out
    assert "Duke" in out and "Army" in out
    assert out.index("Duke") < out.index("Army")   # favorite listed first
    assert out.count("Duke") == 1


def test_preview_only_shows_next_game_day(capsys):
    rows = game(1, 1, 2, date="2026-11-10") + game(2, 3, 4, date="2026-11-11", start="2026-11-11 19:00")
    out = preview_output(capsys, rows, {1: 1600, 2: 1500, 3: 1600, 4: 1500})
    assert "Tuesday, November 10" in out
    assert "BYU" not in out


def test_preview_shows_tbd_and_sorts_by_tip_time(capsys):
    rows = (
        game(1, 1, 2, start="2026-11-10 00:00", tbd=True)
        + game(2, 3, 4, start="2026-11-10 21:00")
    )
    out = preview_output(capsys, rows, {1: 1600, 2: 1500, 3: 1600, 4: 1500})
    assert "TBD" in out
    assert "9:00 PM" in out
    assert "12:00 AM" not in out
    assert out.index("BYU") < out.index("Duke")    # known times before TBD


def test_preview_names_new_teams(capsys):
    out = preview_output(capsys, game(1, 1, 1600), {1: 1700})
    assert "New team" in out


def test_preview_lists_favorite_even_when_underdog_row_comes_first(capsys):
    # Army's row is first in the data, but Duke is the favorite
    out = preview_output(capsys, game(1, 2, 1), {1: 1800, 2: 1400})
    assert out.index("Duke") < out.index("Army")
    assert "at" in out.split("Duke")[1].split("Army")[0]   # Duke is the road team here


# --- main(): saving and logging ---------------------------------------------

def seed_database(db_path):
    """A tiny database: one 2026 game, one finished 2027 game and one scheduled 2027 game."""
    history = pd.DataFrame({
        "GameID": [1, 1], "Season": [2026, 2026],
        "Date": pd.to_datetime(["2026-01-10"] * 2),
        "TeamID": [1, 2], "OppTeamID": [2, 1], "Loc": ["N", "N"], "Win": [1, 0],
    })
    live = pd.DataFrame({
        "GameID": [1_000_001, 1_000_001, 1_000_002, 1_000_002],
        "Season": [2027] * 4,
        "Date": pd.to_datetime(["2026-11-03"] * 2 + ["2026-11-10"] * 2),
        "StartET": pd.to_datetime(["2026-11-03 19:00"] * 2 + ["2026-11-10 19:00"] * 2),
        "TimeTBD": [False] * 4,
        "TeamID": [3, 4, 1, 2], "OppTeamID": [4, 3, 2, 1],
        "Loc": ["H", "A", "H", "A"],
        "Win": pd.array([1, 0, None, None], dtype="Int64"),
        "Status": ["final", "final", "scheduled", "scheduled"],
    })
    with db.get_connection(db_path) as con:
        db.write_table(con, "team_games", history)
        db.write_table(con, "live_team_games", live)
        db.write_table(con, "teams", TEAMS)


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Point main() at a seeded temporary database instead of the real one."""
    db_path = tmp_path / "test.duckdb"
    seed_database(db_path)
    monkeypatch.setattr(
        predict, "get_connection",
        lambda read_only=False: db.get_connection(db_path, read_only=read_only),
    )
    return db_path


def read_table(db_path, name):
    with db.get_connection(db_path, read_only=True) as con:
        return con.sql(f"SELECT * FROM {name}").df()


def test_main_predicts_only_scheduled_games(test_db, capsys):
    main()
    predictions = read_table(test_db, "predictions")
    assert set(predictions["GameID"]) == {1_000_002}
    assert len(predictions) == 2


def test_main_logs_predictions_with_log_columns(test_db, capsys):
    main()
    log = read_table(test_db, "prediction_log")
    assert list(log.columns) == LOG_COLUMNS
    assert len(log) == 2
    assert log["WinProb"].sum() == pytest.approx(1)


def test_main_log_matches_saved_predictions(test_db, capsys):
    main()
    keys = ["GameID", "TeamID"]
    predictions = read_table(test_db, "predictions").sort_values(keys)
    log = read_table(test_db, "prediction_log").sort_values(keys)
    assert log["WinProb"].tolist() == pytest.approx(predictions["WinProb"].tolist())


def test_second_run_replaces_predictions_but_appends_log(test_db, capsys):
    main()
    main()
    assert len(read_table(test_db, "predictions")) == 2
    log = read_table(test_db, "prediction_log")
    assert len(log) == 4
    assert log["PredictedAtUTC"].nunique() == 2
