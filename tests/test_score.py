import math

import pandas as pd
import pytest

from cbb_predictor import db
from cbb_predictor.features import score
from cbb_predictor.features.score import latest_predictions, score_games, summarize


def make_log(rows):
    """rows: list of (GameID, TeamID, WinProb, PredictedAtUTC)."""
    df = pd.DataFrame(rows, columns=["GameID", "TeamID", "WinProb", "PredictedAtUTC"])
    df["PredictedAtUTC"] = pd.to_datetime(df["PredictedAtUTC"])
    return df


def make_results(rows, status="final"):
    """rows: list of (GameID, TeamID, Win, Score, OppScore), like live_team_games."""
    df = pd.DataFrame(rows, columns=["GameID", "TeamID", "Win", "Score", "OppScore"])
    df["Win"] = df["Win"].astype("Int64")
    df["Date"] = pd.Timestamp("2026-11-02")   # extra column shared with the log
    df["Status"] = status
    return df


def prob(df, game_id, team_id):
    match = df[(df["GameID"] == game_id) & (df["TeamID"] == team_id)]
    assert len(match) == 1
    return match["WinProb"].iloc[0]


# --- latest_predictions ---------------------------------------------------------

def test_latest_keeps_most_recent_run():
    log = make_log([
        (7, 1, 0.95, "2026-11-02 19:00"), (7, 2, 0.05, "2026-11-02 19:00"),
        (7, 1, 0.96, "2026-11-02 23:00"), (7, 2, 0.04, "2026-11-02 23:00"),
    ])
    latest = latest_predictions(log)
    assert len(latest) == 2
    assert prob(latest, 7, 1) == 0.96
    assert prob(latest, 7, 2) == 0.04


def test_latest_does_not_depend_on_row_order():
    # The newest run is written first: sorting must still pick it
    log = make_log([
        (7, 1, 0.96, "2026-11-02 23:00"), (7, 2, 0.04, "2026-11-02 23:00"),
        (7, 1, 0.95, "2026-11-02 19:00"), (7, 2, 0.05, "2026-11-02 19:00"),
    ])
    latest = latest_predictions(log)
    assert prob(latest, 7, 1) == 0.96


def test_latest_keeps_both_teams_of_each_game():
    log = make_log([
        (7, 1, 0.9, "2026-11-02 19:00"), (7, 2, 0.1, "2026-11-02 19:00"),
        (8, 3, 0.6, "2026-11-02 19:00"), (8, 4, 0.4, "2026-11-02 19:00"),
    ])
    latest = latest_predictions(log)
    assert set(zip(latest["GameID"], latest["TeamID"])) == {(7, 1), (7, 2), (8, 3), (8, 4)}


def test_latest_keeps_game_only_predicted_in_earlier_run():
    # Game 8 started before the second run, so only the first run predicted it
    log = make_log([
        (8, 3, 0.6, "2026-11-02 15:00"), (8, 4, 0.4, "2026-11-02 15:00"),
        (7, 1, 0.9, "2026-11-02 23:00"), (7, 2, 0.1, "2026-11-02 23:00"),
    ])
    latest = latest_predictions(log)
    assert prob(latest, 8, 3) == 0.6


def test_latest_does_not_modify_input():
    log = make_log([(7, 1, 0.95, "2026-11-02 19:00"), (7, 1, 0.96, "2026-11-02 23:00")])
    latest_predictions(log)
    assert len(log) == 2


# --- score_games ----------------------------------------------------------------

LATEST = make_log([
    (7, 1, 0.8, "2026-11-02 19:00"), (7, 2, 0.2, "2026-11-02 19:00"),   # played
    (8, 3, 0.6, "2026-11-02 19:00"), (8, 4, 0.4, "2026-11-02 19:00"),   # not played yet
]).assign(Date=pd.Timestamp("2026-11-02"))
RESULTS = make_results([
    (5, 9, 1, 70, 60), (5, 10, 0, 60, 70),   # played but never predicted
    (7, 1, 1, 81, 64), (7, 2, 0, 64, 81),
])


def test_score_keeps_only_games_with_prediction_and_result():
    scored = score_games(LATEST, RESULTS)
    assert set(scored["GameID"]) == {7}
    assert len(scored) == 2


def test_score_attaches_results_to_the_right_team():
    scored = score_games(LATEST, RESULTS).set_index("TeamID")
    assert (scored.loc[1, "Win"], scored.loc[1, "Score"], scored.loc[1, "OppScore"]) == (1, 81, 64)
    assert (scored.loc[2, "Win"], scored.loc[2, "Score"], scored.loc[2, "OppScore"]) == (0, 64, 81)


def test_score_does_not_duplicate_shared_columns():
    # Only RESULT_COLUMNS come from results, so no Date_x / Date_y
    scored = score_games(LATEST, RESULTS)
    assert "Date" in scored.columns
    assert not any(col.endswith(("_x", "_y")) for col in scored.columns)
    assert "Status" not in scored.columns


def test_score_win_is_plain_int():
    scored = score_games(LATEST, RESULTS)
    assert scored["Win"].dtype == "int64"


def test_score_raises_on_duplicate_predictions():
    duplicated = pd.concat([LATEST, LATEST.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        score_games(duplicated, RESULTS)


# --- summarize ------------------------------------------------------------------

def scored_rows(games):
    """games: list of (GameID, favorite's WinProb, favorite won?) -> both mirrored rows."""
    rows = []
    for game_id, p, fav_won in games:
        rows.append({"GameID": game_id, "WinProb": p, "Win": int(fav_won)})
        rows.append({"GameID": game_id, "WinProb": 1 - p, "Win": int(not fav_won)})
    return pd.DataFrame(rows)


def test_summarize_counts_games_not_rows():
    assert summarize(scored_rows([(1, 0.8, True), (2, 0.7, True)]))["Games"] == 2


def test_summarize_accuracy():
    result = summarize(scored_rows([(1, 0.8, True), (2, 0.7, True), (3, 0.6, False), (4, 0.9, True)]))
    assert result["Accuracy"] == pytest.approx(0.75)


def test_summarize_log_loss_and_brier_match_hand_calculation():
    result = summarize(scored_rows([(1, 0.8, True), (2, 0.6, False)]))
    assert result["Log Loss"] == pytest.approx((-math.log(0.8) - math.log(0.4)) / 2)
    assert result["Brier"] == pytest.approx((0.2 ** 2 + 0.6 ** 2) / 2)


def test_summarize_single_game_works():
    # Both mirrored rows give log loss one 1 and one 0, so one game is enough
    result = summarize(scored_rows([(1, 0.8, True)]))
    assert result["Log Loss"] == pytest.approx(-math.log(0.8))


def test_summarize_upset_counts_as_miss():
    result = summarize(scored_rows([(1, 0.9, False)]))
    assert result["Accuracy"] == 0


# --- main -----------------------------------------------------------------------

@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A tiny database: game 7 predicted and won by the underdog, game 5 never predicted."""
    db_path = tmp_path / "test.duckdb"
    log = LATEST.assign(OppTeamID=[2, 1, 4, 3], Loc=["H", "A", "N", "N"])
    live = pd.concat([
        RESULTS.assign(OppTeamID=[10, 9, 2, 1], Loc=["H", "A", "H", "A"]),
        make_results([(8, 3, None, None, None), (8, 4, None, None, None)], status="scheduled")
        .assign(OppTeamID=[4, 3], Loc=["N", "N"]),
    ], ignore_index=True)
    # Make game 7 an upset: team 2 (20%) beat team 1
    game_7 = live["GameID"] == 7
    live.loc[game_7, "Win"] = pd.array([0, 1], dtype="Int64")
    live.loc[game_7, ["Score", "OppScore"]] = [[64, 81], [81, 64]]
    teams = pd.DataFrame({"TeamID": [1, 2, 3, 4], "TeamName": ["Duke", "Army", "BYU", "Ohio St"]})

    with db.get_connection(db_path) as con:
        db.write_table(con, "prediction_log", log)
        db.write_table(con, "live_team_games", live)
        db.write_table(con, "teams", teams)

    monkeypatch.setattr(
        score, "get_connection",
        lambda read_only=False: db.get_connection(db_path, read_only=read_only),
    )
    return db_path


def test_main_reports_missing_games_and_upsets(test_db, capsys):
    score.main()
    out = capsys.readouterr().out
    assert "Games not predicted: 1" in out
    assert "Army" in out and "20.0%" in out
    assert out.index("Army") < out.rindex("Duke")   # winner listed before loser


def test_main_with_nothing_to_score_does_not_crash(test_db, capsys):
    # Only the unpredicted game 5 is final: predictions exist, but none can be scored yet
    with db.get_connection(test_db) as con:
        con.execute("UPDATE live_team_games SET Status = 'scheduled' WHERE GameID = 7")
    score.main()
    assert "No finished games to score yet" in capsys.readouterr().out
