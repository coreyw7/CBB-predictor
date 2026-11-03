import pandas as pd
import pytest

from cbb_predictor.features.elo import compute_elo, current_ratings, expected_score, run_elo

def make_games(rows):
    """rows: list of (GameID, Date, Season, TeamID, OppTeamID, Loc), winner first."""
    df = pd.DataFrame(rows, columns=["GameID", "Date", "Season", "TeamID", "OppTeamID", "Loc"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Win"] = 1
    return df

def get_elo(elo, game_id, team_id):
    """Look up one team's pre-game rating for one game."""
    return elo.loc[(elo["GameID"] == game_id) & (elo["TeamID"] == team_id), "EloPre"].iloc[0]

def test_equal_ratings_neutral_is_half():
    assert expected_score(1500, 1500) == 0.5

def test_expected_scores_sum_to_one():
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == 1

def test_home_adv_increases_expected_score():
    assert expected_score(1500, 1500, home_adv=100) > 0.5
    assert expected_score(1500, 1500, home_adv=-100) < 0.5

def test_zero_sum():
    games = make_games([
        (0, "2025-01-01", 2025, 1, 2, "H"),
        (1, "2025-01-02", 2025, 1, 2, "N"),
    ])
    elo = compute_elo(games)
    game_1 = elo[elo['GameID'] == 1]
    assert game_1["EloPre"].iloc[0] != 1500
    assert game_1["EloPre"].sum() == pytest.approx(3000)

def test_pre_game_rating_ignores_results():
    team_1_wins = make_games([
        (0, "2024-11-10", 2025, 1, 2, "N"),
        (1, "2024-11-15", 2025, 1, 2, "N"),
    ])
    team_2_wins = make_games([
        (0, "2024-11-10", 2025, 2, 1, "N"),   # flipped: team 2 is the winner
        (1, "2024-11-15", 2025, 1, 2, "N"),
    ])
    elo_a = compute_elo(team_1_wins)
    elo_b = compute_elo(team_2_wins)

    for team in [1, 2]:
        assert get_elo(elo_a, 0, team) == get_elo(elo_b, 0, team)

    assert get_elo(elo_a, 1, 1) > get_elo(elo_b, 1, 1)


# One neutral-site 2026 game: equal teams, so the winner gains exactly K / 2 = 25
SEASON_2026 = [(0, "2026-01-10", 2026, 1, 2, "N")]


def test_run_elo_returns_final_ratings_and_last_season():
    pre_game, ratings, last_season = run_elo(make_games(SEASON_2026))
    assert len(pre_game) == 2
    assert ratings == {1: pytest.approx(1525), 2: pytest.approx(1475)}
    assert last_season == 2026


def test_compute_elo_matches_run_elo():
    games = make_games(SEASON_2026 + [(1, "2026-01-15", 2026, 2, 1, "H")])
    pre_game, _, _ = run_elo(games)
    pd.testing.assert_frame_equal(compute_elo(games), pre_game)


def test_current_ratings_same_season_not_regressed():
    ratings = current_ratings(make_games(SEASON_2026), season=2026)
    assert ratings[1] == pytest.approx(1525)


def test_current_ratings_regresses_before_new_season_starts():
    # No 2027 games yet: ratings must still be pulled 10% toward 1500
    ratings = current_ratings(make_games(SEASON_2026), season=2027)
    assert ratings[1] == pytest.approx(1522.5)
    assert ratings[2] == pytest.approx(1477.5)


def test_current_ratings_does_not_regress_twice():
    # A 2027 game between other teams triggers the regression inside the loop
    games = make_games(SEASON_2026 + [(1, "2026-11-03", 2027, 3, 4, "N")])
    ratings = current_ratings(games, season=2027)
    assert ratings[1] == pytest.approx(1522.5)


def test_current_ratings_include_new_season_results():
    games = make_games(SEASON_2026 + [(1, "2026-11-03", 2027, 1, 3, "N")])
    ratings = current_ratings(games, season=2027)
    # Team 1 starts 2027 at 1522.5 and beats a new 1500 team, so it gains
    assert ratings[1] > 1522.5
    assert ratings[3] < 1500
