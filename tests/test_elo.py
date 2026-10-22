import pandas as pd
import pytest

from cbb_predictor.features.elo import compute_elo, expected_score

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

