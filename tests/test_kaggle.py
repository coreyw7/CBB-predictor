import pandas as pd

from cbb_predictor.ingest.kaggle import to_team_games


def make_games():
    """Two fake games: a home win and a neutral-site win."""
    return pd.DataFrame({
        "GameID": [0, 1],
        "Season": [2025, 2025],
        "DayNum": [10, 12],
        "WTeamID": [1, 3], "WScore": [80, 70],
        "LTeamID": [2, 4], "LScore": [70, 65],
        "WLoc": ["H", "N"],
        "NumOT": [0, 0],
        "WFGA": [60, 55], "WOR": [10, 8], "WTO": [12, 11], "WFTA": [20, 15],
        "LFGA": [58, 57], "LOR": [9, 10], "LTO": [14, 13], "LFTA": [18, 16],
    })


def test_reshape_doubles_rows():
    result = to_team_games(make_games())
    assert len(result) == 4

def test_win_rate_is_half():
    results = to_team_games(make_games())
    assert results['Win'].mean() == 0.5

def test_loc_is_flipped():
    results = to_team_games(make_games())
    assert results.loc[results['TeamID'] == 2, 'Loc'].iloc[0] == 'A'
    assert results.loc[results['TeamID'] == 1, 'Loc'].iloc[0] == 'H'

def test_neutral_stays_neutral():
    results = to_team_games(make_games())
    assert results.loc[results['TeamID'] == 3, 'Loc'].iloc[0] == 'N'
    assert results.loc[results['TeamID'] == 4, 'Loc'].iloc[0] == 'N'

def test_opponent_columns_mirror():
    results = to_team_games(make_games())

    # Every column that has an Opp twin, e.g. Score/OppScore, TeamID/OppTeamID
    stats = [col for col in results.columns if 'Opp' + col in results.columns]

    # Each team's own stats, renamed to how its opponent's row labels them
    actual = results[['GameID'] + stats].rename(columns={col: 'Opp' + col for col in stats})

    # Self-join: line up every row with its opponent's real row in the same game
    paired = results.merge(actual, on=['GameID', 'OppTeamID'], suffixes=('', '_actual'))

    # Every row should find exactly one opponent
    assert len(paired) == len(results)

    for col in stats:
        if col == 'TeamID':
            continue  # already matched by the join
        opp_col = 'Opp' + col
        assert (paired[opp_col] == paired[opp_col + '_actual']).all(), f"{opp_col} doesn't match the opponent's {col}"

