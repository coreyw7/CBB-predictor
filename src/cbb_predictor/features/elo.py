import pandas as pd

K = 50
HOME_ADV = 100
KEEP = 0.90
START_RATING = 1500

def expected_score(rating_a: float, rating_b: float, home_adv: float = 0) -> float:
    """Probability that team A beats team B."""
    expected = 1 / (1 + 10 ** (-(rating_a - rating_b + home_adv) / 400))
    return expected


def run_elo(games, k=K, home_adv=HOME_ADV, keep=KEEP):
    """Run Elo over games in date order.

    Returns (pre-game ratings DataFrame, final ratings dict, last season processed).
    """
    winners = games[games["Win"] == 1].sort_values(["Date", "GameID"])
    
    ratings = {}          # TeamID -> current rating
    results = []          # one dict per team per game
    current_season = None

    for row in winners.itertuples():
        # 1. New season? Pull every rating partway back toward START_RATING.
        if row.Season != current_season:
            ratings = {team: START_RATING + keep * (rating - START_RATING) for team, rating in ratings.items()}
            current_season = row.Season

        # 2. Pre-game ratings (START_RATING the first time a team appears)
        winner_pre = ratings.get(row.TeamID, START_RATING)
        loser_pre = ratings.get(row.OppTeamID, START_RATING)

        # 3. SAVE pre-game ratings BEFORE updating (the leakage rule)
        results.append({"GameID": row.GameID, "TeamID": row.TeamID, "EloPre": winner_pre})
        results.append({"GameID": row.GameID, "TeamID": row.OppTeamID, "EloPre": loser_pre})

        # 4. Home advantage from the winner's point of view:
        #    "H" -> +home_adv, "A" -> -home_adv, "N" -> 0
        adv = home_adv if row.Loc == "H" else -home_adv if row.Loc == "A" else 0

        # 5. Expected score, then update both ratings
        expected = expected_score(winner_pre, loser_pre, adv)
        change = k * (1 - expected)
        ratings[row.TeamID] = winner_pre + change
        ratings[row.OppTeamID] = loser_pre - change

    return pd.DataFrame(results), ratings, current_season


# change = K × (actual − expected)
# new_rating = 1500 + keep × (old_rating − 1500)
def compute_elo(games: pd.DataFrame, k=K, home_adv=HOME_ADV, keep=KEEP) -> pd.DataFrame:
    """Run Elo over games in date order.

    Returns one row per team per game with that team's PRE-game rating:
    columns GameID, TeamID, EloPre.
    """
    pre_game, _, _ = run_elo(games, k, home_adv, keep)

    return pd.DataFrame(pre_game)

def current_ratings(games, season, k=K, home_adv=HOME_ADV, keep=KEEP) -> dict:
    """Each team's rating going into its next game of `season`."""
    _, ratings, last_season = run_elo(games, k, home_adv, keep)
    if last_season != season:
        ratings = {team: START_RATING + keep * (rating - START_RATING) for team, rating in ratings.items()}
    return ratings
