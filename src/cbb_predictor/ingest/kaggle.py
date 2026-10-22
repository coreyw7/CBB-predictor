import pandas as pd

from cbb_predictor.config import RAW_DIR


def load_games(raw_dir=RAW_DIR) -> pd.DataFrame:
    """Load regular season and NCAA tournament detailed results into one table."""
    reg_games = pd.read_csv(raw_dir / "MRegularSeasonDetailedResults.csv")
    reg_games['GameType'] = 'R'
    tourney_games = pd.read_csv(raw_dir / "MNCAATourneyDetailedResults.csv")
    tourney_games['GameType'] = 'T'
    games = pd.concat([reg_games, tourney_games], ignore_index=True)
    games["GameID"] = range(len(games))
    return games

def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape one-row-per-game (W/L format) into one-row-per-team-per-game."""
    w_rename = {}
    for col in games.columns:
        if col == 'WLoc':
            continue
        if col.startswith('W'):
            w_rename[col] = col[1:]
        elif col.startswith('L'):
            w_rename[col] = "Opp" + col[1:]
    winners = games.rename(columns=w_rename).drop(columns=['WLoc'])
    winners['Loc'] = games['WLoc']
    winners['Win'] = 1

    l_rename = {}
    for col in games.columns:
        if col == 'WLoc':
            continue
        if col.startswith('L'):
            l_rename[col] = col[1:]
        elif col.startswith('W'):
            l_rename[col] = "Opp" + col[1:]

    losers = games.rename(columns=l_rename).drop(columns=['WLoc'])
    losers['Loc'] = games['WLoc'].map({'H': 'A', 'A': 'H', 'N': 'N'})
    losers['Win'] = 0

    team_games = pd.concat([winners, losers], ignore_index=True)
    
    return team_games


def add_dates(team_games: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Add a real calendar Date column using each season's DayZero."""
    team_games = team_games.copy()
    seasons = seasons.copy()
    seasons['DayZero'] = pd.to_datetime(seasons['DayZero'], format="%m/%d/%Y")
    team_games = team_games.merge(seasons[['Season', 'DayZero']], on='Season', how='left')
    team_games['Date'] = pd.to_timedelta(team_games['DayNum'], unit='D') + team_games['DayZero']
    return team_games


def add_possessions(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add estimated game possessions (average of both teams' estimates)."""
    team_games = team_games.copy()
    team_games['Possessions'] = ((team_games['FGA'] - team_games['OR'] + team_games['TO'] + 0.475 * team_games['FTA']) + (team_games['OppFGA'] - team_games['OppOR'] + team_games['OppTO'] + 0.475 * team_games['OppFTA']))/2
    return team_games

def build_team_games(raw_dir=RAW_DIR) -> pd.DataFrame:
    """Run the full pipeline: load, reshape, add dates and possessions."""
    games = load_games(raw_dir)
    team_games = to_team_games(games)
    seasons = pd.read_csv(raw_dir / "MSeasons.csv")
    team_games = add_dates(team_games, seasons)
    team_games = add_possessions(team_games)
    return team_games