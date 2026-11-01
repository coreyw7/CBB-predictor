import pandas as pd
import pytest

from cbb_predictor.ingest.teams import check_mapping, map_team_ids, normalize_name


def make_cbbd_teams(names):
    """CBBD team list with sequential IDs."""
    return pd.DataFrame({"cbbd_id": range(1, len(names) + 1), "cbbd_name": names})


def make_spellings(rows):
    """rows: list of (spelling, TeamID)."""
    return pd.DataFrame(rows, columns=["TeamNameSpelling", "TeamID"])


def make_kaggle_teams(rows):
    """rows: list of (TeamID, LastD1Season)."""
    return pd.DataFrame(rows, columns=["TeamID", "LastD1Season"])


def team_id(mapped, name):
    return mapped.loc[mapped["cbbd_name"] == name, "TeamID"].iloc[0]


SPELLINGS = make_spellings([
    ("duke", 1181),
    ("san jose state", 1361),
    ("centenary (la)", 1147),
    ("miami (oh)", 1275),
])
KAGGLE_TEAMS = make_kaggle_teams([
    (1181, 2026),   # Duke: current D1
    (1361, 2026),   # San Jose State: current D1
    (1147, 2011),   # Centenary: left D1 in 2011
    (1275, 2026),   # Miami (OH): current D1
])


def test_normalize_name_strips_accents_case_and_spaces():
    assert normalize_name("  San José State ") == "san jose state"


def test_normalize_name_keeps_punctuation():
    assert normalize_name("St. John's") == "st. john's"


def test_spelling_match():
    mapped = map_team_ids(make_cbbd_teams(["Duke"]), SPELLINGS, KAGGLE_TEAMS)
    assert team_id(mapped, "Duke") == 1181


def test_accented_name_matches_after_normalizing():
    mapped = map_team_ids(make_cbbd_teams(["San José State"]), SPELLINGS, KAGGLE_TEAMS)
    assert team_id(mapped, "San José State") == 1361


def test_former_d1_team_is_unmapped():
    mapped = map_team_ids(make_cbbd_teams(["Centenary (LA)"]), SPELLINGS, KAGGLE_TEAMS)
    assert pd.isna(team_id(mapped, "Centenary (LA)"))


def test_override_fills_gap():
    mapped = map_team_ids(make_cbbd_teams(["UAlbany"]), SPELLINGS, KAGGLE_TEAMS)
    assert team_id(mapped, "UAlbany") == 1107


def test_override_does_not_replace_spelling_match():
    # "miami" is in TEAM_OVERRIDES (-> 1274), but a real spelling match must win
    spellings = make_spellings([("miami", 1275)])
    mapped = map_team_ids(make_cbbd_teams(["Miami"]), spellings, KAGGLE_TEAMS)
    assert team_id(mapped, "Miami") == 1275


def test_new_team_gets_new_id():
    mapped = map_team_ids(make_cbbd_teams(["West Florida"]), SPELLINGS, KAGGLE_TEAMS)
    assert team_id(mapped, "West Florida") == 1600


def test_unknown_team_is_missing_without_error():
    mapped = map_team_ids(make_cbbd_teams(["Montreat"]), SPELLINGS, KAGGLE_TEAMS)
    assert pd.isna(team_id(mapped, "Montreat"))


def test_output_columns_and_integer_ids():
    mapped = map_team_ids(make_cbbd_teams(["Duke", "Montreat"]), SPELLINGS, KAGGLE_TEAMS)
    assert list(mapped.columns) == ["cbbd_id", "cbbd_name", "TeamID"]
    assert str(mapped["TeamID"].dtype) == "Int64"


def test_input_is_not_modified():
    teams = make_cbbd_teams(["Duke"])
    map_team_ids(teams, SPELLINGS, KAGGLE_TEAMS)
    assert list(teams.columns) == ["cbbd_id", "cbbd_name"]


def make_mapping(rows):
    """rows: list of (cbbd_name, TeamID or None)."""
    df = pd.DataFrame(rows, columns=["cbbd_name", "TeamID"])
    df.insert(0, "cbbd_id", range(1, len(df) + 1))
    df["TeamID"] = df["TeamID"].astype("Int64")
    return df


def test_check_mapping_passes_when_all_d1_teams_mapped():
    kaggle = make_kaggle_teams([(1181, 2026), (1361, 2026)])
    mapping = make_mapping([("Duke", 1181), ("San Jose State", 1361), ("Montreat", None)])
    check_mapping(mapping, kaggle)


def test_check_mapping_raises_on_unmapped_d1_team():
    kaggle = make_kaggle_teams([(1181, 2026), (1361, 2026)])
    mapping = make_mapping([("Duke", 1181)])
    with pytest.raises(ValueError, match="1361"):
        check_mapping(mapping, kaggle)


def test_check_mapping_allows_expected_unmapped():
    # 1384 (St. Francis PA) is in EXPECTED_UNMAPPED
    kaggle = make_kaggle_teams([(1181, 2026), (1384, 2026)])
    mapping = make_mapping([("Duke", 1181)])
    check_mapping(mapping, kaggle)


def test_check_mapping_ignores_former_d1_teams():
    kaggle = make_kaggle_teams([(1181, 2026), (1147, 2011)])
    mapping = make_mapping([("Duke", 1181)])
    check_mapping(mapping, kaggle)


def test_check_mapping_raises_on_duplicate_team_id():
    kaggle = make_kaggle_teams([(1181, 2026)])
    mapping = make_mapping([("Duke", 1181), ("Duke Copy", 1181)])
    with pytest.raises(ValueError, match="more than one"):
        check_mapping(mapping, kaggle)


def test_check_mapping_ignores_many_unmapped_non_d1_teams():
    kaggle = make_kaggle_teams([(1181, 2026)])
    mapping = make_mapping([("Duke", 1181), ("Montreat", None), ("Regent", None)])
    check_mapping(mapping, kaggle)
