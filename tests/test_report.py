import pandas as pd
import pytest

from cbb_predictor import db
from cbb_predictor.features import report
from cbb_predictor.features.report import build_report

NEXT_DAY = pd.Timestamp("2026-11-10")


def make_table(rows):
    """rows: list of (tip, favorite, loc word, underdog, WinProb), like preview_table's output."""
    return pd.DataFrame(rows, columns=["Tip Off (ET)", "Favorite", "", "Underdog", "WinProb"])


TABLE = make_table([
    ("7:00 PM", "Duke", "vs", "Army", 0.953),
    ("9:00 PM", "BYU", "at", "Ohio St", 0.612),
])


def test_is_a_complete_html_page():
    html = build_report(NEXT_DAY, TABLE).strip()
    assert html.lower().startswith("<!doctype html>")
    assert html.endswith("</html>")
    assert '<meta charset="utf-8">' in html


def test_title_and_heading_show_the_date():
    html = build_report(NEXT_DAY, TABLE)
    assert "<title>CBB predictions 2026-11-10</title>" in html
    assert "Tuesday, November 10" in html


def test_shows_game_count():
    assert "2 games" in build_report(NEXT_DAY, TABLE)


def test_every_game_is_in_the_table():
    html = build_report(NEXT_DAY, TABLE)
    for name in ["Duke", "Army", "BYU", "Ohio St"]:
        assert name in html


def test_win_probability_shown_as_percent():
    html = build_report(NEXT_DAY, TABLE)
    assert "95.3%" in html
    assert "61.2%" in html
    assert "0.953" not in html


def test_win_probability_has_bars():
    # .bar() draws each bar as a linear-gradient background on the WinProb cell
    assert build_report(NEXT_DAY, TABLE).count("linear-gradient") == 2


def test_page_css_is_included():
    html = build_report(NEXT_DAY, TABLE)
    assert "<style>" in html
    assert "border-collapse" in html


def test_table_is_not_modified():
    table = TABLE.copy()
    build_report(NEXT_DAY, table)
    pd.testing.assert_frame_equal(table, TABLE)


# --- main -----------------------------------------------------------------------

@pytest.fixture
def report_env(tmp_path, monkeypatch):
    """A seeded temp database, a temp reports folder that doesn't exist yet, and a fake browser."""
    db_path = tmp_path / "test.duckdb"
    predictions = pd.DataFrame({
        "GameID": [1, 1], "Date": pd.to_datetime(["2026-11-10"] * 2),
        "StartET": pd.to_datetime(["2026-11-10 19:00"] * 2), "TimeTBD": [False, False],
        "TeamID": [1, 2], "OppTeamID": [2, 1], "Loc": ["H", "A"], "WinProb": [0.9, 0.1],
    })
    teams = pd.DataFrame({"TeamID": [1, 2], "TeamName": ["Duke", "Army"]})
    with db.get_connection(db_path) as con:
        db.write_table(con, "predictions", predictions)
        db.write_table(con, "teams", teams)

    reports_dir = tmp_path / "reports"
    opened = []
    monkeypatch.setattr(report, "get_connection",
                        lambda read_only=False: db.get_connection(db_path, read_only=read_only))
    monkeypatch.setattr(report, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(report.webbrowser, "open", opened.append)
    return reports_dir, opened


def test_main_creates_folder_and_dated_report(report_env, capsys):
    reports_dir, _ = report_env
    report.main()
    path = reports_dir / "2026-11-10.html"
    assert path.exists()
    assert "Duke" in path.read_text(encoding="utf-8")


def test_main_opens_the_saved_report(report_env, capsys):
    reports_dir, opened = report_env
    report.main()
    assert opened == [(reports_dir / "2026-11-10.html").as_uri()]


def test_main_prints_where_report_was_saved(report_env, capsys):
    report.main()
    assert "2026-11-10.html" in capsys.readouterr().out


def test_main_runs_twice_without_error(report_env, capsys):
    # The folder already exists the second time
    report.main()
    report.main()
