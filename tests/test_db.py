# tests/test_db.py
import pandas as pd

from cbb_predictor.db import append_table, get_connection, write_table


def test_write_and_read_back(tmp_path):
    df = pd.DataFrame({"TeamID": [1, 2], "Score": [80, 70]})

    with get_connection(tmp_path / "test.duckdb") as con:
        write_table(con, "scores", df)
        df = con.sql("SELECT * FROM scores").df()
        assert len(df) == 2
        assert df["Score"].tolist() == [80, 70]

def test_double_write(tmp_path):
    df1 = pd.DataFrame({"TeamID": [1, 2], "Score": [80, 70]})
    df2 = pd.DataFrame({"TeamID": [3, 4], "Score": [60, 50]})

    with get_connection(tmp_path / "test.duckdb") as con:
        write_table(con, "scores", df1)
        write_table(con, "scores", df2)
        df = con.sql("SELECT * FROM scores").df()
        assert len(df) == 2
        assert df["TeamID"].tolist() == [3, 4]
        assert df["Score"].tolist() == [60, 50]


def test_append_creates_table_first_time(tmp_path):
    df = pd.DataFrame({"TeamID": [1, 2], "Score": [80, 70]})

    with get_connection(tmp_path / "test.duckdb") as con:
        append_table(con, "log", df)
        result = con.sql("SELECT * FROM log").df()
        assert result["TeamID"].tolist() == [1, 2]


def test_append_keeps_earlier_rows(tmp_path):
    df1 = pd.DataFrame({"TeamID": [1, 2], "Score": [80, 70]})
    df2 = pd.DataFrame({"TeamID": [3, 4], "Score": [60, 50]})

    with get_connection(tmp_path / "test.duckdb") as con:
        append_table(con, "log", df1)
        append_table(con, "log", df2)
        result = con.sql("SELECT * FROM log ORDER BY TeamID").df()
        assert result["TeamID"].tolist() == [1, 2, 3, 4]


def test_append_matches_columns_by_name(tmp_path):
    first = pd.DataFrame({"TeamID": [1], "Score": [80]})
    reordered = pd.DataFrame({"Score": [60], "TeamID": [3]})

    with get_connection(tmp_path / "test.duckdb") as con:
        append_table(con, "log", first)
        append_table(con, "log", reordered)
        result = con.sql("SELECT * FROM log ORDER BY TeamID").df()
        assert result["TeamID"].tolist() == [1, 3]
        assert result["Score"].tolist() == [80, 60]


def test_append_survives_reconnecting(tmp_path):
    db_path = tmp_path / "test.duckdb"

    with get_connection(db_path) as con:
        append_table(con, "log", pd.DataFrame({"TeamID": [1]}))
    with get_connection(db_path) as con:
        append_table(con, "log", pd.DataFrame({"TeamID": [2]}))
        assert con.sql("SELECT COUNT(*) FROM log").fetchone()[0] == 2
