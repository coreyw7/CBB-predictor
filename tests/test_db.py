# tests/test_db.py
import pandas as pd

from cbb_predictor.db import get_connection, write_table


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