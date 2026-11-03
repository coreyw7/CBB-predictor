# src/cbb_predictor/db.py
import duckdb
import pandas as pd

from cbb_predictor.config import DB_PATH


def get_connection(db_path=DB_PATH, read_only=False) -> duckdb.DuckDBPyConnection:
    """Open a connection to the project's DuckDB database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def write_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Create or replace a table from a DataFrame."""
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")


def append_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Add a DataFrame's rows to a table, creating the table the first time."""
    con.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM df LIMIT 0")
    con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM df")