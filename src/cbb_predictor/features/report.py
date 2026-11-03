import webbrowser

from cbb_predictor.config import REPORTS_DIR
from cbb_predictor.db import get_connection
from cbb_predictor.features.predict import preview_table

CSS = """
        body {
            font-family: system-ui, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            color: #222;
            background: white;
        }

        table {
            border-collapse: collapse;
            width: 100%;
        }

        th, td {
            padding: 6px 10px;
            text-align: left;
        }

        th {
            border-bottom: 3px solid #999;
        }

        tbody tr:nth-child(even) {
            background: #f5f5f5;
        }

        p {
            font-size: 14px;
            color: #666;
        }
"""


def build_report(next_day, table) -> str:
    """Return a full HTML page for one game day."""
    styled = (
        table.style
        .bar(subset="WinProb", vmin=0.5, vmax=1.0, cmap="RdYlGn")
        .format({"WinProb": "{:.1%}"})
        .hide(axis="index")
    )
    return f"""
            <!doctype html>
            <html>
                <head>
                    <meta charset="utf-8"><title>CBB predictions {next_day:%Y-%m-%d}</title>
                    <style>{CSS}</style>
                </head>
                <body>
                    <h1>{next_day:%A, %B %-d}</h1>
                    <p>{len(table)} games · Elo win probabilities</p>
                    {styled.to_html()}
                </body>
            </html>
            """


def main() -> None:
    """Build and write daily report in reports directory, then open the report in a web browser."""
    with get_connection(read_only=True) as con:
        predictions = con.sql("SELECT * FROM predictions").df()
        teams = con.sql("SELECT TeamID, TeamName FROM teams").df()

    next_day, table = preview_table(predictions, teams)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{next_day:%Y-%m-%d}.html"
    path.write_text(build_report(next_day, table), encoding="utf-8")
    webbrowser.open(path.as_uri())

    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()