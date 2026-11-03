from cbb_predictor.features import predict, report, score
from cbb_predictor.ingest import cbbd, live, teams

STEPS = [
    ("fetch", cbbd.main),
    ("teams", teams.main),
    ("live", live.main),
    ("score", score.main),
    ("predict", predict.main),
    ("report", report.main),
]


def run_steps(steps) -> None:
    """Run each (name, function) in order, stopping at the first error."""
    for name, step in steps:
        print(f"\n=== {name} ===")
        step()


def main() -> None:
    """Fetch, rebuild, score, predict and report in one command."""
    run_steps(STEPS)


if __name__ == "__main__":
    main()