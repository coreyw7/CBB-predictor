import pytest

from cbb_predictor.update import STEPS, run_steps


def test_runs_steps_in_order(capsys):
    ran = []
    steps = [(name, lambda name=name: ran.append(name)) for name in ["a", "b", "c"]]
    run_steps(steps)
    assert ran == ["a", "b", "c"]


def test_stops_at_first_error(capsys):
    ran = []

    def broken():
        raise RuntimeError("API down")

    steps = [("a", lambda: ran.append("a")), ("b", broken), ("c", lambda: ran.append("c"))]
    with pytest.raises(RuntimeError, match="API down"):
        run_steps(steps)
    assert ran == ["a"]


def test_prints_each_step_name(capsys):
    run_steps([("fetch", lambda: None), ("teams", lambda: None)])
    out = capsys.readouterr().out
    assert out.index("fetch") < out.index("teams")


def test_pipeline_order():
    assert [name for name, _ in STEPS] == ["fetch", "teams", "live", "score", "predict", "report"]


def test_steps_are_functions_not_results():
    # ("fetch", cbbd.main()) would run the step at import time and store None
    assert all(callable(step) for _, step in STEPS)
