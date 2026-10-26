from pathlib import Path

import numpy as np
import pytest

from nsh_pipeline.orchestration import runner


def test_missing_module_is_a_placeholder(tmp_path):
    result = runner.run_module("nsh_pipeline.not_merged_yet.run", [], cwd=tmp_path)
    assert result["status"] == "placeholder"


def test_module_available():
    assert runner.module_available("json.tool")
    assert not runner.module_available("no_such_package.child")


def test_successful_module_returns_ok(tmp_path, monkeypatch, caplog):
    (tmp_path / "okmod.py").write_text("print('hello from stage')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with caplog.at_level("INFO"):
        result = runner.run_module("okmod", [], cwd=tmp_path)
    assert result["status"] == "ok"
    assert "hello from stage" in caplog.text  # stage output reaches the task log


def test_failing_module_raises_with_output(tmp_path, monkeypatch):
    (tmp_path / "badmod.py").write_text("import sys\nprint('schema check failed')\nsys.exit(3)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(runner.StageFailed, match="code 3") as err:
        runner.run_module("badmod", [], cwd=tmp_path)
    assert "schema check failed" in str(err.value)


def test_args_are_passed(tmp_path, monkeypatch, caplog):
    (tmp_path / "argmod.py").write_text("import sys\nprint('ARGS', sys.argv[1:])\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with caplog.at_level("INFO"):
        runner.run_module("argmod", ["--params", "params.yaml"], cwd=tmp_path)
    assert "'--params', 'params.yaml'" in caplog.text


def test_timeout_raises(tmp_path, monkeypatch):
    (tmp_path / "slowmod.py").write_text("import time\ntime.sleep(5)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(runner.StageFailed, match="timed out"):
        runner.run_module("slowmod", [], cwd=tmp_path, timeout=0.5)


def test_jsonable_converts_numpy_and_paths():
    value = {"swipes": np.int64(4800), "nested": [np.float64(1.5), Path("data/raw")]}
    assert runner.jsonable(value) == {"swipes": 4800, "nested": [1.5, str(Path("data/raw"))]}
