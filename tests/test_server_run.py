"""Check server orchestration recovery and overwrite protection, without training."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("server_run", Path(__file__).resolve().parents[1] / "scripts/server_run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def configure(monkeypatch, tmp_path, resume=False):
    args = ["server_run.py", "--run-dir", str(tmp_path / "run")]
    if resume:
        args.append("--resume")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "fingerprint", lambda: "source-v1")


def test_existing_output_is_never_silently_overwritten(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    original = output / "keep.txt"
    original.write_text("previous research")
    with pytest.raises(SystemExit):
        runner.main()
    assert original.read_text() == "previous research"


def test_failed_phase_retains_log_and_resumes_with_new_attempt(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "phases", lambda config, directory: [
        ("probe", [sys.executable, "-c", "print('retained failure'); raise SystemExit(2)"], [])])
    with pytest.raises(RuntimeError, match="exit 2"):
        runner.main()
    output = tmp_path / "run"
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert "retained failure" in (output / "logs/probe_attempt_01.log").read_text()
    configure(monkeypatch, tmp_path, resume=True)
    monkeypatch.setattr(runner, "phases", lambda config, directory: [
        ("probe", [sys.executable, "-c", "print('recovered')"], [])])
    runner.main()
    state = json.loads((output / "run.json").read_text())
    assert state["status"] == "complete"
    assert state["phases"]["probe"]["attempt"] == 2
    assert "retained failure" in (output / "logs/probe_attempt_01.log").read_text()
    runner.main()
    assert not (output / "logs/probe_attempt_03.log").exists()


def test_resume_rejects_source_changes_and_missing_completed_outputs(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "phases", lambda config, directory: [])
    runner.main()
    configure(monkeypatch, tmp_path, resume=True)
    monkeypatch.setattr(runner, "fingerprint", lambda: "different-source")
    with pytest.raises(SystemExit):
        runner.main()
    monkeypatch.setattr(runner, "fingerprint", lambda: "source-v1")
    file = tmp_path / "run/run.json"
    state = json.loads(file.read_text())
    state["phases"]["probe"] = {"status": "complete", "attempt": 1}
    file.write_text(json.dumps(state))
    monkeypatch.setattr(runner, "phases", lambda config, directory: [("probe", [], ["missing.npz"])])
    with pytest.raises(RuntimeError, match="missing outputs"):
        runner.main()
