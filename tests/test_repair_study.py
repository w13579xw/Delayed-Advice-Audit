import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from advice_audit.protocol import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    spec = importlib.util.spec_from_file_location("repair_value_pilot", ROOT/"scripts/repair_value_pilot.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def protocol(tmp_path, **changes):
    config = json.loads((ROOT/"configs/repair_value_pilot_v1.json").read_text())
    config.update(backend="toy", **changes)
    (tmp_path/"protocol.json").write_text(json.dumps(dict(config=config, sha256=canonical_hash(config))))
    return config


def test_oracle_source_advantage_alone_is_not_a_reason_to_train_selector(monkeypatch, tmp_path):
    m = module(monkeypatch)
    c = protocol(tmp_path)
    validation = np.zeros((132, 2, 5))
    validation[:, 0, 4] = .05
    validation[:, 1, 2] = .03
    np.savez(tmp_path/"validation.npz", gain=validation)
    m.select(tmp_path)
    selection = json.loads((tmp_path/"selection.json").read_text())
    assert selection["choices"]["expert"]["mode"] == "track"
    assert selection["choices"]["local_nominal"]["mode"] == "prefix3"
    gains = np.zeros_like(validation)
    gains[:, 0, 4] = .05
    gains[:, 1, 4] = .04
    np.savez(tmp_path/"test.npz", gain=gains, success_gain=np.zeros_like(gains),
             episode=np.repeat(np.arange(c["test_seed"], c["test_seed"]+12), 11),
             request_step=np.tile([0, 0, 0, 24, 24, 24, 48, 48, 48, 72, 72], 12),
             plan_id=np.full((132, 2), "identity"))
    m.analyze(tmp_path)
    analysis = json.loads((tmp_path/"analysis.json").read_text())
    assert analysis["metrics"]["source_specific_oracle_difference"]["mean"] == pytest.approx(.01)
    assert analysis["metrics"]["contextual_opportunity"]["mean"] == 0
    assert analysis["decision"].startswith("no_go")
    with pytest.raises(ValueError, match="precede"):
        m.select(tmp_path)


def test_phase_collection_keeps_sources_and_repairs_paired(monkeypatch, tmp_path):
    m = module(monkeypatch)
    protocol(tmp_path, validation_episodes=1)
    m.collect_phase(tmp_path, "validation")
    with np.load(tmp_path/"validation.npz", allow_pickle=False) as data:
        assert data["gain"].shape == (11, 2, 5)
        assert not data["gain"][:, :, 0].any()
        zero = data["delay"] == 0
        expired = data["delay"] == 12
        np.testing.assert_array_equal(data["gain"][zero, :, 1], data["gain"][zero, :, 3])
        assert not data["gain"][expired, :, 3:].any()
        for time in [0, 24, 48, 72]:
            mask = data["request_step"] == time
            for source in [0, 1]:
                assert len(set(data["plan_id"][mask, source])) == 1
