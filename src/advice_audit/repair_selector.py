"""Observable repair/source selection prototype, with independent calibration.

This is an action-value ensemble over a declared repair portfolio, not a novel
network architecture. Offline local-only coverage does not transfer to an
adaptive closed-loop policy. Oracle values are never inputs to choose().
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .calibration import group_lower_offset
from .features import arrival_features
from .model import GainModel
from .policies import expert_plan
from .repair import MODES, local_nominal_plan

OPTIONS = [(source, mode) for source in range(3) for mode in range(1, len(MODES))]


def option_mask(delays, chunk):
    delays = np.asarray(delays)
    mask = np.ones((len(delays), len(OPTIONS)), dtype=bool)
    for j, (_, mode) in enumerate(OPTIONS):
        if MODES[mode] in ("suffix", "track"):
            mask[:, j] = delays < chunk
        if MODES[mode] == "suffix":
            mask[:, j] &= delays > 0  # at d=0 suffix exactly duplicates raw
    return mask


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def features_for(data, backend, chunk=12, horizon=80):
    features = [[] for _ in range(data["plan_id"].shape[1])]
    for i, requested in enumerate(data["requested"]):
        plans = [expert_plan(requested, horizon=chunk, backend=backend),
                 local_nominal_plan(requested, horizon=chunk, backend=backend)]
        if len(features) == 3:
            plans.append(local_nominal_plan(requested, horizon=chunk, backend=backend, gain=12.))
        for source, plan in enumerate(plans):
            digest = hashlib.sha256(np.ascontiguousarray(plan, dtype="<f4").tobytes()).hexdigest()
            if digest != data["plan_id"][i, source]:
                raise ValueError("Source reconstruction changed; use the frozen source revision")
            features[source].append(arrival_features(data["current"][i], requested, plan,
                int(data["delay"][i]), step=int(data["arrival_step"][i]), horizon=horizon))
    return [np.asarray(x) for x in features]


class RepairSelector:
    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path/"selector.json").read_text())
        self.models = [GainModel.load(self.path/f"option_{j}.npz") for j in range(len(OPTIONS))]

    def scores(self, inputs, delays, calibrated=False, cheap_only=False):
        scores = []
        q = self.config["cheap_offset"] if cheap_only else self.config["offset"]
        for j, ((source, _), model) in enumerate(zip(OPTIONS, self.models)):
            mean, std = model.predict(inputs[source])
            score = mean-self.config["beta"]*std
            if calibrated:
                score -= (q*self.config["scales"][j]) if q is not None else float("inf")
            scores.append(score)
        scores = np.asarray(scores).T
        allowed = option_mask(delays, self.config["chunk"])
        if cheap_only:
            allowed[:, :4] = False
        return np.where(allowed, scores, -np.inf)

    def choose(self, current, requested, plans, delay, step, horizon, calibrated=False, cheap_only=False):
        inputs = [arrival_features(current, requested, plan, delay, step=step, horizon=horizon)[None, :] for plan in plans]
        scores = self.scores(inputs, [delay], calibrated, cheap_only)[0]
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            return None, "reject"
        source, mode = OPTIONS[best]
        return source, MODES[mode]


def fit_selector(root, destination, seed, config):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data = {role: load_data(Path(root)/(role+".npz")) for role in ("train", "validation", "calibration")}
    ids = {role: set(value["episode"].tolist()) for role, value in data.items()}
    if any(ids[a] & ids[b] for a, b in [("train", "validation"), ("train", "calibration"), ("validation", "calibration")]):
        raise ValueError("Episode partition overlap")
    inputs = {role: features_for(value, config["backend"], config["chunk"], config["horizon"]) for role, value in data.items()}
    scales, cal_pred, cal_target = [], [], []
    validation_mask = option_mask(data["validation"]["delay"], config["chunk"])
    calibration_mask = option_mask(data["calibration"]["delay"], config["chunk"])
    for j, (source, mode) in enumerate(OPTIONS):
        y = data["train"]["gain"][:, source, mode]
        model = GainModel(seed=seed*100+j, width=64, members=5).fit(inputs["train"][source], y)
        model.save(destination/f"option_{j}.npz")
        mean, std = model.predict(inputs["validation"][source])
        residual = mean-config["beta"]*std-data["validation"]["gain"][:, source, mode]
        scale = max(config["scale_floor"], float(np.sqrt(np.mean(residual[validation_mask[:, j]]**2))))
        scales.append(scale)
        mean, std = model.predict(inputs["calibration"][source])
        normalized = (mean-config["beta"]*std)/scale
        normalized = np.where(calibration_mask[:, j], normalized, -np.inf)
        cal_pred.append(normalized)
        cal_target.append(data["calibration"]["gain"][:, source, mode]/scale)
    pred = np.asarray(cal_pred).T
    target = np.asarray(cal_target).T
    offset, _ = group_lower_offset(pred.ravel(), target.ravel(),
        np.repeat(data["calibration"]["episode"], len(OPTIONS)), config["alpha"])
    cheap_offset, _ = group_lower_offset(pred[:, 4:].ravel(), target[:, 4:].ravel(),
        np.repeat(data["calibration"]["episode"], len(OPTIONS)-4), config["alpha"])
    report = dict(seed=seed, beta=config["beta"], chunk=config["chunk"], scales=scales,
        offset=float(offset) if np.isfinite(offset) else None,
        cheap_offset=float(cheap_offset) if np.isfinite(cheap_offset) else None, options=OPTIONS,
        partitions={k: sorted(v) for k, v in ids.items()},
        source_hashes={role: hashlib.sha256((Path(root)/(role+".npz")).read_bytes()).hexdigest() for role in data},
        note="Validation scales fixed before joint group-max calibration; point and calibrated option selection; local-only labels")
    (destination/"selector.json").write_text(json.dumps(report, indent=2))
    return report
