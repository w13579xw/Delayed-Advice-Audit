"""Episode-group one-sided split conformal calibration.

Guarantee requires exchangeable episode groups and a fixed data-collection policy.
It is NOT a bound on harmful-intervention rate conditional on acceptance and does
NOT automatically transfer to an adaptively deployed gate or shifted delays.
"""
import math
import numpy as np


def group_lower_offset(prediction, target, groups, alpha=.2):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    residual = np.asarray(prediction)-np.asarray(target)
    scores = np.array([np.max(residual[np.asarray(groups)==g]) for g in np.unique(groups)])
    rank = int(math.ceil((len(scores)+1)*(1-alpha)))
    # Infinity is the statistically correct answer when the calibration sample
    # is too small for the requested level, not an invitation to clip the rank.
    offset = float(np.sort(scores)[rank-1]) if rank <= len(scores) else float("inf")
    return offset, scores


def group_coverage(lower, target, groups):
    return float(np.mean([np.all(np.asarray(lower)[groups==g] <= np.asarray(target)[groups==g])
                          for g in np.unique(groups)]))
