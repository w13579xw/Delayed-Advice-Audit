import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
from contact_delay_study import best_fixed, offline_analysis


def fixture():
    return dict(success_gain=np.zeros((2, 3, 4)), return_gain=np.zeros((2, 3, 4)),
                terminal_gain=np.zeros((2, 3, 4)), episode=np.array([1, 2]))


def test_fixed_selection_prioritizes_declared_native_success():
    data = fixture()
    data['success_gain'][:, 0, 1] = 1
    data['return_gain'][:, 0, 1] = -10
    data['return_gain'][:, 1, 3] = 10
    assert best_fixed(data, (0, 1, 2))['source'] == 0
    assert best_fixed(data, (1, 2))['source'] == 1


def test_negative_candidates_do_not_force_acceptance():
    data = fixture()
    data['success_gain'][:, :, 1:] = -1
    data['return_gain'][:, :, 1:] = -3
    assert best_fixed(data, (0, 1, 2))['hold'] == 0


def test_expert_portfolio_increment_is_positive_part_not_signed_gap():
    data = fixture()
    data['return_gain'][0, 0, 1] = 1
    data['return_gain'][1, 1, 1] = 3
    choices = {name: dict(source=0, mode=0) for name in ('full_fixed', 'cheap_fixed', 'public_fixed')}
    metrics = offline_analysis(data, choices)['return_gain']
    assert metrics['expert_portfolio_increment']['mean'] == .5
    assert metrics['full_realized_oracle']['mean'] == 2
