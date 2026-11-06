import itertools
from advice_audit.value_kernel import select_global_minimum


def test_tolerance_ties_do_not_drift_and_are_order_independent():
    candidates=[dict(model='zero',validation_mse=.2,ridge=None,bandwidth=None)]
    candidates += [dict(model='RBFkernelridge',validation_mse=.2-d,ridge=r,bandwidth=.25) for d,r in [(1.44e-12,.01),(1.32e-12,.1),(.73e-12,1.),(.13e-12,10.)]]
    for order in itertools.permutations(candidates):
        chosen=select_global_minimum(order)
        assert chosen['ridge']==1.
        assert chosen['validation_mse']<=chosen['global_minimum_mse']+1e-12


def test_zero_model_wins_when_in_global_tolerance_pool():
    candidates=[dict(model='zero',validation_mse=.2,ridge=None,bandwidth=None),dict(model='RBFkernelridge',validation_mse=.2-.5e-12,ridge=10.,bandwidth=1.)]
    assert select_global_minimum(candidates)['model']=='zero'
