"""Offline fitting/decision reproduction, NumPy only; no native inference claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    spec=importlib.util.spec_from_file_location('frozen_value_kernel',HERE/'value_kernel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fitted = read(ASSETS/'models/analysis.json')
    episodes = read(HERE/'source_data/arrival_episodes.json')
    effects = read(HERE/'source_data/arrival_analysis.json')
    report = {}
    for task in ('push-v3','pick-place-v3'):
        info = fitted[task]['native_return']
        cache=np.load(ASSETS/'cached_distances'/(task+'.npz'))
        train_dist,val_dist=cache['train_distances'],cache['validation_distances']
        all_names = info['models']['cheap_inputs']['decision_cells']['all_actions']['allowed_names']
        yt,yv=cache['train_labels'],cache['validation_labels']
        max_mse_error, max_dual_error = 0.,0.
        selections = {}
        for features,family in info['models'].items():
            block_count=5 if features=='cheap_inputs' else 6
            dt,dv=sum(train_dist[:block_count]),sum(val_dist[:block_count])
            for actions,cell in family['decision_cells'].items():
                stem=task+'_native_return_'+features+'_'+actions
                allowed=[all_names.index(n) for n in cell['allowed_names']]
                zero_mse=float(np.mean(yv[:,allowed]**2))
                candidates=[dict(model='zero',bandwidth=None,ridge=None,validation_mse=zero_mse,dual=np.zeros_like(yt))]
                grid=read(ASSETS/'models'/(stem+'_grid.json'))
                for bandwidth in (.25,1.,4.):
                    kt,kv=np.exp(-dt/(2*bandwidth**2)),np.exp(-dv/(2*bandwidth**2))
                    for ridge in (.01,.1,1.,10.):
                        dual=np.linalg.solve(kt+ridge*np.eye(len(yt)),yt)
                        pred=kv@dual;pred[:,0]=0.
                        mse=float(np.mean((pred[:,allowed]-yv[:,allowed])**2))
                        saved=next(g for g in grid if g['bandwidth']==bandwidth and g['ridge']==ridge)
                        max_mse_error=max(max_mse_error,abs(mse-saved['validation_mse']))
                        candidates.append(dict(model='RBFkernelridge',bandwidth=bandwidth,ridge=ridge,validation_mse=mse,dual=dual))
                best=module.select_global_minimum(candidates)
                expected=cell['selected']
                assert all(best[k]==expected[k] for k in ('model','bandwidth','ridge'))
                saved_dual=np.load(ASSETS/'models'/(stem+'_model.npz'))['dual']
                max_dual_error=max(max_dual_error,float(np.max(np.abs(best['dual']-saved_dual))))
                selections[features+'/'+actions]={k:best[k] for k in ('model','bandwidth','ridge','validation_mse')}
        assert max_mse_error<1e-12 and max_dual_error<1e-10
        policy=object.__new__(module.FrozenKernelDecisions)
        policy.analysis=info;policy.cells={}
        for features,family in info['models'].items():
            for actions,cell in family['decision_cells'].items():
                stem=task+'_native_return_'+features+'_'+actions
                policy.cells[(features,actions)]=(cell,np.load(ASSETS/'models'/(stem+'_model.npz'))['dual'])
        tests=[r for r in episodes if r['task']==task]
        assert len(tests)==250
        max_prediction_error=0.
        for index,row in enumerate(tests):
            stem=task+'_'+str(row['seed'])
            choice=read(ASSETS/'choices'/(stem+'.json'))
            decisions=policy.decide(cache['test_distances'][index])
            for cell,value in decisions.items():
                assert value['choice']==choice['decisions'][cell]==row['decisions'][cell]
                max_prediction_error=max(max_prediction_error,abs(value['predicted_gain']-choice['cell_predictions'][cell]['predicted_gain']))
        assert max_prediction_error<1e-10
        for contrast in effects[task]['primary_contrasts'].values():
            d=np.array([r['conditions'][contrast['left']]['native_return']-r['conditions'][contrast['right']]['native_return'] for r in tests])
            assert np.array_equal(d,np.array(contrast['raw_paired_differences']))
            assert float(d.mean())==contrast['mean']
        report[task]=dict(grid_mse_max_error=max_mse_error,selected_dual_max_error=max_dual_error,
                          runtime_prediction_max_error=max_prediction_error,all250choices_equal=True,
                          all_primary_vectors_equal=True,selected=selections)
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    (args.out/'checks.json').write_text(json.dumps(dict(checks='passed',all500choices_equal=True,
        all8cell_selections_equal=True,all6primary_vectors_equal=True,native_reexecution=False),indent=2))
    (args.out/'run.json').write_text(json.dumps(dict(status='complete',exit_code=0,
        seconds=time.monotonic()-started,scope='Offline cached kernel distances, fitting, decisions and episode-level arithmetic only; no feature-extraction or distance-construction verification.'),indent=2))
    print('COMPLETE',args.out,flush=True)


if __name__=='__main__':
    main()
