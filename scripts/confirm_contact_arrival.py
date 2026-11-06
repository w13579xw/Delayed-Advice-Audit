"""Fresh fixed-model replication of a posthoc arrival/source interaction."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np

from contact_delay_study import collect, local_model
from server_run import ROOT, fingerprint, now, save


def analyze(data):
    ids = sorted(set(data['episode'].tolist()))
    source, raw = {}, {}
    for delay in (0,4,12):
        mask = data['delay']==delay
        gain = data['return_gain'][mask]
        difference = gain[:,0].max(axis=1)-gain[:,1:].max(axis=(1,2))
        source[delay] = np.array([difference[data['episode'][mask]==ep].mean() for ep in ids])
        raw[delay] = np.array([gain[data['episode'][mask]==ep,0,3].mean() for ep in ids])
    quantities = dict(source_gap_d0=source[0], source_gap_d12=source[12],
        source_gap_d12_minus_d0=source[12]-source[0], raw12_d12_minus_d0=raw[12]-raw[0])
    rng = np.random.default_rng(20261103)
    indices = rng.integers(len(ids), size=(10000,len(ids)))
    summaries = {}
    for name, values in quantities.items():
        bootstrap = values[indices].mean(axis=1)
        alpha = .05/3 if name != 'raw12_d12_minus_d0' else .05
        summaries[name] = dict(mean=float(values.mean()), ci=np.quantile(bootstrap,[alpha/2,1-alpha/2]).tolist(),
            confidence=1-alpha, primary=name!='raw12_d12_minus_d0', episode_means=values.tolist())
    gap0, gap12, interaction = (summaries[k] for k in ('source_gap_d0','source_gap_d12','source_gap_d12_minus_d0'))
    return dict(metrics=summaries, episodes=len(ids),
        interaction_supported=interaction['ci'][1]<0,
        strict_crossover_supported=gap0['ci'][0]>0 and gap12['ci'][1]<0 and interaction['ci'][1]<0,
        note='Primary family3 Bonferroni bootstrap intervals, conditional on frozen BC data/model. Raw harm contrast secondary95%. No successful deployment claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();parent,root=Path(a.parent).resolve(),Path(a.out).resolve()
    if root.exists():p.error('Use new output directory')
    assert json.loads((parent/'run.json').read_text())['status']=='complete'
    root.mkdir(parents=True)
    config=dict(task='drawer-open-v3',seeds=list(range(97000,97040)),queries=[24,96,240],delays=[0,4,12],
        sources=['public_requested','bc_requested','bc_arrival'],holds=[0,1,3,12],horizon=500,
        parent=str(parent),hypothesis='Moving-state arrival changes expert-minus-cheap repair-oracle ranking',
        primary_family=['source_gap_d0','source_gap_d12','source_gap_d12_minus_d0'],alpha_family=.05,
        secondary='raw public hold12 gain at delay12 minus delay0',frozen_utc=now(),
        stop_rule='All40 episodes; no retuning, no optional sample extension based on sign or interval')
    save(root/'protocol.json',config)
    save(root/'environment.json',{k:importlib.metadata.version(k) for k in ('metaworld','mujoco','gymnasium','numpy','scipy')})
    for path in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py'),*ROOT.glob('tests_metaworld/*.py')]:
        dest=root/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    code=fingerprint();state=dict(status='running',created_utc=now(),source_fingerprint=code)
    save(root/'run.json',state)
    try:
        local=local_model(parent,'drawer-open-v3')
        data=collect('drawer-open-v3',local,config['seeds'],config['queries'])
        assert len(data['episode'])==360 and sorted(set(data['episode'].tolist()))==config['seeds']
        np.savez_compressed(root/'test.npz',**data)
        save(root/'analysis.json',analyze(data))
        assert code==fingerprint()
        save(root/'checks.json',dict(checks='passed',new_test_episodes=40,rows=360,fixed_model=True,all_episodes_retained=True))
        state.update(status='complete',completed_utc=now(),exit_code=0)
    except BaseException as error:
        state.update(status='failed',error=str(error),ended_utc=now());save(root/'run.json',state);raise
    save(root/'run.json',state);print('COMPLETE',root,flush=True)


if __name__=='__main__':main()
