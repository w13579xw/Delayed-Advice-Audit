"""Select the strongest development BC <=200 demos, then freeze new test."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
import numpy as np

from advice_audit.metaworld_adapter import BCLocal, ContactEnv, expert_action, expert_for
from advice_audit.report import paired_ci
from server_run import ROOT, fingerprint, now, save


class Public:
    def __init__(self, task):
        self.policy=expert_for(task)
    def action(self, obs):
        return expert_action(self.policy, obs)


def evaluate(task, policy, seeds):
    rows=[]
    for seed in seeds:
        env=ContactEnv(task, seed)
        try:
            rewards=[];first=None
            while env.t<env.horizon:
                rewards.append(env.step(policy.action(env.observe())))
                if env.seen_success and first is None:first=env.t
            rows.append(dict(seed=seed,success=int(env.seen_success),terminal_success=int(env.last_success),
                native_return=float(np.sum(rewards))/500,first_success_step=first))
        finally:env.close()
    return rows


def summary(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in ('success','terminal_success','native_return')}


def fit_task(root,parent,task,offset,smoke):
    with np.load(parent/(task+'_demos.npz'),allow_pickle=False) as data:
        obs,actions,episodes=(data[k] for k in ('obs','action','episode'))
    ids=sorted(set(episodes.tolist()))
    val_start=5000000+offset if smoke else 4000000+offset
    test_start=5100000+offset if smoke else 4100000+offset
    nval,ntest=(2,2) if smoke else (24,50)
    scores=[]
    for count in ((40,) if smoke else (40,100,200)):
        mask=np.isin(episodes,ids[:count])
        for k in ((1,) if smoke else (1,3,5)):
            model=BCLocal(obs[mask],actions[mask],k)
            rows=evaluate(task,model,range(val_start,val_start+nval))
            entry=dict(demos=count,k=k,metrics=summary(rows),rows=rows)
            scores.append(entry)
            save(root/(task+'_development.json'),scores)
            print(task,'DEVELOPMENT',count,k,entry['metrics'],flush=True)
    def key(row):
        m=row['metrics'];return (m['success'],m['terminal_success'],m['native_return'],-row['k'],row['demos'])
    choice=max(scores,key=key)
    mask=np.isin(episodes,ids[:choice['demos']])
    path=root/(task+'_local.npz')
    np.savez_compressed(path,obs=obs[mask],action=actions[mask],episode=episodes[mask],k=choice['k'])
    save(root/(task+'_freeze.json'),dict(selected=choice,frozen_utc=now(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    test_started=now()
    chosen=BCLocal(obs[mask],actions[mask],choice['k'])
    baselines={'selected_local':chosen,'original40':BCLocal(obs[np.isin(episodes,ids[:40])],actions[np.isin(episodes,ids[:40])],1),'public_expert':Public(task)}
    best200=[r for r in scores if r['demos']==200]
    if best200:
        c200=max(best200,key=key);baselines['best200']=BCLocal(obs,actions,c200['k'])
    results={}
    for name,model in baselines.items():
        results[name]=evaluate(task,model,range(test_start,test_start+ntest))
        print(task,'TEST',name,summary(results[name]),flush=True)
        save(root/(task+'_test.json'),results)
    assert json.loads((root/(task+'_freeze.json')).read_text())['frozen_utc']<=test_started
    assert set(range(val_start,val_start+nval)).isdisjoint(range(test_start,test_start+ntest))
    base=results['original40'];paired={}
    for name,rows in results.items():
        paired[name]={}
        for metric in ('success','terminal_success','native_return'):
            diff=[r[metric]-b[metric] for r,b in zip(rows,base)]
            paired[name][metric]=dict(mean=float(np.mean(diff)),ci95=paired_ci(diff))
    return dict(selected=dict(demos=choice['demos'],k=choice['k']),test_episodes=ntest,
        metrics={name:summary(rows) for name,rows in results.items()},paired_vs_original40=paired,
        limitation='Baseline qualification, not independent final test of a new intervention rule')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root,parent=Path(a.out).resolve(),Path(a.parent).resolve()
    if root.exists():p.error('Use a new run directory')
    assert json.loads((parent/'run.json').read_text())['status']=='complete'
    root.mkdir(parents=True)
    config=dict(parent=str(parent),tasks=['push-v3','drawer-open-v3'],development_seed=4000000,test_seed=4100000,
        task_offsets=[0,100000],development_episodes=24,test_episodes=50,training_ladder=[40,100,200],k=[1,3,5],
        selection='max development any-success, then cutoff-success, then native-return; no test selection',smoke=a.smoke)
    save(root/'protocol.json',dict(config=config,frozen_utc=now()))
    save(root/'environment.json',dict(python=sys.version,versions={x:importlib.metadata.version(x) for x in ('numpy','metaworld','mujoco','scipy','gymnasium')}))
    for file in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
        dest=root/'source_snapshot'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(file.read_bytes())
    code=fingerprint();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
    try:
        results={task:fit_task(root,parent,task,offset,a.smoke) for task,offset in zip(config['tasks'],config['task_offsets'])}
        save(root/'analysis.json',results)
        assert code==fingerprint()
        save(root/'checks.json',dict(checks='passed',frozen_before_test=True,all_results_retained=True,native_horizon=500))
        state.update(status='complete',completed_utc=now(),exit_code=0)
    except BaseException as error:
        state.update(status='failed',error=str(error),ended_utc=now());save(root/'run.json',state);raise
    save(root/'run.json',state);print('COMPLETE',root,flush=True)


if __name__=='__main__':main()
