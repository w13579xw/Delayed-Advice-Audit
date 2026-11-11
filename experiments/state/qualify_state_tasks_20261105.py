"""Public-source and non-weakened local qualification for three new tasks."""
import argparse,hashlib,inspect,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'experiments/state')]
from server_run import save,now,fingerprint
from state_core_20261105 import ContactEnv,BCLocal,Teacher,sensor,TASKS

def evaluate(task,policy,seeds):
    rows=[]
    for seed in seeds:
        e=ContactEnv(task,seed);rewards=[]
        try:
            while e.t<500:rewards.append(e.step(policy.action(sensor(e.observe()))))
            rows.append(dict(seed=seed,success=int(e.seen_success),terminal_success=int(e.last_success),native_return=float(np.mean(rewards))))
        finally:e.close()
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out).resolve()
    if out.exists():raise RuntimeError('New output directory required')
    out.mkdir(parents=True);start=time.monotonic();fp=fingerprint();save(out/'run.json',dict(status='running',created_utc=now(),source_fingerprint=fp))
    save(out/'protocol.json',dict(frozen_utc=now(),tasks=TASKS,max_demos=200,whole_episode_ladder=[40,100,200],k=[1,3,5],validation_count=12,qualification_count=20,qualification_minimum=.8,
        inputs='same10goal-observable statecoordinates, noisy onlyin laterrobustness; native39notfullyexposed',smoke_only=a.smoke))
    results={}
    try:
        for ti,task in enumerate(TASKS):
            teacher=Teacher(task);base=810000000+ti*10000;obs=[];actions=[];ids=[]
            source=Path(inspect.getfile(type(teacher.policy)));(out/(task+'_expert.py')).write_bytes(source.read_bytes())
            demos=5 if a.smoke else 200
            for i in range(demos):
                e=ContactEnv(task,base+i)
                try:
                    while e.t<500:
                        native=e.observe();x=sensor(native);act=teacher.action(x)
                        np.testing.assert_array_equal(act,teacher.action(native))
                        obs.append(x);actions.append(act);ids.append(base+i);e.step(act)
                finally:e.close()
                if (i+1)%20==0:print('DEMOS',task,i+1,demos,flush=True)
            obs=np.asarray(obs);actions=np.asarray(actions);ids=np.asarray(ids);np.savez_compressed(out/(task+'_demos.npz'),obs=obs,action=actions,episode=ids)
            if a.smoke:
                local=BCLocal(obs,actions,1);seeds=range(base+400,base+402)
                results[task]=dict(profile_only=True,public=evaluate(task,teacher,seeds),local_debug=evaluate(task,local,seeds))
                # Snapshot/derived-array replay at two contact stages.
                for cut in (80,288):
                    e=ContactEnv(task,base+600+cut)
                    try:
                        while e.t<cut:e.step(local.action(sensor(e.observe())))
                        snap=e.snapshot();first=[]
                        while e.t<500:
                            action=local.action(sensor(e.observe()));reward=e.step(action);first.append((action.copy(),reward,e.observe(),e.seen_success,e.last_success))
                        e.restore(snap)
                        for action,reward,ob,success,last in first:
                            aa=local.action(sensor(e.observe()));np.testing.assert_array_equal(aa,action);assert e.step(aa)==reward;np.testing.assert_array_equal(e.observe(),ob);assert e.seen_success==success and e.last_success==last
                    finally:e.close()
            else:
                grid=[];selected=None
                for n in (40,100,200):
                    mask=ids<base+n
                    for k in (1,3,5):
                        model=BCLocal(obs[mask],actions[mask],k);r=evaluate(task,model,range(base+1000,base+1012))
                        grid.append(dict(demos=n,k=k,success=float(np.mean([x['success'] for x in r])),reward=float(np.mean([x['native_return'] for x in r])),rows=r))
                    acceptable=[x for x in grid if x['demos']==n and x['success']>=10/12]
                    if acceptable:selected=max(acceptable,key=lambda x:(x['success'],x['reward'],-x['k']));break
                save(out/(task+'_validation.json'),dict(grid=grid,selected=selected))
                if selected is None:
                    results[task]=dict(qualified=False,reason='Localvalidationbelow10/12after200demonstrations; noadvice-efficacyclaim');save(out/'analysis.json',results);continue
                folder=out/task;folder.mkdir();mask=ids<base+selected['demos'];np.savez_compressed(folder/'model.npz',obs=obs[mask],action=actions[mask],episode=ids[mask]);save(folder/'config.json',dict(task=task,selected_k=selected['k'],selected_demos=selected['demos'],frozen_utc=now(),observation='noisy10coordinate synthetic39container',model_sha256=hashlib.sha256((folder/'model.npz').read_bytes()).hexdigest()))
                local=BCLocal(obs[mask],actions[mask],selected['k']);seeds=range(base+2000,base+2020)
                pr=evaluate(task,teacher,seeds);lr=evaluate(task,local,seeds);rates={k:float(np.mean([x['success'] for x in v])) for k,v in [('public',pr),('local',lr)]}
                results[task]=dict(qualified=min(rates.values())>=.8,rates=rates,public=pr,local=lr,model_path=str(folder),selected_demos=selected['demos'],selected_k=selected['k'])
            save(out/'analysis.json',results);print('TASK_DONE',task,results[task].get('rates','engineeringprofile'),flush=True)
        assert fingerprint()==fp
        save(out/'checks.json',dict(checks='passed',all_tasks_reported=True,source_uses_same_declared_state=True,snapshot_replays=6 if a.smoke else 'seeprerequisitesmoke',smoke_only=a.smoke))
        save(out/'run.json',dict(status='complete',completed_utc=now(),seconds=time.monotonic()-start,exit_code=0));print('COMPLETE',out,flush=True)
    except BaseException as e:save(out/'run.json',dict(status='failed',ended_utc=now(),error=str(e),exit_code=1));raise
if __name__=='__main__':main()
