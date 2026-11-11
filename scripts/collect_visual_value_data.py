"""Episode-grouped strong-source decision data with an observed-input firewall."""
import argparse,hashlib,json,tempfile,shutil,time
from pathlib import Path
import numpy as np
from advice_audit.metaworld_adapter import ContactEnv
from advice_audit.act_bridge import ACTClient,source_action_to_native
from visual_bundle import BundleClient,DeterministicCamera,actions,make_manifest
from visual_value_opportunity import metrics,summarize
from server_run import ROOT,fingerprint,save,now


def write_inputs(path,robot,prediction):
    visible={key:prediction[key] for key in ('head0','head1','head2','head3')}
    feature=prediction['features'];assert feature.shape==(512,16,16) and feature.dtype==np.float16
    np.savez_compressed(path,visual_features=feature,robot=robot,**visible)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();root=Path(a.out).resolve()
    if root.exists():p.error('New output directory required')
    root.mkdir(parents=True);code=fingerprint();start=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
    for source_file in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
        dest=root/'source_snapshot'/source_file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source_file.read_bytes())
    rng=np.random.default_rng(2026110313);schedules={};tasks=[('push-v3','push',60400200 if a.smoke else 62400000),('pick-place-v3','pick',60500200 if a.smoke else 62500000)]
    combos=[(q,d) for q in (16,32,64,128) for d in (0,16,32)]
    for task,tag,base in tasks:
        pairs=[(32,16)] if a.smoke else [combos[i] for i in rng.permutation(12)]+[combos[i] for i in rng.integers(0,12,8)]
        schedules[task]=[dict(seed=base+i,query=int(q),delay=int(d),split='engineering' if a.smoke else ('train' if i<14 else 'validation')) for i,(q,d) in enumerate(pairs)]
    save(root/'protocol.json',dict(frozen_utc=now(),schedules=schedules,source='frozen15k-budgetACTbestofflinevalidation',offsamples=0,training_features_offsamples=4,primary_audit='nativeany-success;cutoff/rewardsecondary',roles='train/validationpilotonly;noformaltest/calibration',options=12,source_packet='same request packet,hidden from controls until arrival',input_firewall='onlyRGB-derivedfeatures/robot4/cheappredictions;outcomesandprefixsuccessoutsideinputfiles',smoke_only=a.smoke));rows=[];parities=[];steps_total=0
    try:
        for task,tag,base in tasks:
            source=ROOT/'experiments/visual_fit'/f'act_budget15k_{tag}_20261103_v1'/'best';manifest,path=make_manifest(task,root,ROOT,source_override=source);local=manifest['selected'];options=['reject']+[f'cheap_head{i}_48' for i in range(4) if i!=local]+['cheap_ensemble48','cheap_request48','cheap_arrival48','act_raw16','act_raw48','act_suffix16','act_suffix48','act_blend48']
            with tempfile.TemporaryDirectory(prefix='advice_audit_valuedata_',dir='/dev/shm') as temporary:
                work=Path(temporary);client=BundleClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/visual_bundle_worker.py',path,work/'bundle');save(root/(task+'_worker_ready.json'),client.ready)
                try:
                    for item in schedules[task]:
                        seed,q,delay=item['seed'],item['query'],item['delay'];arrival=q+delay;stem=f'{task}_{seed}';env=ContactEnv(task,seed);camera=DeterministicCamera(env);snap=None;packet=None;request_action=None;arrival_action=None;request_file=root/(stem+'_request_inputs.npz');arrival_file=root/(stem+'_arrival_inputs.npz');rr=[];baseline_actions=[]
                        try:
                            while env.t<500:
                                t=env.t
                                if t==arrival:snap=env.snapshot()
                                robot=env.observe()[:4].astype('float32');image=camera.image(env);prediction,_=client.predict(robot,image,source=t==q,features=t in (q,arrival));aa=actions(task,robot,prediction,manifest)
                                if t==q:
                                    packet=prediction['source'].copy();request_action=aa[local].copy();write_inputs(request_file,robot,prediction);np.savez_compressed(root/(stem+'_source_packet.npz'),source=packet)
                                    if a.smoke:
                                        plain,_=client.predict(robot,image,source=True,features=False)
                                        for key in ('head0','head1','head2','head3','source'):np.testing.assert_array_equal(plain[key],prediction[key])
                                        accepted=ACTClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/matched_act_worker.py',source,work/'accepted')
                                        try:raw,_=accepted.predict(robot,image);np.testing.assert_array_equal(source_action_to_native(raw[0]),packet)
                                        finally:accepted.close();shutil.copyfile(work/'accepted/worker_stderr.log',root/(task+'_accepted_stderr.log'))
                                        parities.append(dict(task=task,export_does_not_change_predictions=True,accepted_source_exact=True));save(root/'parities.json',parities)
                                if t==arrival:arrival_action=aa[local].copy();write_inputs(arrival_file,robot,prediction)
                                baseline_actions.append(aa[local]);rr.append(env.step(aa[local]));steps_total+=1
                            base_metrics=metrics(rr,env);np.savez_compressed(root/(stem+'_baseline.npz'),actions=baseline_actions,rewards=rr);assert packet is not None and packet.shape==(100,4) and snap is not None
                            outcomes={'reject':base_metrics}
                            if a.smoke:
                                env.restore(snap);tail_rewards=list(rr[:arrival])
                                while env.t<500:
                                    robot=env.observe()[:4].astype('float32');prediction,_=client.predict(robot,camera.image(env));action=actions(task,robot,prediction,manifest)[local];np.testing.assert_array_equal(action,baseline_actions[env.t]);tail_rewards.append(env.step(action));assert tail_rewards[-1]==rr[env.t-1];steps_total+=1
                                assert metrics(tail_rewards,env)==base_metrics
                            for name in options[1:]:
                                if delay==0 and name.startswith('act_suffix'):outcomes[name]=dict(outcomes[name.replace('suffix','raw')]);continue
                                env.restore(snap);reward=list(rr[:arrival]);trace=[];hold=16 if name.endswith('16') else 48
                                while env.t<500:
                                    elapsed=env.t-arrival;robot=env.observe()[:4].astype('float32');prediction,_=client.predict(robot,camera.image(env));aa=actions(task,robot,prediction,manifest);live=aa[local]
                                    if elapsed>=hold:action=live
                                    elif name.startswith('cheap_head'):action=aa[int(name[len('cheap_head')])]
                                    elif name=='cheap_ensemble48':action=np.mean(np.stack(aa),axis=0)
                                    elif name=='cheap_request48':action=request_action
                                    elif name=='cheap_arrival48':action=arrival_action
                                    elif name.startswith('act_raw'):action=packet[elapsed]
                                    elif name.startswith('act_suffix'):action=packet[delay+elapsed]
                                    elif name=='act_blend48':action=.5*packet[delay+elapsed]+.5*live
                                    else:raise ValueError(name)
                                    trace.append(action);reward.append(env.step(action));steps_total+=1
                                outcomes[name]=metrics(reward,env);np.savez_compressed(root/(stem+'_'+name+'.npz'),actions=trace,rewards=reward)
                            row=dict(**item,task=task,arrival=arrival,outcomes=outcomes,request_inputs=request_file.name,arrival_inputs=arrival_file.name,source_packet=stem+'_source_packet.npz',source_packet_sha256=hashlib.sha256(packet.tobytes()).hexdigest(),offline_only_prefix_any_success=bool(snap['seen_success']));rows.append(row);save(root/'rows.json',rows);save(root/'analysis.json',summarize(rows));print('VALUE_DATA',task,item,'outcomes',outcomes,'seconds',time.monotonic()-start,flush=True)
                        finally:camera.close();env.close()
                finally:client.close();shutil.copyfile(work/'bundle/worker_stderr.log',root/(task+'_worker_stderr.log'))
        assert len(rows)==(2 if a.smoke else 40);assert code==fingerprint()
        for row in rows:
            for name in ('request_inputs','arrival_inputs'):
                with np.load(root/row[name],allow_pickle=False) as z:assert set(z.files)=={'visual_features','robot','head0','head1','head2','head3'};assert z['robot'].shape==(4,) and z['visual_features'].shape==(512,16,16)
        save(root/'checks.json',dict(checks='passed',rows=len(rows),options=12,source_frozen=True,no_privileged_input_fields=True,feature_export_parity_passed=len(parities)==2 if a.smoke else 'see prerequisite smoke',smoke_only=a.smoke));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-start,native_steps=steps_total,exit_code=0)
    except BaseException as e:state.update(status='failed',ended_utc=now(),error=str(e));save(root/'run.json',state);raise
    save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
