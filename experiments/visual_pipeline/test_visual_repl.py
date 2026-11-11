"""Prospective frozen single-arrival decisions, then exact native branch execution."""
import argparse,hashlib,json,tempfile,shutil,time
from pathlib import Path
import sys
REPO=Path(str(Path(__file__).resolve().parents[2]));sys.path[:0]=[str(REPO/"src"),str(REPO/"scripts")]
import numpy as np
from advice_audit.metaworld_adapter import ContactEnv
from advice_audit.value_kernel import FrozenKernelDecisions
from visual_bundle import BundleClient,DeterministicCamera,actions,make_manifest
from visual_value_opportunity import metrics
from server_run import ROOT,fingerprint,save,now
from replication_support import make_manifest,source_path


def visible(robot,prediction):
    return dict(robot=robot.copy(),visual_features=prediction['features'].copy(),**{f'head{i}':prediction[f'head{i}'].copy() for i in range(4)})


def file_hashes(paths):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--schedule',required=True);p.add_argument('--models',required=True);a=p.parse_args();root=Path(a.out).resolve()
    if root.exists():p.error('New output directory required')
    models=Path(a.models).resolve();assert json.loads((models/'checks.json').read_text())['checks']=='passed';assert json.loads((ROOT/'outputs/vision/value_runtime_parity_20261104_v1/checks.json').read_text())['checks']=='passed'
    root.mkdir(parents=True);code=fingerprint();start=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
    for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
        dest=root/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    schedule_payload=json.loads(Path(a.schedule).read_text());count=2 if a.smoke else 50;schedules=schedule_payload['test_smoke' if a.smoke else 'test'];tasks=[(task,tag,schedules[task][0]['seed']) for task,tag in [('push-v3','push'),('pick-place-v3','pick')]]
    tracked=sorted(p for p in models.iterdir() if p.suffix in ('.npz','.json'));manifest_paths=[];hashes=file_hashes(tracked)
    save(root/'protocol.json',dict(frozen_utc=now(),schedules=schedules,episodes_per_task=count,smoke_only=a.smoke,models=str(models),model_sha256=hashes,primary='wholeepisode mean native reward',secondary=['nativeany-success','cutoffsuccess'],primary_contrasts='3perTask:packetcheap-localcheap,packetfull-packetcheap,packetfull-reject;crossedactorseed/envbootstrapafterall3seeds',reward_scale_reference=.25,native_horizon=500,offsamples=0,training_offsamples=4,decision='allconditionswrittenbeforefuturebranchoutcomes;dedupexactequalchosenplans',conditional_scope='alreadyforcedrequest/arrivedpacket,singlenativeintervention,syntheticstepdelay;notnetacquisitionorrealasync',confidence='6contrastsacrosstasks;30000crossedresamples,confidence1-.05/6;only3actorseeds',samples_fixed=True,latency_scope='prefix/source/arrivalworker andCPUcritic;tailnotbenchmarked,noendtoendnetcostclaim'))
    rows=[];native_steps=0;latencies=[];choice_hashes=[];unique_count=0
    try:
        for task,tag,base in tasks:
            source=source_path(task);manifest,path=make_manifest(task,root,ROOT,source_override=source);local=manifest['selected'];policies={14:FrozenKernelDecisions(models,task)};fixed_cheap=policies[14].analysis['best_fixed_cheap_from_train']['name'];fixed_full=policies[14].analysis['best_fixed_full_from_train']['name'];tracked_task=[Path(v)/'head.safetensors' for v in manifest['heads']]+[source/'core.safetensors'];task_hashes=file_hashes(tracked_task);save(root/(task+'_actor_sha256.json'),task_hashes)
            with tempfile.TemporaryDirectory(prefix='advice_audit_confirm_',dir='/dev/shm') as temporary:
                work=Path(temporary);client=BundleClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/visual_bundle_worker.py',path,work/'bundle');save(root/(task+'_worker_ready.json'),client.ready)
                try:
                    for item in schedules[task]:
                        seed,q,delay=item['seed'],item['query'],item['delay'];arrival=q+delay;env=ContactEnv(task,seed);camera=DeterministicCamera(env);initial_snapshot=env.snapshot();scenes={};prefix=[];request=None;packet=None;request_action=None;request_rgb_hash=None;stem=f'{task}_{seed}'
                        try:
                            while env.t<arrival:
                                t=env.t;robot=env.observe()[:4].astype('float32');image=camera.image(env);prediction,timing=client.predict(robot,image,source=t==q,features=t==q);aa=actions(task,robot,prediction,manifest);latencies.append(dict(task=task,seed=seed,phase='source' if t==q else 'local',seconds=timing['seconds']))
                                if t==q:request=visible(robot,prediction);packet=prediction['source'].copy();request_action=aa[local].copy();request_rgb_hash=hashlib.sha256(image.tobytes()).hexdigest();scenes['request']=image.copy()
                                prefix.append(env.step(aa[local]));native_steps+=1
                            robot=env.observe()[:4].astype('float32');image=camera.image(env);prediction,timing=client.predict(robot,image,source=delay==0,features=True);aa=actions(task,robot,prediction,manifest);at_arrival=visible(robot,prediction);arrival_action=aa[local].copy();latencies.append(dict(task=task,seed=seed,phase='source' if delay==0 else 'arrival_features',seconds=timing['seconds']))
                            if delay==0:request=visible(robot,prediction);packet=prediction['source'].copy();request_action=aa[local].copy();request_rgb_hash=hashlib.sha256(image.tobytes()).hexdigest();scenes['request']=image.copy()
                            scenes['arrival']=image.copy()
                            assert request is not None and packet.shape==(100,4);packet.flags.writeable=False;snapshot=env.snapshot();decision_start=time.monotonic();cell_decisions={f'n{n}/'+key:value for n,policy in policies.items() for key,value in policy.decide(policy.distances(request,at_arrival,packet,q,delay)).items()};decision_seconds=time.monotonic()-decision_start
                            decisions={key:value['choice'] for key,value in cell_decisions.items()};decisions.update(reject='reject',fixed_cheap=fixed_cheap,fixed_full=fixed_full,elapsed_suffix48='act_suffix48')
                            choice_path=root/(stem+'_choices_before_outcomes.json');choice_record=dict(**item,task=task,arrival=arrival,decisions=decisions,cell_predictions=cell_decisions,decision_seconds=decision_seconds,source_sha256=hashlib.sha256(packet.tobytes()).hexdigest(),request_rgb_sha256=request_rgb_hash,arrival_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),written_before_future=True,offline_diagnostic_prefix_any_success=bool(snapshot['seen_success']),created_utc=now());save(choice_path,choice_record);choice_hash=hashlib.sha256(choice_path.read_bytes()).hexdigest();choice_hashes.append(dict(file=choice_path.name,sha256=choice_hash))
                            np.savez_compressed(root/(stem+'_observed_inputs.npz'),**{'request_'+k:v for k,v in request.items()},**{'arrival_'+k:v for k,v in at_arrival.items()},source_packet=packet)
                            unique=list(policies[14].analysis['models']['cheap_inputs']['decision_cells']['all_actions']['allowed_names']) if item.get('headroom',False) else list(dict.fromkeys(decisions.values()));np.random.default_rng(item['branch_order_seed']).shuffle(unique);outcomes={};traces={}
                            for name in unique:
                                env.restore(snapshot);reward=list(prefix);trace=[];hold=0 if name=='reject' else (16 if name.endswith('16') else 48)
                                while env.t<500:
                                    elapsed=env.t-arrival;robot=env.observe()[:4].astype('float32');live_image=camera.image(env);prediction,timing=client.predict(robot,live_image);aa=actions(task,robot,prediction,manifest);live=aa[local]
                                    if item.get('scene_capture',False) and elapsed==48 and name in ('reject',decisions['n14/with_packet/all_actions']):scenes['branch_'+name+'_after48']=live_image.copy()
                                    if elapsed>=hold:action=live
                                    elif name.startswith('cheap_head'):action=aa[int(name[len('cheap_head')])]
                                    elif name=='cheap_ensemble48':action=np.mean(np.stack(aa),axis=0)
                                    elif name=='cheap_request48':action=request_action
                                    elif name=='cheap_arrival48':action=arrival_action
                                    elif name.startswith('act_raw'):action=packet[elapsed]
                                    elif name.startswith('act_suffix'):action=packet[delay+elapsed]
                                    elif name=='act_blend48':action=.5*packet[delay+elapsed]+.5*live
                                    else:raise ValueError(name)
                                    trace.append(action);reward.append(env.step(action));native_steps+=1
                                assert len(reward)==500 and min(reward)>=-1e-12 and max(reward)<=10+1e-12;outcomes[name]=metrics(reward,env);np.savez_compressed(root/(stem+'_'+name+'_trace.npz'),actions=trace,rewards=reward);unique_count+=1
                            if item.get('scene_capture',False):np.savez_compressed(root/(stem+'_scene_frames.npz'),**scenes)
                            # Complete-source competence is a separate reset trajectory.
                            env.restore(initial_snapshot);source_reward=[];source_actions=[];chunk=None
                            while env.t<500:
                                if env.t%100==0:
                                    robot=env.observe()[:4].astype('float32');prediction,_=client.predict(robot,camera.image(env),source=True);chunk=prediction['source'].copy()
                                action=chunk[env.t%100];source_actions.append(action.copy());source_reward.append(env.step(action));native_steps+=1
                            source_metrics=metrics(source_reward,env);np.savez_compressed(root/(stem+'_source100_trace.npz'),actions=source_actions,rewards=source_reward)
                            assert hashlib.sha256(choice_path.read_bytes()).hexdigest()==choice_hash
                            row=dict(task=task,**item,arrival=arrival,choices_file=choice_path.name,choices_sha256=choice_hash,decisions=decisions,unique_options=unique,conditions={condition:dict(outcomes[option]) for condition,option in decisions.items()},offline_diagnostic_prefix_any_success=bool(snapshot['seen_success']),decision_seconds=decision_seconds,all_option_outcomes=outcomes if item.get('headroom',False) else None,source_full100=source_metrics);rows.append(row);save(root/'episodes.json',rows);save(root/'choice_manifest.json',choice_hashes);save(root/'latency.json',latencies)
                            print('CONFIRMATION_PROGRESS',task,'seed',seed,'q',q,'delay',delay,'unique_options',len(unique),'completed',len(rows),'seconds',round(time.monotonic()-start,2),flush=True)
                        finally:camera.close();env.close()
                finally:client.close();shutil.copyfile(work/'bundle/worker_stderr.log',root/(task+'_worker_stderr.log'))
            assert file_hashes(tracked_task)==task_hashes
        assert len(rows)==count*2 and code==fingerprint() and file_hashes(tracked)==hashes
        for task,tag,base in tasks:assert [r['seed'] for r in rows if r['task']==task]==list(range(base,base+count))
        save(root/'checks.json',dict(checks='passed',episodes=len(rows),choices_frozen_before_outcomes=True,same_packet_all_conditions=True,source_and_policy_hashes_unchanged=True,native500=True,reward_bounds=[0,10],exact_plan_deduplication=True,smoke_only=a.smoke));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-start,native_steps=native_steps,executed_unique_option_tails=unique_count,exit_code=0)
    except BaseException as e:state.update(status='failed',ended_utc=now(),error=str(e));save(root/'run.json',state);raise
    save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
