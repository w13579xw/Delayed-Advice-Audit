from pathlib import Path
import json,hashlib,numpy as np
R=Path('/home/ez/DACI/code');D=R/'outputs/jaamas/revision_20261105';old=R/'outputs/vision/collect_critic_curve_data_20261104.py';s=old.read_text().replace('REPO=Path(__file__).resolve().parents[2]','REPO=Path(\'/home/ez/DACI/code\')')
s=s.replace('from server_run import ROOT,fingerprint,save,now','from server_run import ROOT,fingerprint,save,now\nfrom replication_support import make_manifest,source_path')
s=s.replace("source=ROOT/'outputs/vision'/f'act_budget15k_{tag}_20261103_v1'/'best'","source=source_path(task)")
s=s.replace("assert len(rows)==(2 if a.smoke else 80)","assert len(rows)==sum(len(v) for v in schedules.values())")
(D/'collect_visual_repl.py').write_text(s)
old2=R/'outputs/vision/confirm_critic_curve_20261104.py';s=old2.read_text().replace('REPO=Path(__file__).resolve().parents[2]','REPO=Path(\'/home/ez/DACI/code\')').replace('from server_run import ROOT,fingerprint,save,now','from server_run import ROOT,fingerprint,save,now\nfrom replication_support import make_manifest,source_path')
s=s.replace("source=ROOT/'outputs/vision'/f'act_budget15k_{tag}_20261103_v1'/'best'","source=source_path(task)")
s=s.replace("assert all(json.loads((models/f'n{n}'/'checks.json').read_text())['checks']=='passed' for n in (7,14,28))","assert json.loads((models/'checks.json').read_text())['checks']=='passed'")
s=s.replace('count=2 if a.smoke else 100','count=2 if a.smoke else 50')
s=s.replace("tracked=sorted(p for n in (7,14,28) for p in (models/f'n{n}').iterdir() if p.suffix in ('.npz','.json'))","tracked=sorted(p for p in models.iterdir() if p.suffix in ('.npz','.json'))")
s=s.replace("policies={n:FrozenKernelDecisions(models/f'n{n}',task) for n in (7,14,28)};fixed_cheap=policies[28].analysis","policies={14:FrozenKernelDecisions(models,task)};fixed_cheap=policies[14].analysis").replace("fixed_full=policies[28].analysis","fixed_full=policies[14].analysis")
s=s.replace("assert len(rows)==count*2 and code==fingerprint()","assert len(rows)==count*2 and code==fingerprint()")
s=s.replace("camera=DeterministicCamera(env);prefix=[]","camera=DeterministicCamera(env);initial_snapshot=env.snapshot();scenes={};prefix=[]")
s=s.replace("if t==q:request=visible(robot,prediction);packet=prediction['source'].copy();request_action=aa[local].copy();request_rgb_hash=hashlib.sha256(image.tobytes()).hexdigest()","if t==q:request=visible(robot,prediction);packet=prediction['source'].copy();request_action=aa[local].copy();request_rgb_hash=hashlib.sha256(image.tobytes()).hexdigest();scenes['request']=image.copy()")
s=s.replace("if delay==0:request=visible(robot,prediction);packet=prediction['source'].copy();request_action=aa[local].copy();request_rgb_hash=hashlib.sha256(image.tobytes()).hexdigest()","if delay==0:request=visible(robot,prediction);packet=prediction['source'].copy();request_action=aa[local].copy();request_rgb_hash=hashlib.sha256(image.tobytes()).hexdigest();scenes['request']=image.copy()\n                            scenes['arrival']=image.copy()")
s=s.replace("unique=list(dict.fromkeys(decisions.values()));np.random.default_rng(item['branch_order_seed']).shuffle(unique)","unique=list(policies[14].analysis['models']['cheap_inputs']['decision_cells']['all_actions']['allowed_names']) if item.get('headroom',False) else list(dict.fromkeys(decisions.values()));np.random.default_rng(item['branch_order_seed']).shuffle(unique)")
s=s.replace("prediction,timing=client.predict(robot,camera.image(env));aa=actions(task,robot,prediction,manifest);live=aa[local]","live_image=camera.image(env);prediction,timing=client.predict(robot,live_image);aa=actions(task,robot,prediction,manifest);live=aa[local]\n                                    if item.get('scene_capture',False) and elapsed==48 and name in ('reject',decisions['n14/with_packet/all_actions']):scenes['branch_'+name+'_after48']=live_image.copy()")
needle="                            assert hashlib.sha256(choice_path.read_bytes()).hexdigest()==choice_hash\n"
insert="""                            if item.get('scene_capture',False):np.savez_compressed(root/(stem+'_scene_frames.npz'),**scenes)
                            # Complete-source competence is a separate reset trajectory.
                            env.restore(initial_snapshot);source_reward=[];source_actions=[];chunk=None
                            while env.t<500:
                                if env.t%100==0:
                                    robot=env.observe()[:4].astype('float32');prediction,_=client.predict(robot,camera.image(env),source=True);chunk=prediction['source'].copy()
                                action=chunk[env.t%100];source_actions.append(action.copy());source_reward.append(env.step(action));native_steps+=1
                            source_metrics=metrics(source_reward,env);np.savez_compressed(root/(stem+'_source100_trace.npz'),actions=source_actions,rewards=source_reward)
"""
assert needle in s;s=s.replace(needle,insert+needle)
s=s.replace("decision_seconds=decision_seconds);rows.append(row)","decision_seconds=decision_seconds,all_option_outcomes=outcomes if item.get('headroom',False) else None,source_full100=source_metrics);rows.append(row)")
s=s.replace("'4perTask:n28full-reject,n28full-n14full,n28packetcheap-n28localcheap,n28packetfull-n28packetcheap'","'3perTask:packetcheap-localcheap,packetfull-packetcheap,packetfull-reject;crossedactorseed/envbootstrapafterall3seeds'")
s=s.replace("'8pairedrewardcontrasts;50000pairedbootstrap,confidence1-.05/8;no finite-sample guarantee'","'6contrastsacrosstasks;30000crossedresamples,confidence1-.05/6;only3actorseeds'")
(D/'test_visual_repl.py').write_text(s)
# Freeze same original development seeds/clocks/splits; expand test schedule.
base=json.loads((R/'outputs/vision/value_data_40_20261103_v1/protocol.json').read_text())['schedules'];rng=np.random.default_rng(2026110557);payload=dict(development=base,collection_smoke={},test_smoke={},test={})
for ti,task in enumerate(['push-v3','pick-place-v3']):
 payload['collection_smoke'][task]=[dict(seed=859000000+ti*10000,query=32,delay=32,split='engineering')]
 payload['test_smoke'][task]=[dict(seed=859000100+ti*10000+i,query=q,delay=d,branch_order_seed=859100100+ti*10000+i,headroom=True,scene_capture=i==0) for i,(q,d) in enumerate([(32,32),(16,0)])]
 payload['test'][task]=[dict(seed=860000000+ti*10000+i,query=int(rng.choice([16,32,64])),delay=int(rng.choice([0,16,32])),branch_order_seed=int(rng.integers(1,2**31-1)),headroom=i<10,scene_capture=False) for i in range(50)]
(D/'visual_schedule.json').write_text(json.dumps(payload,indent=2));(D/'generation_manifest.json').write_text(json.dumps(dict(parent_collect=str(old),parent_collect_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),parent_test=str(old2),parent_test_sha256=hashlib.sha256(old2.read_bytes()).hexdigest(),note='Output-only parameterized replicas; no production scripts edited. No lazy prediction omission used.'),indent=2))
