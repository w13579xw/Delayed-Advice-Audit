"""Frozen original PickPlace visual receiver, literal native S/F/absence channels."""
import argparse,json,hashlib,sys,time,tempfile,shutil,os,textwrap
from pathlib import Path
import numpy as np
R=Path('/home/ez/DelayedAdviceAudit/code');sys.path[:0]=[str(R/'src'),str(R/'scripts'),str(R/'outputs/jaamas/revision_20261105')]
from server_run import save,now,fingerprint
from advice_audit.metaworld_adapter import ContactEnv
from advice_audit.value_kernel import FrozenKernelDecisions
from visual_bundle import BundleClient,DeterministicCamera,actions
from replication_support import make_manifest,source_path
os.environ['ADVICE_AUDIT_REPLICATION_ACTOR_SEED']='0'
TASK='pick-place-v3';MODELS=R/'outputs/vision/value_kernel_pilot_20261104_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def visible(robot,p):return dict(robot=robot.copy(),visual_features=p['features'].copy(),**{f'head{i}':p[f'head{i}'].copy() for i in range(4)})
# Extract the exact original action-rule conditional, not a new temporal decoder.
reference=R/'outputs/jaamas/revision_20261105/test_visual_repl.py'
text=reference.read_text();begin=text.index('if elapsed>=hold:action=live');end=text.index('else:raise ValueError(name)',begin)+len('else:raise ValueError(name)')
rule=textwrap.dedent(' '*36+text[begin:end])
# First line was extracted without its original indentation; align all clauses explicitly.
lines=text[begin:end].splitlines();indent=len(lines[1])-len(lines[1].lstrip());rule='\n'.join([lines[0]]+[line[indent:] for line in lines[1:]])
exec('def choose(name,elapsed,hold,aa,live,packet,delay,request_action,arrival_action):\n'+textwrap.indent(rule,'    ')+'\n    return action\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['smoke','identity','main'],required=True);a=p.parse_args();O=Path(a.out);O.mkdir(parents=True,exist_ok=False);t0=time.monotonic();fp=fingerprint();save(O/'run.json',dict(status='running',utc=now(),mode=a.mode));manifest,mp=make_manifest(TASK,O,R,source_override=source_path(TASK,0));policy=FrozenKernelDecisions(MODELS,TASK)
 tracked=[*MODELS.glob('*.json'),*MODELS.glob('*.npz'),*[Path(h)/n for h in manifest['heads'] for n in ['config.json','head.safetensors']],*[source_path(TASK,0)/n for n in ['metadata.json','core.safetensors','normalization.json','source_config.json']]]
 hashes={str(x):sha(x) for x in tracked};schedule=[]
 if a.mode=='main':
  cell=0
  for d in [16,32]:
   for q in [16,32,64]:
    for i in range(10):schedule.append(dict(seed=870010000+100*cell+i,q=q,delay=d,cell=cell,index=i))
    cell+=1
 elif a.mode=='identity':
  for j,q in enumerate([16,32,64]):
   for i in range(2):schedule.append(dict(seed=869010100+10*j+i,q=q,delay=0,cell=j,index=i))
 else:schedule=[dict(seed=869019900+i,q=32,delay=d,cell=i,index=0) for i,d in enumerate([0,32])]
 save(O/'protocol.json',dict(frozen_utc=now(),schedule=schedule,task=TASK,actor_seed=0,models=str(MODELS),model_hashes=hashes,source_fingerprint=fp,entry_sha256=sha(Path(__file__)),reference_entry_sha256=sha(reference),original_action_rule_sha256=hashlib.sha256(rule.encode()).hexdigest(),original_action_rule=rule,channels=dict(S='sameACT100x4 onrequestRGB+robot4',F='sameACT100x4 onarrivalRGB+robot4,literalreplacement;unchangedrequestdelayandsuffixindexing',E='absence->predeclaredrejectwrapper'),primary=['PF:F-S','PF:S-E','PF:F-E'],inference='main60nonzerodelayenv;6equalcells;30000stratifiedpairedbootstrap;98.3333%family3;RNG2026110563',missing_wrapper='fixedbeforetests;nozeroinput',scope='fixedreceivercontentrefresh;freshjointshift;noage-awaredecoderoptimization/privateinfo/servicecost/extra refreshlatency',mode=a.mode))
 shutil.copy2(__file__,O/'entry_snapshot.py');rows=[];steps=0;parity=[];modelcells=list(policy.decide([np.zeros(14) for _ in range(6)]));count=0
 try:
  with tempfile.TemporaryDirectory(prefix='advice_audit_visual_channel_',dir='/dev/shm') as temp:
   tmp=Path(temp);client=BundleClient(R/'.venv-vision/bin/python',R/'scripts/visual_bundle_worker.py',mp,tmp/'bundle');save(O/'worker_ready.json',client.ready)
   try:
    for case in schedule:
     env=ContactEnv(TASK,case['seed']);cam=DeterministicCamera(env);q=case['q'];delay=case['delay'];arrival=q+delay;prefix_r=[];prefix_a=[];prefix_s=[];prefix_robot=[];request=None;stale=None;rqact=None
     folder=O/f"case_{case['seed']}";folder.mkdir()
     try:
      while env.t<arrival:
       t=env.t;robot=env.observe()[:4].astype('float32');img=cam.image(env);pred,_=client.predict(robot,img,source=t==q,features=t==q);aa=actions(TASK,robot,pred,manifest)
       if t==q:request=visible(robot,pred);stale=pred['source'].copy();rqact=aa[manifest['selected']].copy()
       act=aa[manifest['selected']];prefix_robot.append(robot.copy());prefix_a.append(act.copy());prefix_r.append(env.step(act));prefix_s.append(int(env.last_success));steps+=1
      robot=env.observe()[:4].astype('float32');img=cam.image(env);pred,_=client.predict(robot,img,source=True,features=True);ar=visible(robot,pred);aa=actions(TASK,robot,pred,manifest);fresh=pred['source'].copy();aract=aa[manifest['selected']].copy()
      if delay==0:request=visible(robot,pred);stale=fresh.copy();rqact=aract.copy();np.testing.assert_array_equal(stale,fresh)
      if a.mode=='smoke':
       plain,_=client.predict(robot,img,source=False,features=True)
       for k in ['head0','head1','head2','head3','features']:np.testing.assert_array_equal(plain[k],pred[k])
       parity.append(dict(seed=case['seed'],source_mode_does_not_change_local_predictions=True))
      assert stale.shape==fresh.shape==(100,4);stale.flags.writeable=False;fresh.flags.writeable=False;snap=env.snapshot();packets={'S':stale,'F':fresh};decisions={ch:policy.decide(policy.distances(request,ar,pa,q,delay)) for ch,pa in packets.items()}
      decisions['E']={key:dict(choice='reject',predicted_gain=None,null_model=None) for key in modelcells}
      for inp in ['cheap_inputs/cheap_actions','cheap_inputs/all_actions']:assert decisions['S'][inp]['choice']==decisions['F'][inp]['choice']
      def key(name,pa):return name+('/'+hashlib.sha256(pa.tobytes()).hexdigest() if name.startswith('act_') else '/local')
      plans={key('reject',stale):dict(name='reject',packet=stale)};mapping={}
      for ch in ['S','F','E']:
       pa=packets.get(ch,stale);mapping[ch]={}
       for label,dec in decisions[ch].items():
        kk=key(dec['choice'],pa);mapping[ch][label]=kk;plans.setdefault(kk,dict(name=dec['choice'],packet=pa))
      record=dict(**case,arrival=arrival,decisions=decisions,plan_keys=mapping,source_packet_hashes={ch:hashlib.sha256(pa.tobytes()).hexdigest() for ch,pa in packets.items()},written_before_future=True,utc=now())
      save(folder/'choices_before_outcomes.json',record);choicehash=sha(folder/'choices_before_outcomes.json')
      np.savez_compressed(folder/'observed_inputs.npz',**{'request_'+k:v for k,v in request.items()},**{'arrival_'+k:v for k,v in ar.items()},packet_S=stale,packet_F=fresh)
      keys=list(plans);np.random.default_rng(case['seed']+37).shuffle(keys);branches={}
      for j,kk in enumerate(keys):
       pl=plans[kk];name=pl['name'];pa=pl['packet'];hold=0 if name=='reject' else (16 if name.endswith('16') else 48);env.restore(snap);rr=list(prefix_r);acts=list(prefix_a);ss=list(prefix_s);robots=list(prefix_robot)
       while env.t<500:
        t=env.t;rob=env.observe()[:4].astype('float32');pr,_=client.predict(rob,cam.image(env));aa=actions(TASK,rob,pr,manifest);act=choose(name,t-arrival,hold,aa,aa[manifest['selected']],pa,delay,rqact,aract);robots.append(rob.copy());acts.append(act.copy());rr.append(env.step(act));ss.append(int(env.last_success));steps+=1
       assert len(rr)==500 and np.isfinite(rr).all() and min(rr)>=-1e-12 and max(rr)<=10+1e-12
       f=f'branch{j:02d}.npz';np.savez_compressed(folder/f,reward=rr,action=acts,last_success=ss,robot4=robots);branches[kk]=dict(file=f,name=name,native_return=float(np.mean(rr)),success=int(any(ss)),terminal_success=int(ss[-1]));count+=1
      # Independently repeat the complete reject suffix from the same native snapshot.
      base=branches[key('reject',stale)];env.restore(snap)
      with np.load(folder/base['file']) as z:
       while env.t<500:
        t=env.t;rob=env.observe()[:4].astype('float32');np.testing.assert_array_equal(rob,z['robot4'][t]);pr,_=client.predict(rob,cam.image(env));act=actions(TASK,rob,pr,manifest)[manifest['selected']];np.testing.assert_array_equal(act,z['action'][t]);assert env.step(act)==z['reward'][t] and int(env.last_success)==z['last_success'][t];steps+=1
      assert sha(folder/'choices_before_outcomes.json')==choicehash
      outcomes={ch:{label:branches[kk] for label,kk in mm.items()} for ch,mm in mapping.items()}
      if delay==0:
       for label in modelcells:assert mapping['S'][label]==mapping['F'][label]
      row=dict(**case,arrival=arrival,folder=str(folder),choice_sha256=choicehash,branches=branches,outcomes=outcomes);rows.append(row);save(O/'rows.json',rows);print('VISUAL_CHANNEL',a.mode,len(rows),len(schedule),'plans',len(plans),'seconds',round(time.monotonic()-t0,2),flush=True)
     finally:cam.close();env.close()
   finally:client.close();shutil.copy2(tmp/'bundle/worker_stderr.log',O/'worker_stderr.log')
  assert len(rows)==len(schedule) and fingerprint()==fp and all(sha(Path(f))==h for f,h in hashes.items());save(O/'checks.json',dict(checks='passed',cases=len(rows),native_steps=steps,recorded_paths=count,all_models_unchanged=True,reject_replayed_every_case=True,source_mode_parity=parity,zero_delay_identity=True));save(O/'run.json',dict(status='complete',exit_code=0,utc=now(),seconds=time.monotonic()-t0));print('COMPLETE',flush=True)
 except BaseException as ex:save(O/'run.json',dict(status='failed',exit_code=1,error=str(ex),utc=now()));raise
if __name__=='__main__':main()
