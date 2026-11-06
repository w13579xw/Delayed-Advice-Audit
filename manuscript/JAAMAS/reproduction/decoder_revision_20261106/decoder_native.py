"""Independent generation-time decoding control; original channel study stays untouched."""
import os,sys,json,time,hashlib,tempfile,shutil,argparse
from pathlib import Path
import numpy as np
R=Path('/home/ez/DACI/code');sys.path[:0]=[str(R/'src'),str(R/'scripts'),str(R/'outputs/jaamas/revision_20261105'),str(R/'outputs/jaamas/bridge_revision_20261105')]
from visual_channel import choose,rule,visible,MODELS,reference
from replication_support import make_manifest,source_path
from daci.metaworld_adapter import ContactEnv
from daci.value_kernel import FrozenKernelDecisions
from visual_bundle import BundleClient,DeterministicCamera,actions
from server_run import save,now,fingerprint
os.environ['DACI_REPLICATION_ACTOR_SEED']='0'
TASK='pick-place-v3';PF='with_packet/all_actions'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--mode',choices=['smoke','identity','main'],required=True);args=ap.parse_args();O=Path(args.out);O.mkdir(parents=True,exist_ok=False);start=time.monotonic();fp=fingerprint();save(O/'run.json',dict(status='running',utc=now(),mode=args.mode))
 manifest,mp=make_manifest(TASK,O,R);policy=FrozenKernelDecisions(MODELS,TASK)
 tracked=[*MODELS.glob('*.json'),*MODELS.glob('*.npz'),*[Path(h)/n for h in manifest['heads'] for n in ['config.json','head.safetensors']],*[source_path(TASK,0)/n for n in ['metadata.json','core.safetensors','normalization.json','source_config.json']]]
 hashes={str(f):sha(f) for f in tracked};entries={str(f):sha(f) for f in [Path(__file__),R/'outputs/jaamas/bridge_revision_20261105/visual_channel.py',reference,R/'outputs/jaamas/revision_20261105/replication_support.py']}
 if args.mode=='main':schedule=[dict(seed=890010000+100*(di*3+qi)+i,q=q,delay=d,cell=di*3+qi,index=i) for di,d in enumerate([16,32]) for qi,q in enumerate([16,32,64]) for i in range(10)]
 elif args.mode=='identity':schedule=[dict(seed=889010100+10*j+i,q=q,delay=0,cell=j,index=i) for j,q in enumerate([16,32,64]) for i in range(2)]
 else:schedule=[dict(seed=889019900+i,q=32,delay=d,cell=i,index=0) for i,d in enumerate([0,32])]
 save(O/'protocol.json',dict(frozen_utc=now(),schedule=schedule,model_hashes=hashes,entry_hashes=entries,production_fingerprint=fp,original_rule=rule,task=TASK,actor_seed=0,gate=PF,primary=['fixed_F_time-fixed_F_legacy','fixed_F_time-fixed_S_legacy','gate_F_time-gate_F_legacy','gate_F_time-gate_S_legacy'],inference=dict(resamples=30000,rng=2026110601,confidence=.9875,family_size=4,unit='wholepaired60environmentsinsixcells;notbranches;noactorseedresampling'),decoder=dict(legacy='d+elapsed for suffix/blend',time='a-generation+elapsed;Sgenerationq,Fgenerationa;raw/localunchanged'),boundary='Newexecutionsemanticscontrol;frozenlegacytrainedgate;NOToriginalfixedreceiverchannelestimand;nogateagefeaturechange;noextraFgenerationlatency/cost/privateinfo'))
 shutil.copy2(__file__,O/'entry_snapshot.py');rows=[];steps=0;parity=[]
 try:
  with tempfile.TemporaryDirectory(prefix='daci_decoder_',dir='/dev/shm') as tmp:
   tmp=Path(tmp);client=BundleClient(R/'.venv-vision/bin/python',R/'scripts/visual_bundle_worker.py',mp,tmp/'bundle');save(O/'worker_ready.json',client.ready)
   try:
    for case in schedule:
     folder=O/f"case_{case['seed']}";folder.mkdir();env=ContactEnv(TASK,case['seed']);cam=DeterministicCamera(env);q=case['q'];delay=case['delay'];arrival=q+delay;prefix={k:[] for k in ['reward','action','robot4','last_success']};request=stale=rqact=None
     try:
      while env.t<arrival:
       t=env.t;robot=env.observe()[:4].astype('float32');pred,_=client.predict(robot,cam.image(env),source=t==q,features=t==q);aa=actions(TASK,robot,pred,manifest);act=aa[manifest['selected']]
       if t==q:request=visible(robot,pred);stale=pred['source'].copy();rqact=act.copy()
       prefix['robot4'].append(robot.copy());prefix['action'].append(act.copy());prefix['reward'].append(env.step(act));prefix['last_success'].append(int(env.last_success));steps+=1
      robot=env.observe()[:4].astype('float32');pred,_=client.predict(robot,cam.image(env),source=True,features=True);ar=visible(robot,pred);aract=actions(TASK,robot,pred,manifest)[manifest['selected']].copy();fresh=pred['source'].copy()
      if delay==0:request=visible(robot,pred);stale=fresh.copy();rqact=aract.copy()
      if args.mode=='smoke':
       plain,_=client.predict(robot,cam.image(env),source=False,features=True)
       for k in ['head0','head1','head2','head3','features']:np.testing.assert_array_equal(plain[k],pred[k])
       parity.append(case['seed'])
      packets={'S':stale,'F':fresh};assert stale.shape==fresh.shape==(100,4);generations={'S':q,'F':arrival};decisions={ch:policy.decide(policy.distances(request,ar,packet,q,delay)) for ch,packet in packets.items()};snap=env.snapshot();plans={};mapping={}
      def plan(condition,name,ch,decoder):
       age=delay if decoder=='legacy' else arrival-generations[ch]
       assert age>=0 and (ch!='S' or age==delay)
       packet=packets[ch];packetsha=hashlib.sha256(packet.tobytes()).hexdigest()
       usesage=name.startswith('act_suffix') or name=='act_blend48'
       key=name+('/'+packetsha+'/'+str(age if usesage else 0) if name.startswith('act_') else '/local')
       mapping[condition]=key
       if key not in plans:plans[key]=dict(name=name,channel=ch,decoder=decoder,effective_age=age,packet=packet)
      plan('R','reject','S','legacy')
      for ch,decoder in [('S','legacy'),('F','legacy'),('F','time')]:
       plan(f'fixed_{ch}_{decoder}','act_suffix48',ch,decoder)
       plan(f'gate_{ch}_{decoder}',decisions[ch][PF]['choice'],ch,decoder)
      if delay==0:
       np.testing.assert_array_equal(stale,fresh);assert decisions['S']==decisions['F']
       for kind in ['fixed','gate']:assert mapping[kind+'_S_legacy']==mapping[kind+'_F_legacy']==mapping[kind+'_F_time']
      record=dict(**case,arrival=arrival,generation_steps=generations,decisions=decisions,plan_mapping=mapping,plans={k:{kk:vv for kk,vv in p.items() if kk!='packet'} for k,p in plans.items()},source_packet_hashes={ch:hashlib.sha256(packet.tobytes()).hexdigest() for ch,packet in packets.items()},written_before_future=True,utc=now())
      save(folder/'choices_before_outcomes.json',record);recordsha=sha(folder/'choices_before_outcomes.json')
      np.savez_compressed(folder/'observed_inputs.npz',**{'request_'+k:v for k,v in request.items()},**{'arrival_'+k:v for k,v in ar.items()},packet_S=stale,packet_F=fresh)
      order=list(plans);np.random.default_rng(case['seed']+41).shuffle(order);branches={}
      for bi,key in enumerate(order):
       p=plans[key];name=p['name'];packet=p['packet'];age=p['effective_age'];hold=0 if name=='reject' else (16 if name.endswith('16') else 48);env.restore(snap);tr={k:list(v) for k,v in prefix.items()}
       while env.t<500:
        t=env.t;elapsed=t-arrival;rob=env.observe()[:4].astype('float32');pr,_=client.predict(rob,cam.image(env));aa=actions(TASK,rob,pr,manifest);act=choose(name,elapsed,hold,aa,aa[manifest['selected']],packet,age,rqact,aract)
        if name=='act_suffix48' and elapsed<48:np.testing.assert_array_equal(act,packet[age+elapsed])
        tr['robot4'].append(rob.copy());tr['action'].append(act.copy());tr['reward'].append(env.step(act));tr['last_success'].append(int(env.last_success));steps+=1
       rr=np.array(tr['reward']);ss=np.array(tr['last_success']);assert rr.shape==(500,) and np.isfinite(rr).all() and min(rr)>=0 and max(rr)<=10+1e-12
       file=f'branch{bi:02d}.npz';np.savez_compressed(folder/file,**tr);branches[key]=dict(file=file,name=name,effective_age=age,native_return=float(rr.mean()),success=int(ss.any()),terminal_success=int(ss[-1]))
      env.restore(snap)
      with np.load(folder/branches[mapping['R']]['file']) as z:
       while env.t<500:
        t=env.t;rob=env.observe()[:4].astype('float32');np.testing.assert_array_equal(rob,z['robot4'][t]);pr,_=client.predict(rob,cam.image(env));act=actions(TASK,rob,pr,manifest)[manifest['selected']];np.testing.assert_array_equal(act,z['action'][t]);assert env.step(act)==z['reward'][t] and int(env.last_success)==z['last_success'][t];steps+=1
      assert sha(folder/'choices_before_outcomes.json')==recordsha
      row=dict(**case,arrival=arrival,folder=str(folder),choice_sha256=recordsha,branches=branches,outcomes={condition:branches[key] for condition,key in mapping.items()});rows.append(row);save(O/'rows.json',rows)
      print('DECODER',args.mode,len(rows),len(schedule),'plans',len(plans),'seconds',round(time.monotonic()-start,2),flush=True)
     finally:cam.close();env.close()
   finally:client.close();shutil.copy2(tmp/'bundle/worker_stderr.log',O/'worker_stderr.log')
  assert len(rows)==len(schedule) and fingerprint()==fp and all(sha(f)==h for f,h in hashes.items()) and all(sha(f)==h for f,h in entries.items())
  save(O/'checks.json',dict(checks='passed',cases=len(rows),native_steps=steps,recorded_paths=sum(len(r['branches']) for r in rows),all_models_entries_unchanged=True,all_reject_replayed=True,fixed48_index_assertions=True,zero_delay_identity=True,source_mode_local_parity=parity))
  save(O/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,utc=now()));print('COMPLETE',flush=True)
 except BaseException as ex:save(O/'run.json',dict(status='failed',exit_code=1,error=str(ex),utc=now()));raise
if __name__=='__main__':main()
