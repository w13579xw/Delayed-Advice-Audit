"""Exact image-policy-simulator replay plus accepted-worker equivalence."""
import argparse,hashlib,json,tempfile,shutil,sys,time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import numpy as np
from advice_audit.act_bridge import ACTCamera,ACTClient,source_action_to_native
from advice_audit.metaworld_adapter import ContactEnv
from visual_bundle import BundleClient,actions,make_manifest,DeterministicCamera
from server_run import save,now,fingerprint

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out).resolve()
 if root.exists():p.error('New directory required')
 root.mkdir();code=fingerprint();start=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
 for source_file in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  destination=root/'source_snapshot'/source_file.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source_file.read_bytes())
 rows=[];equivalence=[]
 try:
  for task,seedbase in [('push-v3',60400000),('pick-place-v3',60500000)]:
   manifest,path=make_manifest(task,root,ROOT)
   with tempfile.TemporaryDirectory(prefix='advice_audit_replay_',dir='/dev/shm') as temp:
    work=Path(temp);client=BundleClient(ROOT/'.venv-vision/bin/python',Path(__file__).parent/'visual_bundle_worker.py',path,work/'bundle');save(root/(task+'_worker_ready.json'),client.ready)
    try:
     for seed in range(seedbase,seedbase+2):
      env=ContactEnv(task,seed);camera=DeterministicCamera(env);snapshots={};trace=[]
      try:
       while env.t<500:
        t=env.t
        if t in (16,64,192):snapshots[t]=env.snapshot()
        image=camera.image(env);robot=env.observe()[:4].astype('float32');pred,_=client.predict(robot,image);act=actions(task,robot,pred,manifest)[manifest['selected']]
        if seed==seedbase and t==0:
         single=ACTClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/vision_policy_worker.py',Path(manifest['heads'][manifest['selected']]),work/'single')
         try:other,_=single.predict(robot,image);delta=float(np.max(np.abs(other[0]-pred[f"head{manifest['selected']}"])));assert delta<=1e-5
         finally:single.close();shutil.copyfile(work/'single/worker_stderr.log',root/(task+'_single_stderr.log'))
         source=ACTClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/matched_act_worker.py',Path(manifest['source']),work/'source')
         try:
          raw,_=source.predict(robot,image);bundle,_=client.predict(robot,image,source=True);delta_source=float(np.max(np.abs(source_action_to_native(raw[0])-bundle['source'])));assert delta_source<=1e-5
         finally:source.close();shutil.copyfile(work/'source/worker_stderr.log',root/(task+'_source_stderr.log'))
         equivalence.append(dict(task=task,head_max_error=delta,source_max_error=delta_source));save(root/'equivalence.json',equivalence)
        reward=env.step(act);trace.append(dict(image=hashlib.sha256(image.tobytes()).hexdigest(),prediction=pred[f"head{manifest['selected']}"].copy(),action=act.copy(),reward=reward,observation=env.observe(),success=env.seen_success,terminal=env.last_success))
       for cut,snapshot in snapshots.items():
        env.restore(snapshot);count=0
        while env.t<500:
         t=env.t;expected=trace[t];image=camera.image(env);robot=env.observe()[:4].astype('float32');pred,_=client.predict(robot,image);act=actions(task,robot,pred,manifest)[manifest['selected']]
         assert hashlib.sha256(image.tobytes()).hexdigest()==expected['image'],('image',task,seed,cut,t)
         np.testing.assert_array_equal(pred[f"head{manifest['selected']}"],expected['prediction']);np.testing.assert_array_equal(act,expected['action']);reward=env.step(act);assert reward==expected['reward'],('reward',task,seed,cut,t);np.testing.assert_array_equal(env.observe(),expected['observation']);assert env.seen_success==expected['success'] and env.last_success==expected['terminal'];count+=1
        rows.append(dict(task=task,seed=seed,cut=cut,replayed_steps=count,bitwise_equal=True));save(root/'cases.json',rows);print('EXACT_REPLAY',rows[-1],flush=True)
       np.savez_compressed(root/f'{task}_{seed}_uninterrupted.npz',action=[r['action'] for r in trace],reward=[r['reward'] for r in trace],observation=[r['observation'] for r in trace],prediction=[r['prediction'] for r in trace],image_hash=[r['image'] for r in trace])
      finally:camera.close();env.close()
    finally:client.close();shutil.copyfile(work/'bundle/worker_stderr.log',root/(task+'_bundle_stderr.log'))
  assert code==fingerprint();save(root/'checks.json',dict(checks='passed',cases=len(rows),bitwise_image_policy_physics_reward=True,accepted_worker_equivalence=True));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-start,exit_code=0)
 except BaseException as e:state.update(status='failed',ended_utc=now(),error=str(e));save(root/'run.json',state);raise
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
