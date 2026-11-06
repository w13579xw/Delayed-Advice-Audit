"""Replay immutable native demos into image-only frozen visual features."""
import argparse,hashlib,json,os,time,shutil
from pathlib import Path
import numpy as np
from advice_audit.act_bridge import ACTClient,ACTCamera
from advice_audit.metaworld_adapter import ContactEnv
from server_run import ROOT,fingerprint,save,now

SOURCES=[('push-v3','outputs/contact/capability_20261102T154658Z/push-v3_demos.npz',0),('pick-place-v3','outputs/learned_advice/task_development_20261103_v1/pick-place-v3_demos.npz',6100000)]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--demos',type=int,default=1);p.add_argument('--stride',type=int,default=2);p.add_argument('--batch',type=int,default=16);a=p.parse_args();root=Path(a.out).resolve()
 if root.exists():p.error('new directory required')
 if not 1<=a.demos<=200 or not 1<=a.stride<=500 or a.batch<1:p.error('invalid size')
 assert shutil.disk_usage(ROOT).free>20*1024**3
 root.mkdir(parents=True);code=fingerprint();t0=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
 assets=ROOT/'outputs/vision/encoder_assets_20261103';save(root/'protocol.json',dict(config=vars(a),frozen_utc=now(),sources=[dict(task=t,path=s,environment_seed_offset=o,sha256=hashlib.sha256((ROOT/s).read_bytes()).hexdigest()) for t,s,o in SOURCES],encoder_assets=str(assets),encoder_sha256=json.loads((assets/'manifest.json').read_text())['encoder_sha256'],online_inputs=['corner4RGB512verticalflip','robot4D'],offline_targets=['native_action4D','native_object_goal6D'],sample_steps=list(range(0,500,a.stride)),native_horizon=500,precision='FP32encoder->float16features',failures_filtered=False))
 for file in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  d=root/'source_snapshot'/file.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(file.read_bytes())
 records=[];client=None;timings=[]
 try:
  client=ACTClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/vision_feature_worker.py',assets,root/'ipc');save(root/'worker_ready.json',client.ready)
  for task,path,offset in SOURCES:
   data=np.load(ROOT/path,allow_pickle=False);obs,actions,episodes=(data[k] for k in ('obs','action','episode'));ids=sorted(set(episodes.tolist()))[:a.demos];task_dir=root/task;task_dir.mkdir()
   for ep in ids:
    start=time.monotonic();indices=np.flatnonzero(episodes==ep);assert len(indices)==500;env=ContactEnv(task,int(ep)+offset);camera=ACTCamera(env);features=[];images=[];robot=[];poses=[];targets=[];steps=[];hashes=[];rewards=[]
    def flush():
     if not images:return
     begin=time.monotonic();f,timing=client.predict(np.asarray(robot[-len(images):],dtype='float32'),np.stack(images));timings.append(dict(worker_seconds=timing['seconds'],wall_seconds=time.monotonic()-begin,batch=len(images)));assert f.shape==(len(images),512,16,16) and f.dtype==np.float16 and np.isfinite(f).all();features.append(f);images.clear()
    try:
     for step,index in enumerate(indices):
      actual=env.observe();np.testing.assert_array_equal(actual,obs[index])
      if step%a.stride==0:
       begin=time.monotonic();image=camera.image(env);timings.append(dict(render_seconds=time.monotonic()-begin));images.append(image);hashes.append(hashlib.sha256(image.tobytes()).hexdigest());robot.append(actual[:4].astype('float32'));poses.append(np.r_[actual[4:7],actual[-3:]].astype('float32'));targets.append(actions[index].astype('float32'));steps.append(step)
       if step==0 and ep==ids[0]:np.save(task_dir/'first_rgb.npy',image)
       if len(images)==a.batch:flush()
      rewards.append(env.step(actions[index]))
     flush();f=np.concatenate(features);assert f.shape==(len(steps),512,16,16)
     stem=f'episode_{int(ep):08d}';feature_path=task_dir/(stem+'_features.npy');target_path=task_dir/(stem+'_labels.npz');np.save(feature_path,f);np.savez_compressed(target_path,robot=robot,target_action=targets,target_pose=poses,step=steps,episode=np.full(len(steps),ep),environment_seed=np.full(len(steps),int(ep)+offset))
     row=dict(task=task,episode=int(ep),environment_seed=int(ep)+offset,rows=len(steps),success=int(env.seen_success),terminal_success=int(env.last_success),native_return=float(np.mean(rewards)),seconds=time.monotonic()-start,features=str(feature_path.relative_to(root)),targets=str(target_path.relative_to(root)),feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),rgb_sha256=hashes,source_observations_bitwise_equal=True)
     records.append(row);save(root/'episodes.json',records);save(root/'timings.json',timings);print('DEMO',task,ep,'rows',len(steps),'seconds',round(row['seconds'],3),flush=True)
    except BaseException as e:
     save(root/'failed_episode.json',dict(task=task,episode=int(ep),step=env.t,error=str(e)));raise
    finally:camera.close();env.close()
  client.close();client=None;assert code==fingerprint();save(root/'checks.json',dict(checks='passed',all_source_states_bitwise_equal=True,complete_episodes=len(records),feature_rows=sum(r['rows'] for r in records),native_horizon=500,no_privileged_online_input=True));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-t0,exit_code=0)
 except BaseException as e:
  state.update(status='failed',error=str(e),ended_utc=now());save(root/'run.json',state)
  if client is not None:client.close()
  raise
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
