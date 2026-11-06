"""Evaluate frozen visual heads prospectively, with a separate pose firewall."""
import argparse,hashlib,json,tempfile,shutil,time
from pathlib import Path
import numpy as np
from metaworld.policies import ENV_POLICY_MAP
from advice_audit.act_bridge import ACTClient,ACTCamera
from advice_audit.metaworld_adapter import ContactEnv,expert_action
from server_run import ROOT,fingerprint,save,now
from strong_local_baseline import summary


def pose_action(task,robot4,predicted_pose6):
 if task not in ('push-v3','pick-place-v3'):raise ValueError('Pose input schema only verified for push/pick-place policies')
 if np.shape(robot4)!=(4,) or np.shape(predicted_pose6)!=(6,):raise ValueError('No native object-state input accepted')
 synthetic=np.zeros(39,dtype='float64');synthetic[:4]=robot4;synthetic[4:7]=predicted_pose6[:3];synthetic[-3:]=predicted_pose6[3:]
 return expert_action(ENV_POLICY_MAP[task](),synthetic)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,required=True);p.add_argument('--offsamples',type=int,choices=[0,4],default=None);a=p.parse_args();model=Path(a.model).resolve();root=Path(a.out).resolve();config=json.loads((model/'config.json').read_text());task=config['task'];assert task in ('push-v3','pick-place-v3')
 if root.exists():p.error('new directory required')
 if config.get('overfit_debug'):p.error('Debug-overfit checkpoint cannot be used for competence claims')
 root.mkdir(parents=True);code=fingerprint();t0=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
 save(root/'protocol.json',dict(config=vars(a),model_config=config,head_sha256=hashlib.sha256((model/'head.safetensors').read_bytes()).hexdigest(),frozen_utc=now(),native_horizon=500,primary='native any-success',input_firewall='separate GPU worker receives RGB512+robot4only; object/goal read after decision solely for diagnostic trace',ipc='new task-owned /dev/shm temporary directory; copied stderr retained',control='one visual prediction every native step'))
 for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  d=root/'source_snapshot'/f.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(f.read_bytes())
 rows=[];latencies=[];client=None;temporary=tempfile.TemporaryDirectory(prefix='advice_audit_visual_',dir='/dev/shm')
 try:
  client=ACTClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/vision_policy_worker.py',model,Path(temporary.name)/'ipc');save(root/'worker_ready.json',client.ready)
  for seed in range(a.seed,a.seed+a.episodes):
   env=ContactEnv(task,seed);env.env.model.vis.quality.offsamples=a.offsamples if a.offsamples is not None else env.env.model.vis.quality.offsamples;camera=ACTCamera(env);actions=[];rewards=[];predictions=[];robots=[];offline_pose=[];rgb_hash=[];first=None;snapshots=[]
   try:
    while env.t<500:
     step=env.t;robot=env.observe()[:4].astype('float32');image=camera.image(env);begin=time.monotonic();prediction,timing=client.predict(robot,image);latencies.append(dict(seed=seed,step=step,model_seconds=timing['seconds'],rpc_seconds=time.monotonic()-begin));prediction=prediction[0]
     if config['target']=='pose':action=pose_action(task,robot,prediction)
     else:assert prediction.shape==(4,);action=np.clip(prediction,-1,1)
     # Privileged diagnostics are recorded only after the action is selected.
     native=env.observe();offline_pose.append(np.r_[native[4:7],native[-3:]]);robots.append(robot);predictions.append(prediction);actions.append(action);rgb_hash.append(hashlib.sha256(image.tobytes()).hexdigest())
     if seed==a.seed and step in (0,50,100,200,499):snapshots.append(image)
     rewards.append(env.step(action))
     if env.seen_success and first is None:first=env.t
    row=dict(seed=seed,success=int(env.seen_success),terminal_success=int(env.last_success),native_return=float(np.mean(rewards)),first_success_step=first,predictions=len(predictions));rows.append(row);save(root/'episodes.json',rows);save(root/'latency.json',latencies);save(root/f'episode_{seed}_input_hashes.json',rgb_hash);np.savez_compressed(root/f'episode_{seed}_trace.npz',action=actions,reward=rewards,prediction=predictions,robot=robots,offline_only_groundtruth_pose=offline_pose)
    if snapshots:np.savez_compressed(root/'first_episode_frames.npz',rgb=snapshots)
    print('EPISODE',task,model.name,row,flush=True)
   except BaseException as e:
    save(root/'failed_episode.json',dict(seed=seed,step=env.t,error=str(e)));np.savez_compressed(root/'failed_episode_trace.npz',action=actions,reward=rewards,prediction=predictions);raise
   finally:camera.close();env.close()
  client.close();shutil.copyfile(Path(temporary.name)/'ipc/worker_stderr.log',root/'worker_stderr.log');client=None
  assert code==fingerprint();save(root/'analysis.json',dict(metrics=summary(rows),episodes=len(rows),target=config['target'],model_seconds_median=float(np.median([v['model_seconds'] for v in latencies])),rpc_seconds_median=float(np.median([v['rpc_seconds'] for v in latencies])),no_privileged_deployed_inputs=True));save(root/'checks.json',dict(checks='passed',all_native500=True,task=task,episodes=len(rows),input_robot_dim=4,mode=config['target']));state.update(status='complete',completed_utc=now(),exit_code=0,seconds=time.monotonic()-t0)
 except BaseException as e:
  state.update(status='failed',error=str(e),ended_utc=now());save(root/'run.json',state)
  if client is not None:
   client.close();shutil.copyfile(Path(temporary.name)/'ipc/worker_stderr.log',root/'worker_stderr.log')
  raise
 finally:temporary.cleanup()
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
