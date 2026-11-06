"""Independent native replay of every observed request/arrival source generation."""
import json,sys,tempfile,hashlib,os
from pathlib import Path
import numpy as np
R=Path('/home/ez/DACI/code');sys.path[:0]=[str(R/'src'),str(R/'scripts'),str(R/'outputs/jaamas/revision_20261105')]
from daci.metaworld_adapter import ContactEnv
from replication_support import make_manifest,source_path
from visual_bundle import BundleClient,DeterministicCamera,actions
os.environ['DACI_REPLICATION_ACTOR_SEED']='0'
B=R/'outputs/jaamas/visual_channel_bridge_20261105_v1';O=R/'outputs/jaamas/visual_channel_source_audit_20261105_v1';O.mkdir(exist_ok=False)
manifest,mp=make_manifest('pick-place-v3',O,R,source_override=source_path('pick-place-v3',0));count=steps=0
with tempfile.TemporaryDirectory(dir='/dev/shm',prefix='visual_source_audit_') as tmp:
 client=BundleClient(R/'.venv-vision/bin/python',R/'scripts/visual_bundle_worker.py',mp,Path(tmp)/'bundle')
 try:
  for mode in ['smoke','identity','main']:
   assert json.loads((B/mode/'run.json').read_text())['status']=='complete'
   for row in json.loads((B/mode/'rows.json').read_text()):
    folder=Path(row['folder']);d=json.loads((folder/'choices_before_outcomes.json').read_text());base=next(b for b in row['branches'].values() if b['name']=='reject')
    with np.load(folder/'observed_inputs.npz') as zz:obs={k:zz[k].copy() for k in zz.files}
    with np.load(folder/base['file']) as zz:trace={k:zz[k].copy() for k in zz.files}
    env=ContactEnv('pick-place-v3',row['seed']);cam=DeterministicCamera(env)
    try:
     while env.t<=row['arrival']:
      t=env.t;robot=env.observe()[:4].astype('float32');np.testing.assert_array_equal(robot,trace['robot4'][t]);isq=t==row['q'];isa=t==row['arrival'];pred,_=client.predict(robot,cam.image(env),source=isq or isa,features=isq or isa)
      for flag,prefix,ch in [(isq,'request_','S'),(isa,'arrival_','F')]:
       if flag:
        np.testing.assert_array_equal(pred['source'],obs['packet_'+ch]);np.testing.assert_array_equal(pred['features'],obs[prefix+'visual_features'])
        for h in range(4):np.testing.assert_array_equal(pred[f'head{h}'],obs[prefix+f'head{h}'])
        count+=1
      if isa:break
      act=actions('pick-place-v3',robot,pred,manifest)[manifest['selected']];np.testing.assert_array_equal(act,trace['action'][t]);assert env.step(act)==trace['reward'][t];steps+=1
    finally:cam.close();env.close()
    print(mode,row['seed'],'checked',count,flush=True)
 finally:client.close()
assert count==136
freeze=json.loads((B/'main/protocol.json').read_text())
for file,h in freeze['model_hashes'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h
(O/'checks.json').write_text(json.dumps(dict(checks='passed',independent_native_prefix_steps=steps,source_and_features_checked=count,expected_cases=68,bitwise_packet_features_heads_robots_actions_rewards=True,entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
