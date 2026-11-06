"""Frozen pre-rename consumers and current entry points must share behavior."""
import importlib
import json
import subprocess
import sys
import numpy as np
from advice_audit.protocol import canonical_hash, load_split
from advice_audit.evaluation import episode

def test_old_submodule_imports_share_current_objects():
    for module in ['model','evaluation','envs','protocol','value_kernel']:
        assert importlib.import_module('daci.'+module) is importlib.import_module('advice_audit.'+module)

def test_legacy_module_entry_point_still_runs():
    p=subprocess.run([sys.executable,'-m','daci.cli','--help'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert 'Delayed Advice Audit' in p.stdout

def test_original_split_schema_remains_loadable(tmp_path):
    record=dict(schema='daci.episode_split.v1',split_seed=2,group_ids=dict(train=list(range(12)),validation=list(range(12,15)),calibration=list(range(15,18)),test=[18,19]),unit='complete episode, including all requests/delays/branches')
    record['sha256']=canonical_hash(record)
    path=tmp_path/'legacy_split.json';path.write_text(json.dumps(record))
    masks,loaded=load_split(path,np.arange(20))
    assert loaded==record
    assert [int(masks[k].sum()) for k in ['train','validation','calibration','test']]==[12,3,3,2]
    assert np.all(sum(masks.values())==1)

def test_old_method_id_preserves_numerical_behavior():
    class FixedGate:
        def ask(self,*args,**kwargs):return True
        def accept(self,*args,**kwargs):return True
    kw=dict(seed=311,delay=4,gates=FixedGate(),horizon=32,chunk=4,budget=2)
    current,trace=episode('toy','advice_audit',**kw)
    legacy,old_trace=episode('toy','daci',**kw)
    assert current['method']=='advice_audit' and legacy['method']=='daci'
    assert trace==old_trace
    for k in current:
        if k!='method' and 'seconds' not in k:assert current[k]==legacy[k],k
