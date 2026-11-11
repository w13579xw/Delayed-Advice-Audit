"""Shared, explicit state-observation benchmark mechanics; no visual claims."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from advice_audit.metaworld_adapter import BCLocal,ContactEnv,expert_action
OBS_IDS=np.array([0,1,2,3,4,5,6,36,37,38])
POS_IDS=np.array([0,1,2,4,5,6,36,37,38])
TASKS=('drawer-open-v3','door-open-v3','button-press-v3')
KS=(1,3,5)

def sensor(native,noise=None):
    v=np.zeros(39,dtype='float64');v[OBS_IDS]=np.asarray(native)[OBS_IDS]
    if noise is not None:v[POS_IDS]+=noise
    return v

class Teacher:
    def __init__(self,task):
        from metaworld.policies import ENV_POLICY_MAP
        self.policy=ENV_POLICY_MAP[task]()
    def action(self,obs):return expert_action(self.policy,obs)

class Locals:
    def __init__(self,path):
        path=Path(path);self.config=json.loads((path/'config.json').read_text());self.selected=self.config['selected_k']
        with np.load(path/'model.npz',allow_pickle=False) as z:
            self.models={k:BCLocal(z['obs'],z['action'],k) for k in KS}
    def action(self,obs,k=None):return self.models[self.selected if k is None else k].action(obs)
    def all(self,obs):return np.stack([self.models[k].action(obs) for k in KS])
    @property
    def names(self):return ['reject']+[f'local_k{k}_8' for k in KS if k!=self.selected]+['request_hold8','arrival_hold8','zero_hold8','expert1','expert8','expert32','blend8']

def hold_length(name):
    if name=='reject':return 0
    if name=='expert1':return 1
    if name=='expert32':return 32
    return 8

def plan_action(local,obs,name,elapsed,packet,request_action,arrival_action):
    default=local.action(obs)
    if elapsed>=hold_length(name):return default
    if name.startswith('local_k'):return local.action(obs,int(name[len('local_k')]))
    if name=='request_hold8':return request_action
    if name=='arrival_hold8':return arrival_action
    if name=='zero_hold8':return np.zeros(4)
    if name.startswith('expert'):return packet
    if name=='blend8':return .5*packet+.5*default
    raise ValueError(name)

def feature_groups(request,arrival,q,d,sigma,packet):
    # The only inputs are the declared noisy ten-coordinate states, local
    # predictions, known schedule/sensor setting and optionally the packet.
    qs=np.asarray(request['state10'],dtype='float64');aa=np.asarray(arrival['state10'],dtype='float64')
    assert qs.shape==aa.shape==(10,)
    cheap=np.r_[qs,aa,aa-qs,np.asarray(request['bc']).ravel(),np.asarray(arrival['bc']).ravel(),q/500,d/100,(q+d)/500,sigma]
    assert cheap.shape==(58,)
    return cheap,np.r_[cheap,np.asarray(packet,dtype='float64')]

class Critic:
    def __init__(self,path):
        self.path=Path(path);self.config=json.loads((self.path/'config.json').read_text());self.names=self.config['names'];self.allowed=self.config['allowed']
        with np.load(self.path/'model.npz',allow_pickle=False) as z:self.arrays={k:z[k].copy() for k in z.files}
    def scores(self,x):
        if self.config['model']=='zero':return np.zeros(len(self.names))
        z=(np.asarray(x,dtype='float64')-self.arrays['mean'])/self.arrays['std']
        z=np.tanh(z@self.arrays['w1'].T+self.arrays['b1']);z=np.tanh(z@self.arrays['w2'].T+self.arrays['b2'])
        y=z@self.arrays['w3'].T+self.arrays['b3'];y[0]=0.;assert np.isfinite(y).all();return y
    def decide(self,x):
        y=self.scores(x);i=max(self.allowed,key=lambda k:(y[k],-k));return self.names[i] if y[i]>1e-8 else 'reject'

def metrics(rewards,env):
    assert len(rewards)==500
    return dict(native_return=float(np.mean(rewards)),success=int(env.seen_success),terminal_success=int(env.last_success))

def noise_stream(seed,sigma):return np.random.default_rng(seed).normal(0,sigma,(500,9))

def visible(obs,local):return dict(state10=obs[OBS_IDS].copy(),bc=local.all(obs))
