"""Development-only resplit diagnostic; never reads confirmation outcomes."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from advice_audit.value_kernel import select_global_minimum
from server_run import save,now
PARENT=ROOT/'outputs/vision/value_data_40_20261103_v1'
ORIGINAL=ROOT/'outputs/vision/value_kernel_pilot_20261104_v3'
OUT=ROOT/'outputs/vision/development_resplits_20261104_v1'
TOLS=[0.,1e-14,1e-12,1e-10,1e-8]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load(rows):
    groups=[[] for _ in range(6)]
    for r in rows:
        with np.load(PARENT/r['request_inputs'],allow_pickle=False) as q, np.load(PARENT/r['arrival_inputs'],allow_pickle=False) as a, np.load(PARENT/r['source_packet'],allow_pickle=False) as e:
            assert set(q.files)==set(a.files)=={'visual_features','robot','head0','head1','head2','head3'}
            qr=q['robot'].astype('float64');ar=a['robot'].astype('float64')
            values=[q['visual_features'].astype('float32').ravel(),a['visual_features'].astype('float32').ravel(),
                np.r_[qr,ar,ar-qr],np.concatenate([z[k] for z in (q,a) for k in ('head0','head1','head2','head3')]).astype('float64'),
                np.array([r['query']/500,r['delay']/100,r['arrival']/500]),e['source'].astype('float64').ravel()]
            for g,v in zip(groups,values):g.append(v)
    return [np.array(g,dtype='float32') for g in groups]

def distances(x,y):return np.maximum((x*x).mean(1)[:,None]+(y*y).mean(1)[None,:]-2*(x@y.T)/x.shape[1],0.)

def fit(groups,rows,ti,vi):
    td=[];vd=[]
    for group in groups:
        t=group[ti];v=group[vi];mean=t.mean(0);std=np.maximum(t.std(0),.05)
        t=((t-mean)/std).astype('float64');v=((v-mean)/std).astype('float64')
        dt=distances(t,t);dv=distances(v,t);np.fill_diagonal(dt,0.)
        p=dt[np.triu_indices(len(t),1)];p=p[p>1e-12];scale=float(np.median(p)) if len(p) else 1.
        td.append(dt/scale);vd.append(dv/scale)
    names=list(rows[0]['outcomes']);assert names[0]=='reject' and len(names)==12
    y=np.array([[(r['outcomes'][n]['native_return']-r['outcomes']['reject']['native_return'])/10 for n in names] for r in rows])
    yt,yv=y[ti],y[vi];result={}
    for features,count in [('cheap_inputs',5),('with_packet',6)]:
        dt=sum(td[:count]);dv=sum(vd[:count]);predictions=[]
        for bw in (.25,1.,4.):
            kt=np.exp(-dt/(2*bw**2));kv=np.exp(-dv/(2*bw**2))
            for ridge in (.01,.1,1.,10.):
                dual=np.linalg.solve(kt+ridge*np.eye(len(ti)),yt);pred=kv@dual;pred[:,0]=0.
                predictions.append(dict(model='RBFkernelridge',bandwidth=bw,ridge=ridge,dual=dual,prediction=pred))
        for actions,allowed in [('cheap_actions',[i for i,n in enumerate(names) if not n.startswith('act_')]),('all_actions',list(range(12)))]:
            zero=dict(model='zero',bandwidth=None,ridge=None,dual=np.zeros_like(yt),prediction=np.zeros_like(yv),validation_mse=float(np.mean(yv[:,allowed]**2)))
            candidates=[zero]+[dict(p,validation_mse=float(np.mean((p['prediction'][:,allowed]-yv[:,allowed])**2))) for p in predictions]
            cells=[]
            for tol in TOLS:
                best=select_global_minimum(candidates,tolerance=tol);pred=best['prediction']
                ids=[max(allowed,key=lambda k:(row[k],-k)) for row in pred]
                ids=[k if pred[j,k]>1e-12 else 0 for j,k in enumerate(ids)]
                choices=[names[k] for k in ids]
                realized=np.array([rows[int(v)]['outcomes'][choice]['native_return'] for v,choice in zip(vi,choices)])
                reject=np.array([rows[int(v)]['outcomes']['reject']['native_return'] for v in vi])
                cells.append(dict(tolerance=tol,selected={k:best[k] for k in ('model','bandwidth','ridge','validation_mse','global_minimum_mse')},
                    choices=choices,all_reject=all(n=='reject' for n in choices),
                    validation_reward=float(realized.mean()),validation_reward_minus_reject=float((realized-reject).mean()),
                    null_mse=zero['validation_mse'],null_excess_mse=zero['validation_mse']-best['global_minimum_mse']))
            result[features+'/'+actions]=dict(cells=cells,candidates=candidates)
    return result

def main():
    if OUT.exists():raise RuntimeError('New output directory required')
    OUT.mkdir(parents=True);start=time.monotonic();allrows=json.loads((PARENT/'rows.json').read_text())
    rng=np.random.default_rng(2026110420);schedules={}
    for task in ('push-v3','pick-place-v3'):
        schedules[task]=[]
        for i in range(100):
            perm=rng.permutation(20);schedules[task].append(dict(split_id=i,train=sorted(perm[:14].tolist()),validation=sorted(perm[14:].tolist())))
    source_hashes={}
    for r in allrows:
        for key in ('request_inputs','arrival_inputs','source_packet'):source_hashes[r[key]]=sha(PARENT/r[key])
    save(OUT/'protocol.json',dict(frozen_utc=now(),source_rows_sha256=sha(PARENT/'rows.json'),source_input_hashes=source_hashes,
        schedules=schedules,resplits_per_task=100,whole_episode_split=True,tolerances=TOLS,rng=2026110420,
        training_only_normalization=True,selection='unchanged original global minimum tolerance pool',
        role='development resplit sensitivity; validation reused for selection, so reward is not unbiased test performance',
        no_confirmation_data_access=True,script_sha256=sha(Path(__file__))))
    (OUT/'script_snapshot.py').write_bytes(Path(__file__).read_bytes());save(OUT/'run.json',dict(status='running',created_utc=now()))
    original=json.loads((ORIGINAL/'analysis.json').read_text());records=[];audits=[]
    try:
        for task in ('push-v3','pick-place-v3'):
            rows=[r for r in allrows if r['task']==task];assert len(rows)==20
            groups=load(rows);ti=[i for i,r in enumerate(rows) if r['split']=='train'];vi=[i for i,r in enumerate(rows) if r['split']=='validation']
            ref=fit(groups,rows,ti,vi)
            for key,value in ref.items():
                features,actions=key.split('/');expected=original[task]['native_return']['models'][features]['decision_cells'][actions]
                observed=value['cells'][2];assert observed['choices']==expected['choices']
                for field in ('model','bandwidth','ridge'):assert observed['selected'][field]==expected['selected'][field]
                error=abs(observed['selected']['validation_mse']-expected['selected']['validation_mse']);assert error<=1e-14
                stem=task+'_native_return_'+features+'_'+actions
                with np.load(ORIGINAL/(stem+'_model.npz'),allow_pickle=False) as z:
                    chosen=select_global_minimum(value['candidates']);dual_error=float(np.max(np.abs(chosen['dual']-z['dual'])))
                    assert dual_error<=1e-13
                audits.append(dict(task=task,cell=key,validation_mse_error=error,dual_max_error=dual_error,choices_equal=True))
            save(OUT/'original_split_audit.json',audits)
            for item in schedules[task]:
                ti=item['train'];vi=item['validation'];assert len(ti)==14 and len(vi)==6 and set(ti).isdisjoint(vi)
                fitted=fit(groups,rows,ti,vi)
                records.append(dict(task=task,split_id=item['split_id'],train_seeds=[rows[i]['seed'] for i in ti],
                    validation_seeds=[rows[i]['seed'] for i in vi],cells={k:v['cells'] for k,v in fitted.items()}))
                save(OUT/'records.json',records)
                if (item['split_id']+1)%10==0:print('RESPLIT',task,item['split_id']+1,'seconds',round(time.monotonic()-start,2),flush=True)
        summary={}
        for task in ('push-v3','pick-place-v3'):
            rr=[r for r in records if r['task']==task];task_summary={}
            for key in rr[0]['cells']:
                cells=[]
                for j,tol in enumerate(TOLS):
                    values=[r['cells'][key][j] for r in rr];d=np.array([r['validation_reward_minus_reject'] for r in values])
                    cells.append(dict(tolerance=tol,splits=len(values),null_count=sum(r['selected']['model']=='zero' for r in values),
                        all_reject_count=sum(r['all_reject'] for r in values),
                        selected_validation_reward_minus_reject_mean=float(d.mean()),
                        selected_validation_reward_difference_q10_q50_q90=np.quantile(d,[.1,.5,.9]).tolist(),
                        not_independent_tests=True,selection_uses_this_validation_set=True))
                task_summary[key]=cells
            summary[task]=task_summary
        save(OUT/'analysis.json',summary)
        assert all(sha(PARENT/n)==h for n,h in source_hashes.items())
        save(OUT/'checks.json',dict(checks='passed',resplits=200,cells_per_resplit=20,selected_cells=4000,
             original_split_cells_reproduced=8,max_original_mse_error=max(a['validation_mse_error'] for a in audits),
             max_original_dual_error=max(a['dual_max_error'] for a in audits),no_confirmation_read=True,all_source_inputs_unchanged=True))
        save(OUT/'run.json',dict(status='complete',completed_utc=now(),seconds=time.monotonic()-start,exit_code=0))
        print('COMPLETE',OUT,flush=True)
    except BaseException as e:
        save(OUT/'run.json',dict(status='failed',ended_utc=now(),error=str(e),exit_code=1));raise

if __name__=='__main__':main()
