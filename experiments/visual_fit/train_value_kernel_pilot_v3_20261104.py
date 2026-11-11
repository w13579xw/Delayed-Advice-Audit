"""Standard-kernel2x2feasibility; episode-heldout validation,not a formal test."""
import json,hashlib,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from server_run import save,now
from strong_local_baseline import summary
from advice_audit.value_kernel import select_global_minimum
parent=ROOT/'outputs/vision/value_data_40_20261103_v1';assert json.loads((parent/'run.json').read_text())['status']=='complete';assert json.loads((parent/'checks.json').read_text())['checks']=='passed'
root=ROOT/'outputs/vision/value_kernel_pilot_20261104_v3'
if root.exists():raise RuntimeError('New run directory required')
root.mkdir();(root/'script_snapshot.py').write_bytes(Path(__file__).read_bytes());save(root/'protocol.json',dict(frozen_utc=now(),parent=str(parent),parent_rows_sha256=hashlib.sha256((parent/'rows.json').read_bytes()).hexdigest(),role='offlinevalidation2x2 feasibility,not test/closedloop/newkernelmethod',hyperparameters=dict(bandwidth=[.25,1.,4.],ridge=[.01,.1,1.,10.]),objective_targets=['success_gain','native_return_gain/10'],selection='validation predictionMSE;tie larger ridge',no_privileged_features=True,numerical_margin=1e-12,null_model='zero gain predictor eligible;validationMSE differences<=1e-12 tie towardzero',revision='v1/v2preserved;v3globalminimumtiepool+float64frozengeometry,notnewdata/test'));save(root/'run.json',dict(status='running',created_utc=now()));start=time.monotonic();allrows=json.loads((parent/'rows.json').read_text());results={};rng=np.random.default_rng(2026110414)

def load_design(rows):
    groups=[[] for _ in range(6)]
    for row in rows:
        with np.load(parent/row['request_inputs'],allow_pickle=False) as q,np.load(parent/row['arrival_inputs'],allow_pickle=False) as a,np.load(parent/row['source_packet'],allow_pickle=False) as e:
            assert set(q.files)==set(a.files)=={'visual_features','robot','head0','head1','head2','head3'}
            qr=q['robot'].astype('float64');ar=a['robot'].astype('float64');assert qr.shape==ar.shape==(4,)
            values=[q['visual_features'].astype('float32').ravel(),a['visual_features'].astype('float32').ravel(),np.r_[qr,ar,ar-qr],np.concatenate([z[k] for z in (q,a) for k in ('head0','head1','head2','head3')]).astype('float64'),np.array([row['query']/500,row['delay']/100,row['arrival']/500]),e['source'].astype('float64').ravel()]
            assert values[-1].shape==(400,)
            for group,value in zip(groups,values):group.append(value)
    return [np.array(g,dtype='float32') for g in groups]

def squared_distances(x,y):
    return np.maximum((x*x).mean(1)[:,None]+(y*y).mean(1)[None,:]-2*(x@y.T)/x.shape[1],0.)

def target_matrix(rows,names,metric):
    scale=10. if metric=='native_return' else 1.
    return np.array([[(r['outcomes'][n][metric]-r['outcomes']['reject'][metric])/scale for n in names] for r in rows])

def score_choices(rows,choices):return summary([row['outcomes'][choice] for row,choice in zip(rows,choices)])

try:
 for task in ('push-v3','pick-place-v3'):
    rows=[r for r in allrows if r['task']==task];train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];assert len(train)==14 and len(val)==6;assert set(r['seed'] for r in train).isdisjoint(r['seed'] for r in val);names=list(train[0]['outcomes']);assert names[0]=='reject' and len(names)==12
    xt,xv=load_design(train),load_design(val);train_dist=[];val_dist=[];normalization={};arrays={}
    for i,(t,v) in enumerate(zip(xt,xv)):
        mean=t.mean(0);std=np.maximum(t.std(0),.05);t=((t-mean)/std).astype('float64');v=((v-mean)/std).astype('float64');dt=squared_distances(t,t);dv=squared_distances(v,t);np.fill_diagonal(dt,0.);positive=dt[np.triu_indices(len(t),1)];positive=positive[positive>1e-12];scale=float(np.median(positive)) if len(positive) else 1.;train_dist.append(dt/scale);val_dist.append(dv/scale);arrays.update({f'train_{i}':t.copy(),f'mean_{i}':mean,f'std_{i}':std});normalization[str(i)]=dict(distance_scale=scale,dimensions=t.shape[1])
    np.savez_compressed(root/(task+'_geometry.npz'),**arrays);save(root/(task+'_normalization.json'),normalization);task_results={};donors=[]
    for r in val:
        available=[i for i,t in enumerate(train) if t['query']==r['query']];donors.append(int(rng.choice(available if available else np.arange(len(train)))))
    for metric in ('success','native_return'):
        yt,yv=target_matrix(train,names,metric),target_matrix(val,names,metric);models={}
        for feature_kind,block_count in [('cheap_inputs',5),('with_packet',6)]:
            dt=sum(train_dist[:block_count]);dv=sum(val_dist[:block_count]);cells={}
            for choice_kind,allowed in [('cheap_actions',[i for i,n in enumerate(names) if not n.startswith('act_')]),('all_actions',list(range(len(names))))]:
                grid=[];initial=dict(model='zero',bandwidth=None,ridge=None,validation_mse=float(np.mean(yv[:,allowed]**2)),dual=np.zeros_like(yt),prediction=np.zeros_like(yv));candidates=[initial]
                for bandwidth in (.25,1.,4.):
                    kt=np.exp(-dt/(2*bandwidth**2));kv=np.exp(-dv/(2*bandwidth**2))
                    for ridge in (.01,.1,1.,10.):
                        dual=np.linalg.solve(kt+ridge*np.eye(len(train)),yt);pred=kv@dual;pred[:,0]=0.;mse=float(np.mean((pred[:,allowed]-yv[:,allowed])**2));entry=dict(bandwidth=bandwidth,ridge=ridge,validation_mse=mse);grid.append(entry)
                        candidates.append(dict(model='RBFkernelridge',**entry,dual=dual,prediction=pred))
                best=select_global_minimum(candidates);pred=best['prediction'];selected=[]
                for row in pred:
                    index=max(allowed,key=lambda i:(row[i],-i));selected.append(names[index] if row[index]>1e-12 else 'reject')
                negative_control=None
                if feature_kind=='with_packet' and best['model']!='zero':
                    wrong_d=train_dist[5][donors];wrong_kernel=np.exp(-(sum(val_dist[:5])+wrong_d)/(2*best['bandwidth']**2));wrong=wrong_kernel@best['dual'];wrong[:,0]=0.
                    ids=[max(allowed,key=lambda i:(v[i],-i)) for v in wrong];choices=[names[k] if wrong[i,k]>1e-12 else 'reject' for i,k in enumerate(ids)];negative_control=dict(donor_training_seeds=[train[d]['seed'] for d in donors],choices=choices,metrics=score_choices(val,choices),note='Context-mismatchnegativecontrolmaycauseOOD;notanindependentcausalinformationcertificate')
                modelname=task+'_'+metric+'_'+feature_kind+'_'+choice_kind
                np.savez_compressed(root/(modelname+'_model.npz'),dual=best['dual']);save(root/(modelname+'_grid.json'),grid)
                cells[choice_kind]=dict(selected={k:best[k] for k in ('model','bandwidth','ridge','validation_mse','global_minimum_mse')},choices=selected,metrics=score_choices(val,selected),mean_predicted_gain=float(np.mean([pred[i,names.index(n)] for i,n in enumerate(selected)])),validation_predictions=pred.tolist(),allowed_names=[names[i] for i in allowed],packet_mismatch_control=negative_control,prediction_baselines=dict(zero_mse=float(np.mean(yv[:,allowed]**2)),train_mean_mse=float(np.mean((yv[:,allowed]-yt[:,allowed].mean(0))**2))))
            models[feature_kind]=dict(decision_cells=cells)
        fixed_cheap=max([n for n in names if not n.startswith('act_')],key=lambda n:np.mean([r['outcomes'][n][metric] for r in train]));fixed_all=max(names,key=lambda n:np.mean([r['outcomes'][n][metric] for r in train]));task_results[metric]=dict(models=models,reject_validation=score_choices(val,['reject']*len(val)),best_fixed_cheap_from_train=dict(name=fixed_cheap,metrics=score_choices(val,[fixed_cheap]*len(val))),best_fixed_full_from_train=dict(name=fixed_all,metrics=score_choices(val,[fixed_all]*len(val))),prediction_baselines=dict(zero_validation_mse=float(np.mean(yv**2)),train_mean_validation_mse=float(np.mean((yv-yt.mean(0))**2))),validation_seeds=[r['seed'] for r in val],train_seeds=[r['seed'] for r in train],not_a_test=True)
    results[task]=task_results;save(root/'analysis.json',results);print('KERNEL_PILOT',task,json.dumps({metric:{kind:v['decision_cells'] for kind,v in result['models'].items()} for metric,result in task_results.items()}),flush=True)
 save(root/'checks.json',dict(checks='passed',train_validation_episode_disjoint=True,no_future_input_fields=True,standard_kernel_not_novel=True,formal_test=False));save(root/'run.json',dict(status='complete',completed_utc=now(),seconds=time.monotonic()-start,exit_code=0));print('COMPLETE',root,flush=True)
except BaseException as e:save(root/'run.json',dict(status='failed',ended_utc=now(),error=str(e)));raise
