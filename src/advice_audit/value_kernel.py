"""Frozen standard-kernel arrival decisions; no simulator or privileged inputs."""
import json
from pathlib import Path
import numpy as np


def select_global_minimum(candidates, tolerance=1e-12):
    """Tolerance is relative to the global optimum, avoiding chained tie drift."""
    if not candidates or tolerance < 0:
        raise ValueError('Need candidates and nonnegative tolerance')
    minimum=min(c['validation_mse'] for c in candidates)
    eligible=[c for c in candidates if c['validation_mse']<=minimum+tolerance]
    result=min(eligible,key=lambda c:(c['model']!='zero',-(c['ridge'] or 0.),c['bandwidth'] or 0.))
    return dict(result,global_minimum_mse=minimum)


class FrozenKernelDecisions:
    """Exactly four fixed development cells, evaluated before any outcome."""
    def __init__(self, root, task, objective='native_return'):
        root=Path(root);self.task=task;self.objective=objective
        self.analysis=json.loads((root/'analysis.json').read_text())[task][objective]
        self.norm=json.loads((root/(task+'_normalization.json')).read_text())
        with np.load(root/(task+'_geometry.npz'),allow_pickle=False) as z:
            self.geometry={k:z[k].copy() for k in z.files}
        self.cells={}
        for features,model in self.analysis['models'].items():
            for actions,cell in model['decision_cells'].items():
                stem=task+'_'+objective+'_'+features+'_'+actions
                with np.load(root/(stem+'_model.npz'),allow_pickle=False) as z:dual=z['dual'].copy()
                self.cells[(features,actions)]=(cell,dual)

    def distances(self, request, arrival, packet, query, delay):
        allowed={'visual_features','robot','head0','head1','head2','head3'}
        if set(request)!=allowed or set(arrival)!=allowed:
            raise ValueError('Only explicit observed-input fields accepted')
        qr=np.asarray(request['robot'],dtype='float64');ar=np.asarray(arrival['robot'],dtype='float64')
        if qr.shape!=(4,) or ar.shape!=(4,):raise ValueError('Only robot4D accepted')
        for value in (request,arrival):
            if np.shape(value['visual_features'])!=(512,16,16):raise ValueError('Need full observed spatial map')
        groups=[np.asarray(request['visual_features'],dtype='float32').ravel(),np.asarray(arrival['visual_features'],dtype='float32').ravel(),np.r_[qr,ar,ar-qr],np.concatenate([z[k] for z in (request,arrival) for k in ('head0','head1','head2','head3')]),np.array([query/500,delay/100,(query+delay)/500]),np.asarray(packet).ravel()]
        if groups[-1].shape!=(400,):raise ValueError('Need arrived100x4packet')
        result=[]
        for i,value in enumerate(groups):
            raw=np.asarray(value,dtype='float32')
            if not np.isfinite(raw).all():raise ValueError('Nonfinite observed input')
            mean=self.geometry[f'mean_{i}'];std=self.geometry[f'std_{i}'];train=self.geometry[f'train_{i}']
            if train.dtype!=np.float64:raise ValueError('Need frozen full-precision geometry')
            v=((raw-mean)/std).astype('float64')
            d=np.maximum(np.mean(v*v)+np.mean(train*train,axis=1)-2*(train@v)/len(v),0.)
            result.append(d/self.norm[str(i)]['distance_scale'])
        return result

    def decide(self, distances):
        result={}
        for (features,actions),(cell,dual) in self.cells.items():
            selected=cell['selected'];names=cell['allowed_names']
            if selected['model']=='zero':
                result[features+'/'+actions]=dict(choice='reject',predicted_gain=0.,null_model=True)
                continue
            count=5 if features=='cheap_inputs' else 6
            kernel=np.exp(-sum(distances[:count])/(2*selected['bandwidth']**2));pred=kernel@dual;pred[0]=0.
            # Cell exports preserve the common12arm column order.
            all_names=self.analysis['models']['cheap_inputs']['decision_cells']['all_actions']['allowed_names']
            ids=[all_names.index(name) for name in names];index=max(ids,key=lambda i:(pred[i],-i));choice=all_names[index] if pred[index]>1e-12 else 'reject'
            result[features+'/'+actions]=dict(choice=choice,predicted_gain=float(pred[index]),null_model=False)
        return result
