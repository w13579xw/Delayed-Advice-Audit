"""Descriptive fixed-action delay curves and finite-sample benefit bounds."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from advice_audit.report import paired_ci


def upper_positive_probability(wins, n, alpha):
    return 1. if wins == n else float(beta.ppf(1-alpha, wins+1, n-wins))


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root,out=Path(a.parent),Path(a.out)
    assert json.loads((root/'run.json').read_text())['status']=='complete'
    assert not out.exists();out.mkdir(parents=True)
    c=json.loads((root/'protocol.json').read_text())['config']
    result={'posthoc':True,'tasks':{},'note':'Exploratory. CP bounds for paired positive discordance probability bound net success improvement from above; not equivalence proof. Family=2 contrasts x3 delays x2 tasks=12.'}
    for task in c['tasks']:
        with np.load(root/(task+'_test.npz'),allow_pickle=False) as d:
            grouped={}
            for ep,q,delay,g in zip(d['episode'],d['request_step'],d['delay'],d['return_gain']):
                grouped.setdefault((int(ep),int(q)),{})[int(delay)]=g[:,3]
            flip=int(sum(min(rows[d][0] for d in (0,4,12)) < -1e-8 and max(rows[d][0] for d in (0,4,12)) > 1e-8 for rows in grouped.values()))
            curves={str(delay):{'public_raw12':float(d['return_gain'][d['delay']==delay,0,3].mean()),'public_oracle':float(d['return_gain'][d['delay']==delay,0].max(axis=1).mean()),'cheap_oracle':float(d['return_gain'][d['delay']==delay,1:].max(axis=(1,2)).mean())} for delay in (0,4,12)}
            hashes_verified=all(len(set(d['plan_id'][(d['episode']==ep)&(d['request_step']==q),0]))==1 for ep,q in grouped)
            ids=sorted(set(d['episode'].tolist()))
            arrival_contrasts={}
            for delay in (4,12):
                values=[float(np.mean([v[delay][0]-v[0][0] for (seed,_),v in grouped.items() if seed==ep])) for ep in ids]
                arrival_contrasts[str(delay)+'-0']=dict(mean=float(np.mean(values)),ci95=paired_ci(values),episode_means=values)
            source_contrasts={}
            for delay in (0,4,12):
                mask=d['delay']==delay
                public=d['return_gain'][mask,0].max(axis=1)
                cheap=d['return_gain'][mask,1:].max(axis=(1,2))
                values=[float(np.mean((public-cheap)[d['episode'][mask]==ep])) for ep in ids]
                source_contrasts[str(delay)]=dict(mean=float(np.mean(values)),ci95=paired_ci(values))
        closed=json.loads((root/(task+'_closed.json')).read_text())
        bounds=[]
        for delay in (0,4,12):
            bymethod={name:{r['seed']:r for r in closed if r['delay']==delay and r['method']==name} for name in ('reject','full_fixed','cheap_fixed')}
            for baseline in ('reject','cheap_fixed'):
                rows=[bymethod['full_fixed'][ep]['success']-bymethod[baseline][ep]['success'] for ep in bymethod['full_fixed']]
                wins,losses=sum(v>0 for v in rows),sum(v<0 for v in rows)
                bounds.append(dict(delay=delay,contrast='full_fixed-'+baseline,n=len(rows),wins=wins,losses=losses,mean=float(np.mean(rows)),
                    positive_discordance_upper95=upper_positive_probability(wins,len(rows),.05),
                    positive_discordance_upper_family95=upper_positive_probability(wins,len(rows),.05/12)))
        result['tasks'][task]=dict(raw_public12_dense_signflip_groups=flip,total_request_groups=len(grouped),independent_test_episodes=c['test_episodes'],fixed_public_hashes_verified=hashes_verified,curves=curves,paired_arrival_contrasts=arrival_contrasts,paired_public_minus_cheap_oracle=source_contrasts,closed_benefit_bounds=bounds)
    result['parent_test_sha256']={task:hashlib.sha256((root/(task+'_test.npz')).read_bytes()).hexdigest() for task in c['tasks']}
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
