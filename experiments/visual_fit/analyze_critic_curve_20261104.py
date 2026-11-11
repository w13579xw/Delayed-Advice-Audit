"""Eight fixed prospective learning-curve contrasts; all other curves descriptive."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from server_run import save,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    parent=Path(a.parent).resolve();out=Path(a.out).resolve()
    if out.exists():raise RuntimeError('New output directory required')
    assert json.loads((parent/'run.json').read_text())['status']=='complete'
    out.mkdir(parents=True);rng=np.random.default_rng(2026110427);rows=json.loads((parent/'episodes.json').read_text())
    assert len(rows)==200
    pairs={
      'n28_total_vs_reject':('n28/with_packet/all_actions','reject'),
      'n14_to_n28_data_increment':('n28/with_packet/all_actions','n14/with_packet/all_actions'),
      'n28_packet_input_on_cheap':('n28/with_packet/cheap_actions','n28/cheap_inputs/cheap_actions'),
      'n28_actuation_given_packet':('n28/with_packet/all_actions','n28/with_packet/cheap_actions')}
    results={};vectors={}
    for task in ('push-v3','pick-place-v3'):
        rr=[r for r in rows if r['task']==task];assert len(rr)==100
        indices=rng.integers(0,100,(50000,100));primary={};means={};choices={}
        for name,(left,right) in pairs.items():
            d=np.array([r['conditions'][left]['native_return']-r['conditions'][right]['native_return'] for r in rr],dtype='float64')
            assert np.isfinite(d).all();boot=d[indices].mean(1);ci=np.quantile(boot,[.05/16,1-.05/16]).tolist()
            primary[name]=dict(left=left,right=right,mean=float(d.mean()),bootstrap_ci=ci,confidence=.99375,
                positive=int((d>1e-10).sum()),negative=int((d< -1e-10).sum()),within_tolerance=int((np.abs(d)<=1e-10).sum()),
                minimum=float(d.min()),maximum=float(d.max()),vector_sha256=hashlib.sha256(d.tobytes()).hexdigest(),
                lower_above_zero=ci[0]>0,lower_above_reward_scale_reference=ci[0]>.25)
            vectors[task+'_'+name]=d
        for condition in rr[0]['conditions']:
            means[condition]={metric:float(np.mean([r['conditions'][condition][metric] for r in rr]))
                              for metric in ('native_return','success','terminal_success')}
            hist={}
            for r in rr:hist[r['decisions'][condition]]=hist.get(r['decisions'][condition],0)+1
            choices[condition]=dict(histogram=hist,acceptance_count=100-hist.get('reject',0))
        results[task]=dict(primary_contrasts=primary,condition_means=means,choices=choices,
            joint_positive_data_rescue=primary['n28_total_vs_reject']['lower_above_zero'] and primary['n14_to_n28_data_increment']['lower_above_zero'])
    np.savez_compressed(out/'primary_vectors.npz',**vectors);save(out/'analysis.json',results)
    save(out/'protocol.json',dict(bootstrap_samples=50000,rng=2026110427,primary_family=8,individual_confidence=.99375,
        interval='wholeepisode paired percentilebootstrap; approximate Bonferroni family8',reward_scale_reference=.25,
        no_finite_sample_guarantee=True,no_sample_extension=True,all_other_outcomes_secondary=True))
    save(out/'checks.json',dict(checks='passed',primary_contrasts=8,episodes=200,all_finite=True))
    save(out/'run.json',dict(status='complete',completed_utc=now(),exit_code=0));print('COMPLETE',out)

if __name__=='__main__':main()
