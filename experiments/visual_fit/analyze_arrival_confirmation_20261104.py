"""Fixed six paired estimands and predeclared positive-event bounds."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from server_run import save,now
from strong_local_baseline import summary
parent=ROOT/'outputs/vision/arrival_confirmation_20261104_v1';assert json.loads((parent/'run.json').read_text())['status']=='complete';assert json.loads((parent/'checks.json').read_text())['checks']=='passed';root=ROOT/'outputs/vision/arrival_confirmation_analysis_20261104_v1'
if root.exists():raise RuntimeError('New analysis directory required')
root.mkdir();(root/'script_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=json.loads((parent/'episodes.json').read_text());protocol=json.loads((parent/'protocol.json').read_text());assert len(rows)==500 and protocol['episodes_per_task']==250;alpha=.05/12;delta=.25;rng=np.random.default_rng(2026110416);results={}
contrasts=[('packet_input_on_cheap','with_packet/cheap_actions','cheap_inputs/cheap_actions'),('actuation_given_packet','with_packet/all_actions','with_packet/cheap_actions'),('total_assistance','with_packet/all_actions','reject')]
for task in ('push-v3','pick-place-v3'):
 rr=[r for r in rows if r['task']==task];assert len(rr)==250;names=list(rr[0]['conditions']);per={name:summary([r['conditions'][name] for r in rr]) for name in names};indices=rng.integers(0,250,size=(50000,250));comp={}
 for key,left,right in contrasts:
  d=np.array([r['conditions'][left]['native_return']-r['conditions'][right]['native_return'] for r in rr]);bs=d[indices].mean(1);ci=np.quantile(bs,[alpha/2,1-alpha/2]).tolist();positive=int((d>1e-10).sum());negative=int((d < -1e-10).sum());p_upper=float(1-alpha**(1/250)) if positive==0 else (1. if positive==250 else float(beta.ppf(1-alpha,positive+1,250-positive)));event_bound=1e-10+10*p_upper
  comp[key]=dict(left=left,right=right,n_independent_paired_episodes=250,mean=float(d.mean()),bootstrap_ci=ci,bootstrap_confidence=1-alpha,bootstrap_note='PairedpercentileCI,approximatecoverage;notafinite-samplesafetyguarantee',positive_cases=positive,negative_cases=negative,zero_within_1e_minus10_cases=250-positive-negative,positive_event_probability_upper=p_upper,event_bound_mean_positive_gain=event_bound,event_confidence=1-alpha,event_bound_assumptions='iidcompleteepisodeandclockdraws,frozenpolicies,reward0..10;E[d]<=1e-10+10Pr(d>1e-10)',minimum_meaningful_gain=delta,bootstrap_degenerate=bool(np.all(d==d[0])),bootstrap_excludes_gain_atleast_delta=bool(ci[1]<delta and not np.all(d==d[0])),finite_event_bound_excludes_gain_atleast_delta=event_bound<delta,approx_positive_support=bool(ci[0]>0 and not np.all(d==d[0])),approx_support_for_meaningful_gain=bool(ci[0]>delta and not np.all(d==d[0])),raw_paired_differences=d.tolist())
 diagnostic={}
 for flag in (False,True):
  subgroup=[r for r in rr if r['offline_diagnostic_prefix_any_success']==flag];diagnostic[str(flag)]=dict(n=len(subgroup),means={name:summary([r['conditions'][name] for r in subgroup]) for name in names} if subgroup else {},descriptive_only=True)
 choices={name:{option:sum(r['decisions'][name]==option for r in rr) for option in sorted({r['decisions'][name] for r in rr})} for name in names}
 model_data=json.loads((Path(protocol['models'])/'analysis.json').read_text())[task]['native_return'];roles={features+'/'+actions:dict(model=cell['selected']['model'],critic_reads_packet_features=features=='with_packet' and cell['selected']['model']!='zero',allows_source_actions=actions=='all_actions',null_can_skip_query=cell['selected']['model']=='zero') for features,m in model_data['models'].items() for actions,cell in m['decision_cells'].items()}
 results[task]=dict(policy_cell_roles=roles,packet_contrast_scope='Frozenendtoendvariants,includingofflinevalidationmodelselection;nullmodelneedsnopacketandcannotdemonstrateruntimeinformationvalue',condition_means=per,primary_contrasts=comp,prefix_success_diagnostic=diagnostic,choice_counts=choices,n_unique_episode_seeds=250,executed_unique_option_tails=sum(len(r['unique_options']) for r in rr),not_training_seed_replication=True)
 save(root/'analysis.json',results)
save(root/'protocol.json',dict(created_utc=now(),source=str(parent),source_episodes_sha256=hashlib.sha256((parent/'episodes.json').read_bytes()).hexdigest(),primary='mean native reward',families='6bootstrapCIs+6positive-eventbounds,alpha.05/12each',test_samples_fixed=True,min_meaningful_gain=.25,secondary='nativeany/cutoffdiagnostic;nonepromoted',scope='conditionalalreadyrequestedarrival,singleintervention,frozenmodels;notnetcost/generalabsence'))
save(root/'checks.json',dict(checks='passed',all500episodes_included=True,whole_episode_paired_bootstrap=True,no_hindsight_union_asbaseline=True));save(root/'run.json',dict(status='complete',completed_utc=now(),exit_code=0));print('COMPLETE',root,flush=True)
