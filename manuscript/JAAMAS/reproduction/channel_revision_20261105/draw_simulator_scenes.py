"""Actual preassigned engineering scenes, uncurated and excluded from test inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ez/DelayedAdviceAudit/code');D=R/'outputs/jaamas/visual_pipeline_20261105_v1/seed0_test_smoke';P=Path('/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS');F=P/'figures';assert json.loads((D/'run.json').read_text())['status']=='complete'
plt.style.use(R/'outputs/vision/critic_learning_curve_report_20261104_v2/academic.mplstyle');plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','figure.dpi':220})
fig,axs=plt.subplots(2,4,figsize=(4.7,3.5));provenance=[]
for i,(task,seed,label) in enumerate([('push-v3',859000100,'Push'),('pick-place-v3',859010100,'PickPlace')]):
 stem=f'{task}_{seed}';choices=D/(stem+'_choices_before_outcomes.json');dec=json.loads(choices.read_text());assert dec['query']==32 and dec['delay']==32;name=dec['decisions']['n14/with_packet/all_actions'];src=D/(stem+'_scene_frames.npz')
 with np.load(src,allow_pickle=False) as z:
  keys=['request','arrival','branch_reject_after48','branch_'+name+'_after48'];images=[z[k].copy() for k in keys]
 for j,img in enumerate(images):
  # Undo the model input's declared vertical flip for the camera-orientation illustration.
  axs[i,j].imshow(np.flipud(img));axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
  for spine in axs[i,j].spines.values():spine.set_color('#D8D1C7');spine.set_linewidth(.4)
  if i==0:axs[i,j].set_title(['Request\nq = 32','Arrival\na = 64','Reject\na + 48','Full policy\na + 48'][j],fontsize=8,pad=4)
 axs[i,0].set_ylabel(('(a) ' if i==0 else '(b) ')+label,fontsize=9,labelpad=5)
 labels={'reject':'reject (same path)','cheap_ensemble48':'local ensemble\n48 steps','cheap_request48':'request hold\n48 steps','cheap_arrival48':'arrival hold\n48 steps','act_blend48':'ACT/local blend\n48 steps'}
 text=(f'local pose head {name[len("cheap_head")]}\n48 feedback steps' if name.startswith('cheap_head') and name[len('cheap_head')] in '23' else labels.get(name,name.replace('cheap_head','local head ').replace('_48',' / 48').replace('act_','ACT ')))
 axs[i,3].set_xlabel(text,fontsize=8,labelpad=3)
 provenance.append(dict(task=task,seed=seed,query=32,delay=32,selected_option=name,raw_scene_file=str(src),raw_scene_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),choices_sha256=hashlib.sha256(choices.read_bytes()).hexdigest(),frame_keys=keys,display_transform='verticalflipundoneforcameraorientation;nocrop,nopixeladjustment',sample_role='firstpreassignedengineeringsmokecase,notselectedbyoutcome;notformaltest'))
fig.subplots_adjust(left=.055,right=.995,top=.84,bottom=.13,wspace=.025,hspace=.45)
for ext in ['pdf','svg','png']:fig.savefig(F/('simulator_scenes.'+ext),dpi=400,facecolor='white')
plt.close(fig);(F/'simulator_scene_record.json').write_text(json.dumps(dict(surface_class='internal_review',review='pending actual render inspection',script=str(Path(__file__).resolve()),sources=provenance,claim='Illustrates timing/continuation and selectedcommands;notpositiveinterventionexample'),indent=2));print(provenance)
