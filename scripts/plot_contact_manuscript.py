import json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'paper/figures';out.mkdir(exist_ok=True)
style=(out/'academic.mplstyle').read_text()
style=re.sub(r': #([0-9A-Fa-f]{6})',r': "#\1"',style)
(out/'academic.mplstyle').write_text(style);plt.style.use(out/'academic.mplstyle')
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
colors=['#7F8F84','#B7A99A','#8A9199','#B88C8C'];catalog=[]
def export(fig,name,claim,sources):
 for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=240)
 plt.close(fig);catalog.append(dict(id=name,surface_class='paper_main',claim=claim,sources=sources,generator='scripts/plot_contact_manuscript.py',exports=[str(out/(name+'.'+e)) for e in ('png','pdf','svg')],qa='Rendered; inspection pending'))
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
for ax,b,title in zip(axes,('toy','fetch'),('a  Toy reach','b  Fetch reach')):
 p=root/'paper/source_data'/f'reach_{b}.json';d=json.loads(p.read_text())
 for i,k in enumerate(('point_full','point_cheap_only','calibrated_full','calibrated_cheap_only')):
  v=d['results'][k]['value'];m=v['mean']*1000;lo,hi=np.array(v['ci95'])*1000
  ax.errorbar(m,3-i,xerr=[[m-lo],[hi-m]],fmt='o',color=colors[i],elinewidth=1.4,markersize=4)
 ax.axvline(0,color='#8A9199',linewidth=.8,linestyle='--');ax.set_yticks(range(4),['Calibrated, cheap','Calibrated, full','Point, cheap','Point, full'] if b=='toy' else ['']*4)
 ax.set_xlabel('Single-intervention gain (×10⁻³)');ax.set_title(title);ax.grid(axis='y',visible=False);ax.set_ylim(-.5,3.5);ax.set_xlim(right=.25)
export(fig,'fig4_reach_selection','Frozen selectors fail to realize positive average gain. Crossed95% intervals conditional on shared training data.',['paper/source_data/reach_toy.json','paper/source_data/reach_fetch.json'])
p=root/'paper/source_data/contact_posthoc.json';d=json.loads(p.read_text())
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
for ax,task,title in zip(axes,('push-v3','drawer-open-v3'),('a  Push','b  Drawer opening')):
 values=[d['tasks'][task]['paired_public_minus_cheap_oracle'][str(delay)] for delay in (0,4,12)]
 m=np.array([v['mean'] for v in values]);ci=np.array([v['ci95'] for v in values])
 ax.errorbar([0,4,12],m,yerr=np.array([m-ci[:,0],ci[:,1]-m]),fmt='o-',color=colors[0],linewidth=1.5)
 ax.axhline(0,color='#8A9199',linewidth=.8,linestyle='--');ax.set_xticks([0,4,12]);ax.set_xlabel('Arrival delay (control steps)');ax.set_ylabel('Public − cheap-union oracle\n(native return / 500)');ax.set_title(title)
export(fig,'fig1_contact_source','Exploratory arrival-specific source ordering uses same cached public content; panels have different y scales and nominal95% episode intervals.',['paper/source_data/contact_posthoc.json'])
delaystamp='20261102T155844Z';p=root/'paper/source_data/contact_delay.json';d=json.loads(p.read_text())
fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
for ax,task,title in zip(axes,('push-v3','drawer-open-v3'),('a  Push','b  Drawer opening')):
 for j,name,label in ((0,'full_fixed','Full fixed'),(1,'cheap_fixed','Cheap fixed')):
  values=[d['tasks'][task]['closed'][str(delay)][name]['success'] for delay in (0,4,12)]
  m=np.array([v['paired_delta'] for v in values])*100;ci=np.array([v['ci95'] for v in values])*100
  ax.errorbar(np.array([0,4,12])+(-.12 if j==0 else .12),m,yerr=np.array([m-ci[:,0],ci[:,1]-m]),fmt='o-' if j==0 else 's--',color=colors[j],linewidth=1.4,label=label,markersize=4)
 ax.set_ylim(-36,36);ax.set_yticks([-30,-15,0,15,30]);ax.axhline(0,color='#8A9199',linewidth=.8);ax.set_xticks([0,4,12]);ax.set_xlabel('Arrival delay (control steps)');ax.set_title(title);ax.set_ylabel('Paired any-success change\n(percentage points)')
 if task=='push-v3':ax.legend(loc='lower left')
 else:ax.text(.03,.05,'20/20 baseline success;\nzero width does not imply equivalence',transform=ax.transAxes,fontsize=7,color='#6B7280')
export(fig,'fig2_contact_closed','Twenty independent test episodes per task, paired fixed-request closed-loop success differences versus reject. Bootstrap95% intervals; no equivalence claim.',['paper/source_data/contact_delay.json'])
(out/'figure_catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
print('Rendered',len(catalog),'figures')

confirmstamp='20261102T162603Z'
p=root/'paper/source_data/contact_confirmation.json';d=json.loads(p.read_text())
fig,ax=plt.subplots(figsize=(7.2,2.5),layout='constrained')
for index,key,color in ((0,'source_gap_d12_minus_d0',colors[0]),(1,'raw12_d12_minus_d0',colors[2])):
 v=d['metrics'][key];m=v['mean'];lo,hi=v['ci'];ax.errorbar(m,index,xerr=[[m-lo],[hi-m]],fmt='o',color=color,elinewidth=1.6,markersize=5)
ax.axvline(0,color='#8A9199',linewidth=.8,linestyle='--');ax.set_yticks([0,1],['Source-relative gap\n(primary, 98.33% CI)','Raw public action\n(secondary, 95% CI)']);ax.set_ylim(-.6,1.6);ax.grid(axis='y',visible=False)
ax.set_xlabel('Paired change from delay 0 to 12 (native return / 500)');ax.set_title('Independent drawer replication (n = 40 episodes)')
export(fig,'fig3_arrival_replication','Prospective40-episode replication distinguishes reduced raw-action harm from a declining public-minus-cheap-union oracle gap. Primary family3 intervals98.333%; secondary raw contrast95%.',[str(p)])
(out/'figure_catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')

print('Rendered 4 manuscript figures from bundled source data')
