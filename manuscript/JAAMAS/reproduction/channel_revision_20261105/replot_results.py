from pathlib import Path
import json,hashlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ez/DACI/code');P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');F=P/'figures'
plt.style.use(R/'outputs/vision/critic_learning_curve_report_20261104_v2/academic.mplstyle');plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','figure.dpi':180})
records=[]
def export(fig,name,source,message):
 for ext in ['pdf','svg','png']:fig.savefig(F/(name+'.'+ext),facecolor='white')
 plt.close(fig);records.append(dict(name=name,surface_class='internal_review',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),message=message,review='pending actual image inspection'))
source=R/'outputs/jaamas/state_campaign_20261105_v1/analysis/analysis.json';data=json.loads(source.read_text());fig,axs=plt.subplots(3,1,figsize=(4.7,4.4),sharex=True)
for ax,key,title in zip(axs,['packet_cheap_minus_local_cheap','packet_full_minus_packet_cheap','packet_full_minus_reject'],['(a) Packet vs local inputs: local actions','(b) Full vs local actions: packet inputs','(c) Full packet variant vs reject']):
 ax.axvline(0,color='#8A9199',lw=.8)
 for j,task in enumerate(['drawer-open-v3','door-open-v3','button-press-v3']):
  x=data[task]['primary_contrasts'][key];m=x['mean'];lo,hi=x['ci'];ax.errorbar(m,2-j,xerr=[[m-lo],[hi-m]],fmt='o',ms=4,color='#637C83',capsize=2,lw=1.1)
 ax.set_yticks([2,1,0],['Drawer','Door','Button']);ax.set_ylim(-.6,2.6);ax.set_xlim(-.78,.08);ax.set_title(title,loc='left',pad=6);ax.grid(axis='x',alpha=.14);ax.spines[['top','right']].set_visible(False)
axs[-1].set_xlabel('Mean native reward difference (same scale in every panel)');fig.subplots_adjust(left=.17,right=.98,bottom=.13,top=.95,hspace=.72);export(fig,'state_contrasts',source,'Common horizontal scale across all three primary comparison types.')
source=R/'outputs/vision/critic_learning_curve_recovery_20261104_v1/analysis/analysis.json';data=json.loads(source.read_text());fig,axs=plt.subplots(2,1,figsize=(4.7,4.5),sharex=True,sharey=True)
for ax,task,title in zip(axs,['push-v3','pick-place-v3'],['(a) Push','(b) PickPlace']):
 means=data[task]['condition_means'];base=means['reject']['native_return']
 for inp,color,mark,lab in [('cheap_inputs','#637C83','o','Local inputs'),('with_packet','#B29B91','D','Packet inputs')]:
  for act,style,aname in [('cheap_actions','--','local actions'),('all_actions','-','full actions')]:
   yy=[means[f'n{n}/{inp}/{act}']['native_return']-base for n in [7,14,28]];ax.plot([7,14,28],yy,color=color,marker=mark,ms=4,ls=style,lw=1.3,label=f'{lab}, {aname}')
 ax.axhline(0,color='#8A9199',lw=.8);ax.axhline(.25,color='#B7A99A',ls=':',lw=.8);ax.set_title(title,loc='left');ax.set_xscale('log',base=2);ax.set_xticks([7,14,28],['7','14','28']);ax.set_ylabel('Mean reward minus reject');ax.grid(axis='y',alpha=.14);ax.spines[['top','right']].set_visible(False)
axs[-1].set_xlabel('Independent critic training episodes');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8,frameon=False,columnspacing=1.,handlelength=2.);fig.subplots_adjust(left=.16,right=.98,bottom=.23,top=.95,hspace=.38);export(fig,'critic_learning_curve',source,'Descriptive actor-seed0 learning curves; readable at final text width, same scale in both tasks.')
(F/'revision_result_figure_records.json').write_text(json.dumps(records,indent=2));print('rendered',len(records))
