from pathlib import Path
import numpy as np,json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ez/DelayedAdviceAudit/code');P=Path('/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS');D=R/'outputs/jaamas/state_temporal_20261105_v1';z=np.load(D/'curves.npz')
style=Path('/home/ez/.codex/skills/deepscientist-figure-polish/assets/deepscientist-academic.mplstyle').read_text()
import re,io
style=re.sub(r': (#[0-9A-Fa-f]{6})',r': "\1"',style)
style_path=Path(__file__).parent/'academic_quoted.mplstyle';style_path.write_text(style)
plt.style.use(style_path)
fig,axs=plt.subplots(2,1,figsize=(4.7,3.65),sharex=True,layout='constrained')
colors=['#6D7D8A','#9B8874','#778678'];styles=['-','--',':']
for ch,col,ls,label in zip(['S','F','E'],colors,styles,['Stale','Fresh','Absent → reject']):
 axs[0].plot(np.arange(1,501),z[ch+'_success'],color=col,ls=ls,label=label)
 axs[1].plot(np.arange(1,501),z[ch+'_reward'],color=col,ls=ls)
axs[0].set(ylabel='Fraction in success',ylim=(-.02,1.02),title='(a) Door: success is transient')
axs[1].set(ylabel='Mean native reward',xlabel='Native step (after action)',ylim=(0,10.4),xlim=(1,500),title='(b) Reward retains trajectory information')
axs[0].legend(fontsize=8,ncol=3,loc='upper right',handlelength=1.5,columnspacing=1)
for ax in axs:ax.grid(axis='x',visible=False);ax.set_xticks([1,100,200,300,400,500])
for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/('door_temporal.'+ext),dpi=220)
(P/'figures/door_temporal_provenance.json').write_text(json.dumps(dict(surface_class='appendix',source=str(D/'curves.npz'),script=str(Path(__file__)),claim='Door ever-success and terminal-success differ because native success is brief; dense reward is a different trajectory endpoint',weighting='60environments equally weighted; average5frozenpacket/fullcritics',exports=['door_temporal.pdf','door_temporal.svg','door_temporal.png'],review='Rendered preview inspected: readable at article width; color plus dash patterns; no collisions; style hex literals quoted for Matplotlib compatibility',no_error_band='descriptive complete-cohort curves; no timewise inference'),indent=2))
