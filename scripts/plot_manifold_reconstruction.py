"""Plot audited reconstruction trade-offs, not operator prediction results."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    a=json.loads((ROOT/'results/manifold_reconstruction_v1/audit.json').read_text())
    if a['checked_runs']!=81:raise ValueError('complete replay audit required')
    fig,axs=plt.subplots(3,3,figsize=(11.3,8.2),sharex=True)
    colors={'POD':'#202020','local':'#1d75b8','global':'#d17b13','scrambled':'#a74575'}
    for row,(task,t) in enumerate(a['tasks'].items()):
        for col,key in enumerate(('field','gradient','curvature')):
            ax=axs[row,col];ranks=np.array([16,32,64])
            for arm in ('POD','local','global','scrambled'):
                if arm=='POD':v=np.array([t['pod'][str(r)][key] for r in ranks]);sd=np.zeros(3)
                else:
                    v=np.array([t['groups'][f'{arm}_r{r}']['metrics']['val'][key]['mean'] for r in ranks])
                    sd=np.array([t['groups'][f'{arm}_r{r}']['metrics']['val'][key]['seed_sd'] for r in ranks])
                ax.plot(ranks,v,'o-',label=arm,color=colors[arm],linewidth=1.5,markersize=4)
                ax.fill_between(ranks,np.maximum(v-sd,1e-12),v+sd,color=colors[arm],alpha=.12)
            ax.set_yscale('log');ax.set_xticks(ranks);ax.grid(alpha=.2)
            if row==0:ax.set_title(dict(field='Field error',gradient='First derivative error',curvature='Second derivative error')[key])
            if col==0:ax.set_ylabel(task.replace('_',' ')+'\nMean relative error')
            if row==2:ax.set_xlabel('Latent scalar budget')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.962),frameon=False)
    fig.suptitle('Compression diagnostic: spatial locality helps, POD remains a strong control',y=.994,fontsize=13)
    fig.text(.5,.014,'Reused development200; AE means ± seed SD (3 seeds); POD fitted on training400.\n'
             'Local/scrambled match capacity; global differs. Reconstruction sees true output; no operator/physics confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.064,1,.934));folder=ROOT/'paper/figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(folder/f'manifold_reconstruction_20261008.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
