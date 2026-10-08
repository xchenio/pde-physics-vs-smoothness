"""Paper figure from audited development records; seed SD, not confidence intervals."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_fno_projection import OUT


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    if audit['audited_runs']!=36:raise ValueError('complete36-run audit required')
    ranks=[16,32,64];tasks=['burgers_original','burgers_independent','allen_cahn']
    titles=['Burgers (original)','Burgers (independent)','Allen–Cahn (synthetic)']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.9))
    for ax,task,title in zip(axes,tasks,titles):
        summary=json.loads((OUT/task/'summary.json').read_text());curves=[]
        training=[summary['groups'][f'pod{r}']['metrics']['val']['mean_relative_l2'] for r in ranks]
        curves.append(('Training-time projection','#0072B2','o',[v['mean'] for v in training],[v['std'] for v in training]))
        full=[json.loads((OUT/task/f'full_s{s}'/'result.json').read_text()) for s in (0,1,2)]
        values=np.array([[v['metrics']['val']['projections'][str(r)]['prediction']['mean_relative_l2'] for r in ranks] for v in full])
        curves.append(('Full training, then projection','#009E73','s',values.mean(0),values.std(0,ddof=1)))
        pooled=[json.loads((ROOT/'results/public_spectral_pod_v1'/task/'summary.json').read_text())['groups'][f'spectral_pod_r{r}']['metrics']['val']['mean_relative_l2'] for r in ranks]
        curves.append(('Prior pooled coefficient head','#D55E00','^',[v['mean'] for v in pooled],[v['std'] for v in pooled]))
        for label,color,marker,means,sds in curves:
            means=np.array(means);sds=np.array(sds)
            ax.plot(ranks,means,color=color,marker=marker,label=label,lw=1.8)
            ax.fill_between(ranks,np.maximum(means-sds,1e-10),means+sds,color=color,alpha=.12)
        ax.axhline(summary['groups']['full']['metrics']['val']['mean_relative_l2']['mean'],color='#333333',ls=':',label='Unprojected full FNO')
        oracle=[full[0]['metrics']['val']['projections'][str(r)]['oracle']['mean_relative_l2'] for r in ranks]
        ax.plot(ranks,oracle,color='#777777',ls='--',marker='.',label='POD projection oracle')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(ranks,[str(r) for r in ranks])
        ax.set_xlabel('POD rank');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean per-sample relative L2')
    fig.suptitle('Shared pointwise FNO head: output projection ablation',fontsize=13,y=.99)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,-.03),frameon=False)
    fig.subplots_adjust(left=.065,right=.99,bottom=.23,top=.85,wspace=.3)
    folder=ROOT/'paper/figures';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(folder/f'fno_projection_20261008.{suffix}',dpi=220,bbox_inches='tight')
    print('Saved paper/figures/fno_projection_20261008.png and .svg')


if __name__=='__main__':main()
