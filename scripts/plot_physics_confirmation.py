"""Render accuracy/network-latency comparison from sealed fresh confirmation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    c=json.loads((ROOT/'results/physics_token_v1/confirmation.json').read_text())
    labels=['Gate','Gate + physics','Attention','Attention + physics','FNO16','FNO32'];arms=['gate_p0','gate_p0.01','attention_p0','attention_p0.01']
    means=[c['summary'][a]['256']['mean'] for a in arms]+[c['small_fno_summary'][w]['mean'] for w in ('16','32')]
    sd=[c['summary'][a]['256']['seed_sd'] for a in arms]+[c['small_fno_summary'][w]['seed_sd'] for w in ('16','32')]
    latency=[np.mean([r['profiles']['256']['mean_ms'] for r in c['records'][a]]) for a in arms]+[np.mean([r['profile']['mean_ms'] for r in c['small_fno_records'][w]]) for w in ('16','32')]
    colors=['#6c8599','#397ea6','#d69c55','#be772b','#457f4a','#205d2a'];fig,axs=plt.subplots(1,2,figsize=(10.7,4.1))
    y=np.arange(6);axs[0].barh(y,means,xerr=sd,color=colors,capsize=3);axs[0].set_yticks(y,labels);axs[0].invert_yaxis()
    axs[0].set_xlabel('Mean trajectory relative L2 (lower is better)');axs[0].set_title('Fresh500: same100 training trajectories')
    axs[1].barh(y,latency,color=colors);axs[1].set_yticks(y,labels);axs[1].invert_yaxis();axs[1].set_xlabel('Synchronized batch64 network latency (ms)');axs[1].set_title('Five256-point fields per input')
    for ax in axs:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.025,'Error bars: SD of3 fitted seeds, not confidence intervals. Same labels/steps; parameters/compute differ.\n'
             'One Burgers100-shot pilot;500 confirmation trajectories now consumed; no temporal or full-sensor extrapolation.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.12,1,1));folder=ROOT/'paper/figures'
    for ext in ('png','svg'):fig.savefig(folder/f'physics_token_confirmation_20261008.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
