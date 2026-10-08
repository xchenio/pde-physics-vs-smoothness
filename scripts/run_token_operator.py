"""Stage3 exact parameter/initialization/batch attention ablation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.token_operator import train_one
from scripts.run_spatial_deeponet import plan as stage2_plan
from scripts.run_spectral_pod import plan as parent_plan,task_data
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/token_operator_v1'


def plan():
    old=stage2_plan()
    p=dict(old,parent=digest(old),source=digest([old['source'],file_hash(Path(__file__)),file_hash(ROOT/'scripts/token_operator.py')]),
           arms=['gate','attention'],architecture='same CNN64/trunk64;16x4 bottleneck tokens expanded to32 channels during mixing and reduced back to16x4; periodic positions; pre-LN32,QKV32->96,projection32,FFN32->64->32',
           control='four-head attention vs pointwise v*sigmoid(q+k); identical active trainable tensors/count/init, all layers used; same seeds/minibatches/steps. Gate does not mix tokens inside block; shared final branch head mixes globally.',
           selection='all18 fits reported; fixed final step; rank64 and all hyperparameters transferred unchanged; no selection or fresh confirmation',
           limits='Fixed-time development-only. Difference isolates registered attention-vs-gate interaction, not all possible nonattention mixers. Equal parameters/steps not equal FLOPs/time. Temporary hidden expansion not counted as compressed bottleneck; all token memory reported separately in future profiles.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def run(p,device):
    old=parent_plan()
    for task in p['tasks']:
        arrays,anchors,parent=task_data(old,task);context=dict(parent=parent,plan=digest(p),task=task);groups={};pair={}
        for kind in p['arms']:
            runs=[]
            for seed in p['seeds']:
                c={k:p[k] for k in ('rank','lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every')}
                c.update(kind=kind,seed=seed,task=task,steps=p['steps'][task],length=2. if task=='allen_cahn' else 1.,final_time=.5 if task=='allen_cahn' else 1.)
                r=train_one(c,arrays,anchors[64],context,OUT/task/f'{kind}_s{seed}',device);runs.append(r)
                value=(r['initial'],r['batch_hash'],r['parameters'])
                if seed in pair and pair[seed]!=value:raise ValueError('attention pairing failed')
                pair[seed]=value
            groups[kind]=dict(parameters=runs[0]['parameters'],metrics={s:{k:dict(mean=float(np.mean([r['metrics'][s]['prediction'][k] for r in runs])),seed_sd=float(np.std([r['metrics'][s]['prediction'][k] for r in runs],ddof=1))) for k in runs[0]['metrics'][s]['prediction']} for s in ('train','val')})
            atomic_json(OUT/task/'progress.json',dict(completed_runs=len(groups)*3,total_runs=6,groups=groups))
        atomic_json(OUT/task/'summary.json',dict(context=context,groups=groups,completed_runs=6,pairing={str(s):v for s,v in pair.items()}));print('TASK COMPLETE',task,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p,torch.device('cuda'))
