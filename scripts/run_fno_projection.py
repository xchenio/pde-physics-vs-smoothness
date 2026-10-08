"""Shared FNO backbone/head: full-field and training-time POD projection controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_spectral_pod import plan as prior_plan,task_data
from scripts.fno_projection_core import train_one
from scripts.run_architecture_stages import group
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import file_hash,atomic_json
OUT=ROOT/'results/public_fno_projection_v1'


def plan():
    old=prior_plan()
    p=dict(parent=digest(old),source=digest([old['source'],file_hash(Path(__file__)),file_hash(ROOT/'scripts/fno_projection_core.py')]),
           tasks=old['tasks'],seeds=[0,1,2],output_ranks=[0,16,32,64],evaluation_ranks=[16,32,64],
           steps=old['steps'],lr=old['lr'],lr_min=old['lr_min'],weight_decay=old['weight_decay'],
           batch_size=32,eval_every=250,checkpoint_every=50,
           architecture='unchanged FNO32width16modes4layers, shared pointwise32->32->1 head;136481 real parameters in every arm',
           objective='full normalized field squared error or normalized POD coefficient squared error; projected field loss differs by fixed target projection constant; same divisor256',
           pairing='seed-identical full state initialization, optimizer, minibatch sequence, cosine schedule; record initial and rolling batch hashes',
           data='reuse existing nested basis64 and normalization; representation400 only,correction600 fit,development200',
           readout='raw field z from common head; arm0 uses z+input mean; armr uses (z Qr) Qr.T+input mean in training/output; raw projected-arm field is diagnostic only and out-of-subspace output is unconstrained by data loss',
           selection='all36 fits; fixed final step, no rank/arm selection for confirmation; compare projected-training armr vs post-hoc rankr projection of paired full-training arm, then previous pooled coefficient head at same rank',
           limits='Repeated development use; three seeds; changing projection changes gradient/objective and output constraint together. Pooled head is a previous contextual control with different parameters. No new confirmation data or novelty claim.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def config(p,rank,seed,task):
    return dict({k:p[k] for k in ('lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every','evaluation_ranks')},
                output_rank=rank,seed=seed,steps=p['steps'][task])


def run(p,device):
    old=prior_plan()
    for task in p['tasks']:
        arrays,anchors,parent_context=task_data(old,task);anchor=anchors[64]
        context=dict(plan=digest(p),parent=parent_context,task=task);groups={};pairing={}
        for rank in p['output_ranks']:
            label='full' if rank==0 else f'pod{rank}'
            runs=[train_one(config(p,rank,s,task),arrays,anchor,context,OUT/task/f'{label}_s{s}',device) for s in p['seeds']]
            groups[label]=dict(output_rank=rank,**group(runs))
            for s,r in zip(p['seeds'],runs):
                pair=(r['initial_sha256'],r['batch_sha256'],r['parameters_real'])
                if s in pairing and pairing[s]!=pair:raise ValueError('paired initialization/batches/parameters differ')
                pairing[s]=pair
            atomic_json(OUT/task/'progress.json',dict(completed_runs=len(groups)*3,total_runs=12,groups=groups))
        atomic_json(OUT/task/'summary.json',dict(context=context,groups=groups,completed_runs=12,pairing=pairing))
        print('TASK COMPLETE',task,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['plan','run']);args=a.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;p=plan()
    if args.stage=='run':run(p,torch.device('cuda'))


if __name__=='__main__':main()
