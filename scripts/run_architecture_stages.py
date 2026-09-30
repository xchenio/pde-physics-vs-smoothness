"""Ordered development-only fitting, fixed-POD CNN, and full-field FNO comparison."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.architecture_core import train_one
from scripts.run_uncertainty_modal import plan as prior_plan,data_and_root
from scripts.run_field_modal import plan as field_plan
from scripts.run_public_ridge_residual import digest,locked_json,git_commit
from scripts.run_public_pod_ridge import predict
from scripts.run_public_benchmark import metrics
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/public_architecture_stages_v1'


def plan():
    prior=prior_plan()
    p=dict(source=digest([prior['source'],file_hash(Path(__file__)),file_hash(ROOT/'scripts/architecture_core.py')]),
        tasks=['burgers_original','burgers_independent','allen_cahn'],seeds=[0,1,2],budgets=[1500,6000],penalties=[0.,.1,1.],
        rank=16,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=32,eval_every=250,checkpoint_every=50,
        objective='guarded training-energy normalized squared field-equivalent error; mean-target penalty on raw correction',
        stage1='MLP32/one tanh; residual penalties0/.1/1 x1500/6000, direct penalty0 x1500/6000;3 paired seeds',
        selection='stage1 fixed-final mean development L2 selects residual penalty/budget and direct budget separately; ties listed order; commit before stage2',
        stage2='periodic CNN16x3,kernel5,tanh,adaptive24-bin pool and positional linear POD head; use both stage1-selected settings unchanged',
        stage3='FNO32width16modes4layers full field; use both same selected settings unchanged; no own architecture-specific tuning',
        limits='Development only; no fresh confirmation IDs/targets opened.600 fitting samples plus fixed representation400 information shared across models. Three seeds, not5; architecture-dependent parameters/compute. Cosine horizon changes with budget; no pure step-count attribution.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def config(p,architecture,mapping,penalty,steps,seed):
    return dict({k:p[k] for k in ['rank','lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every']},
        architecture=architecture,mapping=mapping,penalty=penalty,steps=steps,seed=seed)


def cases(p):
    return [('residual',penalty,steps) for penalty in p['penalties'] for steps in p['budgets']]+[('direct',0.,steps) for steps in p['budgets']]


def name(mapping,penalty,steps):return f'{mapping}_p{penalty:g}_b{steps}'


def group(results):
    return dict(metrics={s:{m:dict(mean=float(np.mean([r['metrics'][s]['prediction'][m] for r in results])),
        std=float(np.std([r['metrics'][s]['prediction'][m] for r in results],ddof=1))) for m in ('mean_relative_l2','rmse','relative_fro')} for s in ('train','val')},
        parameters_real=results[0]['parameters_real'],mean_optimization_seconds=float(np.mean([r['optimization_seconds'] for r in results])))


def run_stage(p,stage,device):
    architecture={'fit':'mlp','spatial':'cnn','fno':'fno'}[stage]
    for task in p['tasks']:
        root=OUT/task;root.mkdir(parents=True,exist_ok=True);arrays,oldroot=data_and_root(task)
        rows=field_plan()['correction_rows'];arrays=dict(arrays,train_x=arrays['train_x'][rows],train_y=arrays['train_y'][rows])
        anchor_path=oldroot/'linear_final/checkpoints/model.pt';anchor=torch.load(anchor_path,weights_only=False)['model']
        context=dict(plan=digest(p),task=task,anchor_sha256=file_hash(anchor_path),stage=stage)
        if stage=='fit':selected_cases=cases(p)
        else:
            selection=json.loads((root/'fit_selection.json').read_text())
            # Require the stage1 choices to exist in Git before any architecture follow-up.
            from scripts.run_incremental_modal import committed
            committed(root/'fit_selection.json')
            selected_cases=[(m,selection[m]['penalty'],selection[m]['steps']) for m in ('residual','direct')]
            context['fit_selection_sha256']=file_hash(root/'fit_selection.json')
        groups={}
        for mapping,penalty,steps in selected_cases:
            label=name(mapping,penalty,steps)
            runs=[train_one(config(p,architecture,mapping,penalty,steps,seed),arrays,anchor,context,
                root/stage/(label+f'_s{seed}'),device) for seed in p['seeds']]
            groups[label]=dict(mapping=mapping,penalty=penalty,steps=steps,**group(runs))
            atomic_json(root/(stage+'_progress.json'),groups)
        atomic_json(root/(stage+'_summary.json'),dict(context=context,groups=groups,neural_runs=len(groups)*len(p['seeds']),
            linear={s:metrics(predict(anchor,arrays[s+'_x']),arrays[s+'_y']) for s in ('train','val')}))
        if stage=='fit':
            chosen={m:min([g for g in groups.values() if g['mapping']==m],key=lambda g:g['metrics']['val']['mean_relative_l2']['mean']) for m in ('residual','direct')}
            locked_json(root/'fit_selection.json',chosen)
        print('STAGE COMPLETE',stage,task,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['plan','fit','spatial','fno']);a=a.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;p=plan()
    if a.stage!='plan':run_stage(p,a.stage,torch.device('cuda'))


if __name__=='__main__':main()
