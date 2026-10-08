"""Replay paired outputs and compare training projection with post-hoc projection."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_fno_projection import OUT,plan,config
from scripts.run_spectral_pod import plan as prior_plan,task_data
from scripts.fno_projection_core import make_model,output,state_digest
from scripts.run_architecture_stages import group
from scripts.run_public_pod_readout import prepare
from scripts.run_public_benchmark import metrics
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import atomic_json,file_hash


def batch_hash(seed,steps,n,batch_size):
    rng=np.random.default_rng(seed+3000);h=hashlib.sha256(b'').hexdigest()
    for _ in range(steps):
        idx=rng.choice(n,min(batch_size,n),replace=False)
        h=hashlib.sha256(bytes.fromhex(h)+idx.astype('<i8').tobytes()).hexdigest()
    return h


def audit(complete=False):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;p=plan();old=prior_plan();records={};comparisons={};count=0;batch_cache={}
    for task in p['tasks']:
        path=OUT/task/'summary.json'
        if not path.exists():continue
        arrays,anchors,parent=task_data(old,task);anchor=anchors[64];prepared,scale=prepare(arrays,anchor);q=anchor['q']
        summary=json.loads(path.read_text());context=dict(plan=digest(p),parent=parent,task=task)
        assert summary['context']==context;paired={};results={}
        for label,g in summary['groups'].items():
            runs=[];predictions=[];rank=g['output_rank']
            for seed in p['seeds']:
                folder=OUT/task/f'{label}_s{seed}';r=json.loads((folder/'result.json').read_text());runs.append(r)
                cp_path=folder/'checkpoints/final.pt';assert file_hash(cp_path)==r['selected_sha256']
                cp=torch.load(cp_path,map_location='cpu',weights_only=False);c=config(p,rank,seed,task)
                assert cp['identity']==r['identity'] and cp['identity']['config']==c and cp['identity']['context']==context
                assert cp['step']==c['steps'] and cp['scale']==scale
                for k,v in anchor.items():np.testing.assert_array_equal(v,cp['anchor'][k])
                model=make_model(c,torch.device('cuda'));assert state_digest(model)==r['initial_sha256']==cp['initial_sha256']
                key=(seed,c['steps'],len(arrays['train_y']),c['batch_size'])
                if key not in batch_cache:batch_cache[key]=batch_hash(*key)
                assert batch_cache[key]==r['batch_sha256']==cp['batch_sha256']
                assert r['parameters_real']==sum(v.numel()*(2 if v.is_complex() else 1) for v in model.parameters())==136481
                pair=(r['initial_sha256'],r['batch_sha256'],r['parameters_real'])
                if seed in paired:assert paired[seed]==pair
                paired[seed]=pair;model.load_state_dict(cp['model']);model.eval();fields={}
                with torch.no_grad():
                    for split,v in prepared.items():
                        x=torch.tensor(v['x'],dtype=torch.float32,device='cuda')
                        z=torch.cat([output(model,b) for b in x.split(128)]).cpu().numpy().astype(float)*scale
                        fields[split]=z
                        measured=metrics(z+v['offset'],v['y'])
                        for k,val in measured.items():assert np.isclose(val,r['metrics'][split]['raw_output'][k],rtol=1e-6,atol=1e-8)
                        for rr in p['evaluation_ranks']:
                            coef=z@q[:,:rr];pred=coef@q[:,:rr].T+v['offset'];m=metrics(pred,v['y'])
                            oracle=v['coef'][:,:rr]@q[:,:rr].T+v['offset'];o=metrics(oracle,v['y'])
                            coefficient_sq=np.sum((coef-v['coef'][:,:rr])**2)/np.sum(v['y']**2)
                            assert abs(m['relative_fro']**2-o['relative_fro']**2-coefficient_sq)<1e-10
                            for k,val in m.items():assert np.isclose(val,r['metrics'][split]['projections'][str(rr)]['prediction'][k],rtol=1e-6,atol=1e-8)
                        expected=r['metrics'][split]['raw_output'] if rank==0 else r['metrics'][split]['projections'][str(rank)]['prediction']
                        assert r['metrics'][split]['prediction']==expected
                predictions.append(fields['val']);count+=1
                if rank==0:
                    oldpath=ROOT/'results/public_architecture_stages_v1'/task/'fno'/f"direct_p0_b{c['steps']}_s{seed}"
                    previous=json.loads((oldpath/'result.json').read_text());oldcp=torch.load(oldpath/'checkpoints/final.pt',map_location='cpu',weights_only=False)
                    for split in ('train','val'):
                        for k,val in previous['metrics'][split]['prediction'].items():assert np.isclose(val,r['metrics'][split]['prediction'][k],rtol=1e-6,atol=1e-8)
                    max_delta=max(float((cp['model'][k]-oldcp['model'][k]).abs().max()) for k in cp['model'])
                    assert max_delta<=1e-6,'historical full parameters failed reproduction tolerance'
                    records[f'{task}/{label}_s{seed}']=dict(historical_full_max_parameter_difference=max_delta,bitwise_equal=all(torch.equal(cp['model'][k],oldcp['model'][k]) for k in cp['model']))
            assert group(runs)['metrics']==g['metrics'];results[label]=dict(runs=runs,predictions=predictions)
        for seed in p['seeds']:assert tuple(summary['pairing'][str(seed)])==paired[seed]
        comparisons[task]={}
        for rank in p['evaluation_ranks']:
            projected=results[f'pod{rank}']['runs'];full=results['full']['runs']
            now=np.array([r['metrics']['val']['prediction']['mean_relative_l2'] for r in projected])
            post=np.array([r['metrics']['val']['projections'][str(rank)]['prediction']['mean_relative_l2'] for r in full])
            pooled=[]
            for seed in p['seeds']:
                r=json.loads((ROOT/'results/public_spectral_pod_v1'/task/f'spectral_pod_r{rank}_s{seed}'/'result.json').read_text())
                pooled.append(r['metrics']['val']['prediction']['mean_relative_l2'])
            errors={}
            for label in ('full',f'pod{rank}'):
                values=[]
                for z in results[label]['predictions']:
                    v=prepared['val'];pred=z@q[:,:rank]@q[:,:rank].T+v['offset']
                    values.append((np.linalg.norm(pred-v['y'],axis=1)/np.maximum(np.linalg.norm(v['y'],axis=1),1e-12)).tolist())
                errors[label]=values
            comparisons[task][str(rank)]=dict(training_projection_mean=float(now.mean()),training_projection_std=float(now.std(ddof=1)),
                full_training_posthoc_projection_mean=float(post.mean()),paired_seed_differences=(now-post).tolist(),
                prior_pooled_head_mean=float(np.mean(pooled)),paired_vs_pooled=(now-np.array(pooled)).tolist(),
                development_errors=errors,scope='exploratory paired development differences; no confidence/significance claim')
    if complete:assert count==36,f'expected36 got{count}'
    atomic_json(OUT/'audit.json',dict(audited_runs=count,historical_reproduction=records,comparisons=comparisons,
                                  scope='checkpoint/identity/predictions/closure/initialization/minibatches/parameter count/aggregates/historical full reproduction'))
    print('AUDITED',count)
    for task,groups in comparisons.items():
        print(task)
        for rank,g in groups.items():print('rank',rank,'train-project',g['training_projection_mean'],'posthoc',g['full_training_posthoc_projection_mean'],'pooled',g['prior_pooled_head_mean'],'paired',g['paired_seed_differences'])
    print('Historical full bitwise reproduction',all(r['bitwise_equal'] for r in records.values()))


def report():
    lines=['# FNO shared-head projection development results','',
           '2026-10-08. Three seeds; fixed final checkpoints; mean per-sample relative L2. Same136481-parameter FNO and head in every current arm. Reused development data, no new confirmation.','',
           '| Task | Arm | Train | Dev mean ± seed SD | Optimization seconds |',
           '|---|---|---:|---:|---:|']
    for task in ('burgers_original','burgers_independent','allen_cahn'):
        path=OUT/task/'summary.json'
        if not path.exists():continue
        for label,g in json.loads(path.read_text())['groups'].items():
            d=g['metrics']['val']['mean_relative_l2']
            lines.append(f"| {task} | {label} | {g['metrics']['train']['mean_relative_l2']['mean']:.6f} | {d['mean']:.6f} ± {d['std']:.6f} | {g['mean_optimization_seconds']:.1f} |")
    path=OUT/'audit.json'
    if path.exists():
        lines+=['','## Same POD output space','',
                '| Task | Rank | Training projection | Full training then projection | Prior pooled coefficient head |',
                '|---|---:|---:|---:|---:|']
        for task,groups in json.loads(path.read_text())['comparisons'].items():
            for rank,g in groups.items():lines.append(f"| {task} | {rank} | {g['training_projection_mean']:.6f} | {g['full_training_posthoc_projection_mean']:.6f} | {g['prior_pooled_head_mean']:.6f} |")
    lines+=['','The projected-training raw field is outside the deployed arm and its unsupervised complement is unconstrained. Gradients/objective change with projection. Previous pooled-head models differ in parameter count. Times exclude evaluation/checkpoints; they are load-dependent. All arms reported, no arm selected for confirmation, no novelty/independent validation claim.','']
    return '\n'.join(lines)


if __name__=='__main__':
    if '--report' in sys.argv:print(report())
    else:audit('--require-complete' in sys.argv)
