"""Independent final-checkpoint replay and complete tabular development evidence."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_spectral_pod import OUT,plan,task_data
from scripts.spectral_pod_models import make_model
from scripts.run_public_pod_readout import prepare
from scripts.run_public_benchmark import metrics
from scripts.run_architecture_stages import group
from src.public_benchmark import atomic_json,file_hash


def audit(complete=False):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan();records={};count=0
    for task in p['tasks']:
        summary_path=OUT/task/'summary.json'
        if not summary_path.exists():continue
        arrays,anchors,context=task_data(p,task);summary=json.loads(summary_path.read_text())
        assert summary['context']==context
        for rank in p['ranks']:
            q=anchors[rank]['q']
            np.testing.assert_array_equal(q,anchors[64]['q'][:,:rank])
            np.testing.assert_allclose(q.T@q,np.eye(rank),atol=1e-12,rtol=0)
        for label,g in summary['groups'].items():
            runs=[];errors=[]
            for seed in p['seeds']:
                folder=OUT/task/f'{label}_s{seed}';r=json.loads((folder/'result.json').read_text());runs.append(r)
                cp_path=folder/'checkpoints/final.pt';assert file_hash(cp_path)==r['selected_sha256']
                cp=torch.load(cp_path,map_location='cpu',weights_only=False)
                assert cp['identity']==r['identity'];assert cp['identity']['context']==context
                c=cp['identity']['config'];assert cp['step']==p['steps'][task]
                assert c['seed']==seed and c['rank']==g['rank'] and c['architecture']==g['architecture']
                for k,v in anchors[g['rank']].items():np.testing.assert_array_equal(v,cp['anchor'][k])
                prepared,scale=prepare(arrays,cp['anchor']);assert scale==cp['scale']
                model=make_model(c,256,torch.device('cuda'));model.load_state_dict(cp['model']);model.eval()
                with torch.no_grad():
                    for split,v in prepared.items():
                        x=torch.tensor(v['x'],dtype=torch.float32,device='cuda')
                        coef=torch.cat([model(b) for b in x.split(128)]).cpu().numpy().astype(float)*scale
                        pred=coef@cp['anchor']['q'].T+v['offset'];m=metrics(pred,v['y'])
                        for k,value in m.items():assert np.isclose(value,r['metrics'][split]['prediction'][k],atol=1e-8,rtol=1e-6)
                        closure=m['relative_fro']**2-v['oracle']['relative_fro']**2-np.sum((coef-v['coef'])**2)/np.sum(v['y']**2)
                        assert abs(closure)<1e-10
                        if split=='val':errors.append((np.linalg.norm(pred-v['y'],axis=1)/np.maximum(np.linalg.norm(v['y'],axis=1),1e-12)).tolist())
                count+=1
            actual=group(runs);assert actual['metrics']==g['metrics'];assert actual['parameters_real']==g['parameters_real']
            field_sq=float(np.mean([r['metrics']['val']['prediction']['relative_fro']**2 for r in runs]))
            projection_sq=runs[0]['metrics']['val']['oracle']['relative_fro']**2
            coefficient_sq=float(np.mean([r['metrics']['val']['coefficient_relative_fro']**2 for r in runs]))
            assert abs(field_sq-projection_sq-coefficient_sq)<1e-10
            records[f'{task}/{label}']=dict(**g,development_errors=errors,
                squared_fro_decomposition=dict(field=field_sq,projection=projection_sq,coefficient=coefficient_sq,
                                                coefficient_fraction=coefficient_sq/field_sq))
        for rank in p['ranks']:
            pair=[summary['groups'][f'{a}_r{rank}']['parameters_real'] for a in p['architectures']];assert pair[0]==pair[1]
        for architecture in p['architectures']:
            best=min([g for g in summary['groups'].values() if g['architecture']==architecture],key=lambda g:g['metrics']['val']['mean_relative_l2']['mean'])
            assert summary['selected_ranks'][architecture]==best['rank']
    if complete:assert count==54,f'expected54 got{count}'
    atomic_json(OUT/'audit.json',dict(audited_runs=count,records=records,scope='checkpoint hashes, identity, training-only basis, prediction replay, projection closure, parameter counts, groups, selections; development only'))
    print('AUDITED',count)
    for label,r in records.items():print(label,'train',round(r['metrics']['train']['mean_relative_l2']['mean'],6),'dev',round(r['metrics']['val']['mean_relative_l2']['mean'],6),'sd',round(r['metrics']['val']['mean_relative_l2']['std'],6),'oracle',round(r['oracle']['mean_relative_l2'],6),'params',r['parameters_real'],'seconds',round(r['mean_optimization_seconds'],1))


def report():
    lines=['# Spectral/POD development results','',
           '2026-10-08. Three seeds; mean per-sample relative L2, lower is better. Development only. Fixed final checkpoints; no new confirmation.','',
           '| Task | Encoder / rank | Train | Dev mean ± seed SD | POD oracle | Nominal real parameters | Optimization seconds |',
           '|---|---|---:|---:|---:|---:|---:|']
    for task in ('burgers_original','burgers_independent','allen_cahn'):
        path=OUT/task/'summary.json'
        if not path.exists():continue
        for label,g in json.loads(path.read_text())['groups'].items():
            d=g['metrics']['val']['mean_relative_l2']
            lines.append(f"| {task} | {label} | {g['metrics']['train']['mean_relative_l2']['mean']:.6f} | {d['mean']:.6f} ± {d['std']:.6f} | {g['oracle']['mean_relative_l2']:.6f} | {g['parameters_real']} | {g['mean_optimization_seconds']:.1f} |")
    lines+=['','## Historical controls (same development rows)','',
            'Previous three-seed means; full FNO has different output space and parameters; earlier small CNN/MLP are not parameter-matched to the new encoders.','',
            '| Task | Best prior MLP residual | Prior direct CNN/POD16 | Prior direct full FNO |','|---|---:|---:|---:|']
    old=ROOT/'results/public_architecture_stages_v1'
    for task in ('burgers_original','burgers_independent','allen_cahn'):
        sel=json.loads((old/task/'fit_selection.json').read_text());values=[sel['residual']['metrics']['val']['mean_relative_l2']['mean']]
        for stage in ('spatial','fno'):
            s=json.loads((old/task/f'{stage}_summary.json').read_text())
            values.append(next(g for g in s['groups'].values() if g['mapping']=='direct')['metrics']['val']['mean_relative_l2']['mean'])
        lines.append(f'| {task} | '+ ' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','Times exclude evaluation/checkpoints and vary with load. Equal nominal parameter counts do not imply equal compute, initialization, effective degrees of freedom or receptive field. Ranks change readout parameter count. All development outcomes reported; no claim of architecture-specific optimum or method novelty.','']
    return '\n'.join(lines)


if __name__=='__main__':
    if '--report' in sys.argv:print(report())
    else:audit('--require-complete' in sys.argv)
