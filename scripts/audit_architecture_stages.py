"""Replay saved final predictions and regenerate a compact development report."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.architecture_core import make_model,output
from scripts.run_public_pod_readout import prepare
from scripts.run_public_benchmark import metrics
from scripts.run_architecture_stages import OUT,group
from scripts.run_uncertainty_modal import data_and_root
from scripts.run_field_modal import plan as field_plan
from src.public_benchmark import atomic_json,file_hash


def main():
    torch.set_num_threads(4)
    audited=[];tables={}
    for task in ('burgers_original','burgers_independent','allen_cahn'):
        arrays,_=data_and_root(task);rows=field_plan()['correction_rows']
        arrays=dict(arrays,train_x=arrays['train_x'][rows],train_y=arrays['train_y'][rows])
        tables[task]={}
        for stage in ('fit','spatial','fno'):
            summary_path=OUT/task/(stage+'_summary.json')
            if not summary_path.exists():continue
            summary=json.loads(summary_path.read_text());tables[task][stage]=summary
            for label,expected in summary['groups'].items():
                runs=[]
                for seed in (0,1,2):
                    root=OUT/task/stage/(label+f'_s{seed}')
                    r=json.loads((root/'result.json').read_text());runs.append(r)
                    cp_path=root/'checkpoints/final.pt'
                    assert file_hash(cp_path)==r['selected_sha256']
                    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
                    assert cp['identity']==r['identity']
                    c=cp['identity']['config'];assert cp['step']==c['steps']
                    anchor=cp['anchor'];prepared,scale=prepare(arrays,anchor)
                    model=make_model(c,arrays['train_x'].shape[1],torch.device('cuda'))
                    model.load_state_dict(cp['model']);model.eval()
                    with torch.no_grad():
                        for s,v in prepared.items():
                            x=torch.tensor(v['x'],device='cuda',dtype=torch.float32)
                            z=torch.cat([output(model,b,c['architecture']) for b in x.split(128)]).cpu().numpy().astype(float)*scale
                            base=v['x']@anchor['weights']+anchor['intercept']
                            if c['architecture']=='fno':pred=z+(base@anchor['q'].T if c['mapping']=='residual' else 0)
                            else:pred=(z+(base if c['mapping']=='residual' else 0))@anchor['q'].T
                            measured=metrics(pred+v['offset'],v['y'])
                            for k,value in measured.items():
                                assert np.isclose(value,r['metrics'][s]['prediction'][k],rtol=1e-6,atol=1e-8),(root,s,k)
                    audited.append(str(root.relative_to(ROOT)))
                actual=group(runs)
                for s in ('train','val'):
                    assert actual['metrics'][s]==expected['metrics'][s]
            if stage=='fit':
                chosen=json.loads((OUT/task/'fit_selection.json').read_text())
                for mapping in ('residual','direct'):
                    best=min([g for g in summary['groups'].values() if g['mapping']==mapping],key=lambda g:g['metrics']['val']['mean_relative_l2']['mean'])
                    assert chosen[mapping]==best
    if '--require-complete' in sys.argv:
        assert len(audited)==108, f'expected108 runs, found{len(audited)}'
    atomic_json(OUT/'audit.json',dict(audited_runs=len(audited),runs=audited,scope='final checkpoint hashes, prediction replay, group aggregation, fit selection; development only'))
    for task,stages in tables.items():
        print('\n'+task)
        for stage,s in stages.items():
            print(stage,'linear',s['linear']['val']['mean_relative_l2'])
            for label,g in s['groups'].items():
                print(label,'train',round(g['metrics']['train']['mean_relative_l2']['mean'],6),'val',round(g['metrics']['val']['mean_relative_l2']['mean'],6),'sd',round(g['metrics']['val']['mean_relative_l2']['std'],6),'params',g['parameters_real'],'sec',round(g['mean_optimization_seconds'],1))
    print('AUDITED',len(audited))


if __name__=='__main__':main()
