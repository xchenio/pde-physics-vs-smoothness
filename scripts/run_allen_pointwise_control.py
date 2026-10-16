"""Same-architecture/five-node/step-budget pointwise-time PDE comparator.

Reuse only each condition's audited reshuffled-split full-state warm. Replaying
all18 old controls is an engineering gate, not additional efficacy evidence.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.allen_interval_replication_data import load_cached_split, observations
from scripts.allen_pointwise_control_core import train_one
from scripts.run_allen_interval_replication import summarize
from scripts.run_public_ridge_residual import digest, locked_json
from src.public_benchmark import atomic_json, file_hash

OUT = ROOT / 'results/allen_pointwise_control_v1'
PARENT = ROOT / 'results/allen_interval_replication_v1'


def entry(folder):
    result, final = folder / 'result.json', folder / 'checkpoints/final.pt'
    r = json.loads(result.read_text())
    if file_hash(final) != r['final_sha256']:
        raise ValueError('parent final receipt changed')
    return dict(result_path=result.relative_to(ROOT).as_posix(), result_sha256=file_hash(result),
                final_path=final.relative_to(ROOT).as_posix(), final_sha256=file_hash(final))


def plan(write=True):
    old = json.loads((PARENT / 'plan.json').read_text())
    previous = json.loads((PARENT / 'audit.json').read_text())
    if not previous['complete'] or previous['checked_runs'] != 24 or previous['checked_warm_runs'] != 6:
        raise ValueError('audited24-record parent required')
    # Fail before formal computation if any parent's sealed training dependency
    # or cached observation bytes no longer reproduce the completed parent.
    for name, expected in {**old['source'], **old['parents'], **old['data_chunks']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('parent sealed dependency changed: ' + name)
    warm_lineage, controls = {}, {}
    for condition in old['conditions']:
        for seed in old['seeds']:
            folder = PARENT / condition / f'warm_s{seed}'
            record = entry(folder)
            latest_path = folder / 'checkpoints/latest.pt'
            latest = torch.load(latest_path, map_location='cpu', weights_only=False)
            final = torch.load(ROOT / record['final_path'], map_location='cpu', weights_only=False)
            audited = previous['evidence'][f'{condition}/warm_s{seed}']
            if file_hash(latest_path) != audited['latest_sha256'] or record['final_sha256'] != audited['final_sha256']:
                raise ValueError('warm differs from audited parent')
            if latest['step'] != 1500 or final['step'] != 1500 or latest['identity'] != final['identity']:
                raise ValueError('warm identity/stage changed')
            if not all(torch.equal(v, final['model'][k]) for k, v in latest['model'].items()):
                raise ValueError('warm latest/final tensor mismatch')
            if any(k not in latest for k in ['optimizer', 'rng', 'collocation_rng']) or not latest['optimizer']['state']:
                raise ValueError('warm lacks full continuation state')
            if any(int(v['step']) != 1500 for v in latest['optimizer']['state'].values()):
                raise ValueError('warm optimizer stage changed')
            record.update(latest_path=latest_path.relative_to(ROOT).as_posix(), latest_sha256=file_hash(latest_path))
            warm_lineage[f'{condition}/s{seed}'] = record
            for kind in old['kinds']:
                name = f'{condition}/{kind}_s{seed}'
                record = entry(PARENT / name)
                if record['final_sha256'] != previous['evidence'][name]['final_sha256']:
                    raise ValueError('historical control differs from parent audit')
                controls[name] = record
    names = ['scripts/run_allen_pointwise_control.py', 'scripts/allen_pointwise_control_core.py',
             'scripts/audit_allen_pointwise_control.py', 'tests/test_allen_pointwise_control.py',
             'docs/ALLEN_POINTWISE_CONTROL_V1.md', 'scripts/allen_interval_replication_data.py',
             'scripts/allen_interval_replication_core.py', 'scripts/run_allen_interval_replication.py',
             'scripts/audit_allen_interval_replication.py', 'scripts/allen_interval_pilot_core.py',
             'scripts/periodic_curvature_core.py', 'scripts/physics_stress_core.py',
             'scripts/time_fno_baseline.py', 'scripts/fno_regularization_core.py', 'scripts/objective_modal_core.py',
             'scripts/physics_token_core.py', 'scripts/diagnose_physics_stress_v2.py',
             'scripts/field_derivative_diagnostics.py', 'scripts/run_public_pod_readout.py',
             'scripts/run_public_ridge_residual.py', 'src/physics_equations_v1.py', 'src/periodic_position_fno.py',
             'src/time_fno.py', 'src/fno.py', 'src/resume.py', 'src/utils.py', 'src/public_benchmark.py']
    p = {k: old[k] for k in ['split_seed', 'parent_ids_pool', 'train_ids', 'val_ids', 'conditions', 'seeds',
                            'warm_steps', 'continuation_steps', 'strength', 'width', 'modes', 'layers',
                            'weight_decay', 'batch_size', 'eval_every', 'checkpoint_every', 'resolved_max_mode',
                            'task_spec', 'interval_half_widths', 'interval_nodes', 'evaluation_columns', 'common_columns', 'data_chunks']}
    parents = [PARENT / 'plan.json', PARENT / 'audit.json', PARENT / 'summary.json', ROOT / 'results/physics_stress_v1/plan.json']
    p.update(version='allen_pointwise_control_v1', exploratory=True,
             replication_plan_sha256=file_hash(PARENT / 'plan.json'),
             parents={f.relative_to(ROOT).as_posix(): file_hash(f) for f in parents},
             source={name: file_hash(ROOT / name) for name in names}, warm_lineage=warm_lineage, parent_controls=controls,
             kinds=['none', 'full', 'interval_uniform', 'pointwise_nodes'],
             architecture='Same periodic single-coordinate FNO34209, identical dense/sparse observations and100/200 development IDs. No new modules or channels.',
             warm_policy='Reuse6 already-audited full-state1500 warm checkpoints from reshuffled development replication ONLY. Same training IDs/supervision; no pre-reshuffle warm. No new warm fitting. All optimizer/RNG/normalizers restored and hashes bound.',
             pointwise='TIME-pointwise PDE in the SAME projected spatial modes0..16: r=(ut-F(u))*T/observed_train_std. Query-time JVP create_graph=True, known initial offset repeated. Weighted SUM of pernode squared Parseval residuals with Simpson5[1,4,2,4,1]/12; NEVER square of weighted residual mean.',
             collocation='EXACT same five time nodes as interval: unchanged rawq RNG .05.. .5, alternatingh.005/.01, center=h+(T-2h)*(q-.05)/.45, five equally spaced nodes. Identical resolved band and4x-dealiased Allen cubic. Not an all-frequency/full-space PINN loss.',
             budget='24 new750-step continuations=18000updates, reuse6x1500warm=9000previousupdates. B16 lr.0003->.00001 wd.0001 eval250 checkpoint50, initial TRAIN-only gradientratio.3 eachobjective then fixed; same final step/no tuning/early stopping/seed selection. All3penaltyforwards everyarm, selected backward/JVP costs differ. Equal optimizer steps andnode layout NOT equalFLOPs orwalltime.',
             pairing='Withincondition/seed samewarm/init/optimizer/batch/query/calibration. Addedpointwise calibrationkey, preserve oldgd/full/interval subset. New18none/full/interval finals MUST reproduce parentmodel tensors exactly; these replays are not independent extra evidence.',
             primary='Direct INTERVAL versus POINTWISE paired comparison within dense4future and sparse3omitted primaries; also same3interior and4future metrics forbothconditions, all3seeds andall5times includingIC. No new effect/success threshold or hyperparameter search. Parent1%vsnone marker doesnotanswer this newhead-to-head.',
             secondary='Gradient/curvature primary/common/pertime, all unfavorable tradeoffs, originalnone/full controls. Report initial calibrationmultipliers, update-time/warmreuse/lastsegmentallocatedmemory/directinferencecost. No model export or singlemetric winner.',
             algebra_qa='On fixed first16 TRAIN warmprobe only: pointwiseenergy=weightedmeanresidualenergy+weightedresidualvariance, FP32 scalar rtol1e-5 atol1e-7. Endpoint-rate minus SimpsonaverageAD-ut RMS reported WITHOUT efficacy gate. Known identity not novelty/causation; quadrature gap prevents assuming old interval loss alwaysbelow pointwise.',
             limits='Single previously consumed development pool/split,3modelseeds,one transferredstrength andtwoobservationconditions. Matchinginitialgradientratio not identical physical strength or充分调参. Direct temporalpointwise projectedPDE comparator not unrestricted PINN orfull spatialPDE. No statistical significance or architecture novelty claim.',
             confirmation='No reserve/newtrajectory/reference/solver/reliabilitymap/tau access or adjustment. All cases and five development querytimes alreadyconsumed. Confirmation remains closed; no winner export.')
    if write:
        locked_json(OUT / 'plan.json', p)
    return p


def run(p):
    for name, expected in {**p['source'], **p['parents'], **p['data_chunks']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('sealed source/parent/cache changed: ' + name)
    clean, data = load_cached_split(p)
    groups, pair = {}, {}
    completed = 0
    for condition in p['conditions']:
        a, obs = observations(clean, condition)
        context = dict(plan=digest(p), task='allen_cahn', condition=condition, data=data, observations=obs)
        base = {k: p[k] for k in ['width', 'modes', 'layers', 'weight_decay', 'batch_size', 'eval_every', 'checkpoint_every',
                                  'resolved_max_mode', 'interval_half_widths', 'interval_nodes']}
        base.update(p['task_spec'])
        base.update(position='periodic', position_center=float(np.mean(a['grid256'])),
                    position_scale=float(np.std(a['grid256']) * np.sqrt(2)), evaluation_columns=p['evaluation_columns'][condition])
        for kind in p['kinds']:
            rows = []
            for seed in p['seeds']:
                warm = p['warm_lineage'][f'{condition}/s{seed}']
                if file_hash(ROOT / warm['latest_path']) != warm['latest_sha256']:
                    raise ValueError('audited full-state warm changed')
                c = dict(base, seed=seed, kind=kind, strength=0. if kind == 'none' else p['strength'],
                         steps=p['continuation_steps'], lr=.0003, lr_min=.00001)
                name = f'{condition}/{kind}_s{seed}'
                r = train_one(c, a, context, OUT / name, torch.device('cuda'), warm_path=ROOT / warm['latest_path'])
                key = (condition, seed)
                value = (r['initial'], r['batch_hash'], r['collocation_hash'], r['calibration'], r['parameters_real'])
                if key in pair and pair[key] != value:
                    raise ValueError('head-to-head fork pairing mismatch')
                pair[key] = value
                rows.append(r)
                completed += 1
                atomic_json(OUT / 'progress.json', dict(completed_runs=completed, total_runs=24, current=name,
                            groups=groups, confirmation_opened=False, exploratory=True))
            groups[f'{condition}/{kind}'] = summarize(rows, p, condition)
    result = dict(plan=digest(p), completed_runs=completed, total_runs=24, reused_warm_count=6,
                  groups=groups, exploratory=True, confirmation_opened=False)
    atomic_json(OUT / 'summary.json', result)
    atomic_json(OUT / 'progress.json', dict(result, status='completed'))
    print('ALLEN POINTWISE24 COMPLETE', json.dumps(groups), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['plan', 'run'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    p = plan()
    if args.stage == 'run':
        run(p)
