"""Data-only rendering of locked confirmation metrics and resource records."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def report():
    folder=ROOT/'results/physics_token_v1';c=json.loads((folder/'confirmation.json').read_text())
    small=json.loads((ROOT/'results/fno_small_trajectory_v1/audit.json').read_text())
    lines=['# Locked100-shot Burgers confirmation, 2026-10-08','',
           '500 newly extracted trajectories AFTER all model/source/data hashes were committed.',
           'These IDs are now CONSUMED. No model, physics weight, seed or checkpoint retuned.',
           'All models use100 training trajectories, five time labels, same labeled minibatch',
           'sequence/1500 steps. Primary is trajectory-average relativeL2 over noninitial times.',
           '100 is the parameter-fitting label budget, not total research label access: earlier',
           'architecture/hyperparameter development used larger existing splits and reused dev.',
           'Gate/attention have exactly33256 active parameters; FNO width16 is a rough, not exact,',
           'capacity bracket. FNO uses five discrete known-time heads, no physics or query512.',
           '','| Model | Parameters | Confirm256 | Seed SD | Confirm512 | New512 query positions |',
           '|---|---:|---:|---:|---:|---:|']
    for arm,g in c['summary'].items():
        extra=np.mean([r['scores']['512_new_queries']['primary'] for r in c['records'][arm]])
        lines.append(f"| {arm} | 33256 | {g['256']['mean']:.6f} | {g['256']['seed_sd']:.6f} | {g['512']['mean']:.6f} | {extra:.6f} |")
    for width,g in c['small_fno_summary'].items():
        count=small['summary']['groups'][width]['parameters_real']
        lines.append(f"| FNO100 width{width} | {count} | {g['mean']:.6f} | {g['seed_sd']:.6f} | unavailable | unavailable |")
    lines.extend(['','## Four registered primary contrasts','',
                  'Paired10k trajectory bootstrap, three fits averaged per trajectory;',
                  'Bonferroni4 two-sided intervals; pass requires >=1% gain AND upper interval<0.',
                  'Conditional on fitted seeds; not a bootstrap of training seed uncertainty.',
                  '','| Contrast | Difference | Relative gain | Adjusted interval | Pass |',
                  '|---|---:|---:|---|---|'])
    for name,r in c['comparisons'].items():
        lo,hi=r['interval'];lines.append(f"| {name} | {r['difference']:.6f} | {100*r['relative_gain']:.3f}% | [{lo:.6f}, {hi:.6f}] | {r['success']} |")
    lines.extend(['','FNO100 contrasts are descriptive strong controls, not additional registered tests.',
                  f"Descriptive physics x attention interaction difference: {c['interaction_difference_descriptive']:.6f}.",
                  '','## Timing and memory','',
                  'Network-only latency, batch64,1280 queries (five256-point fields),10 warmup/50',
                  'synchronized host-inclusive repeats per seed, no concurrent project trainer/tests.',
                  'Reported latency is mean across three fitted models. Incremental allocated peak',
                  'is added inference allocation above live-process baseline, not total/reserved VRAM.',
                  'Training seconds are recorded workstation runtimes, include competing workloads',
                  'in some runs, and are NOT controlled efficiency benchmarks. FNO omits PDE work.',
                  '','| Model | Optim seconds mean | Batch64 latency ms | Incremental inference MiB | Training-segment peak MiB |',
                  '|---|---:|---:|---:|---:|'])
    development=json.loads((folder/'audit.json').read_text())
    for arm,rows in c['records'].items():
        times=np.mean([r['seconds'] for r in development['records'][arm]])
        latency=np.mean([r['profiles']['256']['mean_ms'] for r in rows])
        peak=np.mean([r['profiles']['256']['incremental_peak_allocated_bytes'] for r in rows])/2**20
        training=np.mean([r['peak_allocated_bytes_last_segment'] for r in development['records'][arm]])/2**20
        lines.append(f'| {arm} | {times:.2f} | {latency:.3f} | {peak:.3f} | {training:.3f} |')
    for width,rows in c['small_fno_records'].items():
        times=np.mean([r['seconds'] for r in small['records'][width]])
        latency=np.mean([r['profile']['mean_ms'] for r in rows]);peak=np.mean([r['profile']['incremental_peak_allocated_bytes'] for r in rows])/2**20
        training=np.mean([json.loads((ROOT/'results/fno_small_trajectory_v1'/f"w{width}_s{r['seed']}"/'result.json').read_text())['peak_allocated_bytes_last_segment'] for r in rows])/2**20
        lines.append(f'| FNO100 width{width} | {times:.2f} | {latency:.3f} | {peak:.3f} | {training:.3f} |')
    lines.extend(['','Token512 latency/peak and secondary field/gradient/curvature/PDE results are retained',
                  'per model in confirmation.json. Query512 keeps input sensors256, and extra odd512',
                  'positions are scored separately. No full-input-resolution or temporal extrapolation',
                  'claim: all reported times were supervised, only one Burgers PDE has physics evidence.',
                  'Retained600-shot FNO final-time-only outcomes are contextual in confirmation_fno_context.json.',
                  'No universality, intrinsic-manifold discovery or combination-alone novelty claim.',''])
    return '\n'.join(lines)


if __name__=='__main__':print(report(),end='')
