# Locked100-shot Burgers confirmation, 2026-10-08

500 newly extracted trajectories AFTER all model/source/data hashes were committed.
These IDs are now CONSUMED. No model, physics weight, seed or checkpoint retuned.
All models use100 training trajectories, five time labels, same labeled minibatch
sequence/1500 steps. Primary is trajectory-average relativeL2 over noninitial times.
100 is the parameter-fitting label budget, not total research label access: earlier
architecture/hyperparameter development used larger existing splits and reused dev.
Gate/attention have exactly33256 active parameters; FNO width16 is a rough, not exact,
capacity bracket. FNO uses five discrete known-time heads, no physics or query512.

| Model | Parameters | Confirm256 | Seed SD | Confirm512 | New512 query positions |
|---|---:|---:|---:|---:|---:|
| gate_p0 | 33256 | 0.486568 | 0.006951 | 0.486562 | 0.486552 |
| gate_p0.01 | 33256 | 0.458865 | 0.005585 | 0.458862 | 0.458856 |
| attention_p0 | 33256 | 0.470762 | 0.012892 | 0.470761 | 0.470757 |
| attention_p0.01 | 33256 | 0.442270 | 0.005043 | 0.442272 | 0.442270 |
| FNO100 width16 | 34261 | 0.313237 | 0.000741 | unavailable | unavailable |
| FNO100 width32 | 136613 | 0.318333 | 0.006141 | unavailable | unavailable |

## Four registered primary contrasts

Paired10k trajectory bootstrap, three fits averaged per trajectory;
Bonferroni4 two-sided intervals; pass requires >=1% gain AND upper interval<0.
Conditional on fitted seeds; not a bootstrap of training seed uncertainty.

| Contrast | Difference | Relative gain | Adjusted interval | Pass |
|---|---:|---:|---|---|
| attention_without_physics | -0.015807 | 3.249% | [-0.023463, -0.009010] | True |
| physics_without_attention | -0.027703 | 5.694% | [-0.038032, -0.020183] | True |
| physics_with_attention | -0.028491 | 6.052% | [-0.038879, -0.021158] | True |
| combination_vs_gate | -0.044298 | 9.104% | [-0.058877, -0.033587] | True |

FNO100 contrasts are descriptive strong controls, not additional registered tests.
Descriptive physics x attention interaction difference: -0.000788.

## Timing and memory

Network-only latency, batch64,1280 queries (five256-point fields),10 warmup/50
synchronized host-inclusive repeats per seed, no concurrent project trainer/tests.
Reported latency is mean across three fitted models. Incremental allocated peak
is added inference allocation above live-process baseline, not total/reserved VRAM.
Training seconds are recorded workstation runtimes, include competing workloads
in some runs, and are NOT controlled efficiency benchmarks. FNO omits PDE work.

| Model | Optim seconds mean | Batch64 latency ms | Incremental inference MiB | Training-segment peak MiB |
|---|---:|---:|---:|---:|
| gate_p0 | 61.58 | 1.713 | 1.327 | 69.614 |
| gate_p0.01 | 64.15 | 1.685 | 1.327 | 69.614 |
| attention_p0 | 73.41 | 1.987 | 1.327 | 69.614 |
| attention_p0.01 | 123.29 | 1.918 | 1.327 | 69.614 |
| FNO100 width16 | 41.91 | 1.967 | 5.023 | 70.970 |
| FNO100 width32 | 38.81 | 2.215 | 10.047 | 77.554 |

Token512 latency/peak and secondary field/gradient/curvature/PDE results are retained
per model in confirmation.json. Query512 keeps input sensors256, and extra odd512
positions are scored separately. No full-input-resolution or temporal extrapolation
claim: all reported times were supervised, only one Burgers PDE has physics evidence.
Retained600-shot FNO final-time-only outcomes are contextual in confirmation_fno_context.json.
No universality, intrinsic-manifold discovery or combination-alone novelty claim.
