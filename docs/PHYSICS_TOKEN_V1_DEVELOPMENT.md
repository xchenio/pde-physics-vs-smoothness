# Burgers100-shot physics/token trajectory pilot, 2026-10-08

12 fits;100 labeled trajectories,5 times, three seeds; existing200 development.
Factorial gate/attention x physics0/.01, same initial tensors/batches/collocation/steps.
Primary averages per-trajectory relativeL2 over0.1/0.25/0.5/1.0; no time extrapolation.
512 is native output-query transfer with fixed256 input sensors, not full sensor transfer.
Shared probe PDE RMS is a secondary development diagnostic; never replaces field metric.

| Arm | Params | Train | Dev256 | Dev512 | PDE RMS | Training seconds mean |
|---|---:|---:|---:|---:|---:|---:|
| gate_p0 | 33256 | 0.361171 | 0.463866 | 0.463819 | 2.304690 | 61.58 |
| gate_p0.01 | 33256 | 0.381709 | 0.441088 | 0.441054 | 0.848744 | 64.15 |
| attention_p0 | 33256 | 0.359206 | 0.454306 | 0.454253 | 2.170541 | 73.41 |
| attention_p0.01 | 33256 | 0.378352 | 0.426508 | 0.426471 | 0.774084 | 123.29 |

Only one Burgers PDE; no Allen-Cahn physics validation. No weight or checkpoint tuning.
Retained FNO600 is a different sample/time budget and contextual only.
All model/data/source hashes must be committed and sealed before fresh500 targets.
Checkpoint/runtime logs and per-sample metrics retained in the versioned family.

Training seconds above are recorded workstation runtimes, not controlled efficiency
benchmarks: some runs overlapped regression/FNO jobs. Final inference profiles run
only after project training/tests complete. FNO100 controls, independently audited:
width16 params34261, train0.205927/dev0.287190; width32 params136613,
train0.103427/dev0.286988. Same100 trajectories,5 labels and batch sequence;
all6 FNO controls retained, no development-based width selection.
