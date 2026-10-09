# Paired periodic-position and curvature-band controls (2026-10-09)

Two reused development tasks:100-trajectory dense CLEAN Burgers and nonlinear
synthetic Allen-Cahn, five labels each, development200. No confirmation access.
Same16wide/16mode/4layer FNO and34209 active real parameters in every arm;
zero final output head at initialization. No new layer/channel/network module.

Factorial2 coordinate features x3 smoothing choices x3 seeds x2 PDEs. Coordinate
features are original linear x or a SINGLE periodic scalar
mean(grid)+sqrt2*std(grid)*sin(2*pi*x/L), constants from grid only. Both have
approximately matched marginal mean/variance. The periodic scalar is NOT
injective fullsin/cos encoding, and does not hard-guarantee smooth periodic
output. Absolute-position information changes; do not overclaim that only one
causal seam mechanism has been altered. Periodicity and learned output metrics
must both be checked.

Each coordinate feature starts with the same initial trainable tensors and its
OWN1500-step data-only fit. No feature replacement on pretrained linear weights.
Three shared model/optimizer/RNG750-step continuations per feature: none,
low-mode0..16 curvature, full-mode0..128 curvature. Same labeled minibatch and
query-time streams across all matched seed arms. Three-seed averaging, fixed
final2250-step models, no validation checkpoint/seed selection.

Initial TRAIN-only gradient ratio0.3 for low/full penalties at frozen warm start;
hold weights fixed thereafter. Full Fourier Parseval uses single Nyquist energy.
Both penalty forwards computed in every continuation; backward costs differ.
Fitting loss is the inherited guarded normalized squared field loss. All binary
states/optimizer/calibration/RNG/history persist atomically every50 steps.

48 fits total:12 warm starts plus36 final paired models. Primary is mean relative
fieldL2 over four noninitial times; full-grid gradient/curvature, frequency shares,
boundary8 concentration and seam error are secondary. Exploratory screening,
fixed before outcomes: field<=1.01*same-task linear_none AND curvature<=0.5*
linear_none. Retain every result, no selected postprocessor or independent claim.

Historical reproducibility requirement:18 linear warm/none/low final states must
match their previous dense-clean counterparts tensor-for-tensor. Tests cover
unchanged parameter states/capacity, exact linear forwarding, periodic scalar,
low penalty equivalence, full Parseval/Nyquist and CPU/CUDA exact resume/pairing.
Adapter globals restore on completion/failure; runner is single-writer only.
No historical sealed source file is changed. Source/plan commit before training.

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_periodic_curvature.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_periodic_curvature.py --require-complete
```

Re-run SAME command after shutdown, without --resume. If a worker survives sleep,
do not duplicate it. Completed result/checkpoint identities are checked then
skipped; at most<50 unsaved steps repeated. Checkpoint binaries excluded from
Git, back up locally. Both earlier500 Burgers sets remain consumed; reserved
Allen-Cahn500 not generated. This is controlled development evidence, not a
new architecture/physics-generalization claim or time-dynamics success.
