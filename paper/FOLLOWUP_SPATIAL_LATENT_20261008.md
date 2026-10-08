# Spatial latent tokens: staged hypothesis and development diagnostics

This is a new exploratory line, not a demonstrated method contribution. Full
protocol: docs/MANIFOLD_OPERATOR_ROADMAP_V1.md. All old confirmation sets remain
consumed. Reconstruction and fixed-time operator results must not be conflated.

## Stage1: compression quality

81 neural fits compare global MLP, physically ordered periodic CNN, and identically
parameterized CNN with a fixed spatial permutation; POD fitted to the same400
training output snapshots is retained. Scalar bottlenecks16/32/64 have no per-sample
mean side channel. One training scalar mean/std is shared. Existing200 development
snapshots are repeatedly used; three seeds, fixed1500-step field-MSE objective.

The physically ordered CNN beats the scrambled and global comparators in mean
field, first-derivative and second-derivative error at all nine task/rank settings.
The local/scrambled pair has identical active parameters, trainable initialization
and batches. This supports alignment with physical neighborhoods for this registered
CNN, not superiority over all global-vector architectures: the global MLP differs
in topology/capacity and no architecture-specific tuning was done.

At rank16, independent Burgers field error improves from POD0.124900 to CNN0.116420
(6.79%); synthetic Allen-Cahn improves0.025837 to0.017695 (31.51%). Original
Burgers is essentially tied (POD0.115408 versus CNN0.115602). At rank32 and64,
POD field reconstruction is better on every task. Thus nonlinear compression is
not uniformly superior, and increasing latent scalars alone does not fix this CNN.

The derivative comparison changes the interpretation. Allen-Cahn rank16 CNN first
derivative error0.361935 versus POD0.190273, second derivative8.101366 versus0.506683,
despite the improved field error. Rank64 CNN field0.010969/curvature7.017581 versus
POD0.000006/0.001347. Artificial high-frequency output is consistent with the
nonzero65-128 band error0.002738 on an almost band-limited target. This is a
diagnostic observation, not an isolated causal attribution to nearest upsampling.
For Burgers rank64, CNN curvature is lower than POD (original1.211940 vs1.478754,
independent1.320474 vs1.375499), but field and first derivative are worse.

![Audited representation trade-offs](figures/manifold_reconstruction_20261008.png)

Figure: reused development200; means and seed SD of three AE fits, not confidence
intervals. Same scalar budgets; global capacity differs. Derivatives computed on
the periodic256-point grid, with training-only denominator protection, not certified
continuum derivatives or PDE residuals. Every encoder sees the true output snapshot.
Full field/derivative/band tables: docs/MANIFOLD_RECONSTRUCTION_V1_RESULTS.md.
81 final checkpoints independently replayed; initial/batch/parameter hashes, POD
orthogonality and per-sample Parseval band closure checked. No new holdout opened.

## Implications and subsequent controls

Retain POD as a strong representation control. The promising hypothesis is a
field-versus-derivative trade-off and neighborhood alignment, rather than universal
nonlinear-manifold advantage. A coordinate-smooth decoder and physical training
must demonstrate increment beyond the data-only operator, not rely on reconstruction.

Stage2 registers CNN64 branch and periodic Fourier/tanh coordinate trunk; stage3
adds identically parameterized four-head attention versus an active gated pointwise
control. All initial weights and batches match in the attention contrast; temporary
token width expansion is distinct from the64-scalar bottleneck. Both stages are
fixed-time data-only diagnostics and retain full-field FNO as a strong contextual
baseline. Their completed replay-audited outcomes are reported below.

## Stage2: fixed-time operator baseline

All9 CNN64/smooth-trunk fits completed and independently replayed. Parameters24356
versus retained full FNO136481. Same600 fitting rows, normalized weighted field loss,
minibatches/seeds and1500/6000 step budgets. CNN DeepONet development relativeL2:
original Burgers0.383864, independent0.405222, Allen-Cahn0.023343, all worse than
FNO0.195846/0.220706/0.003315. Therefore continuous coordinate decoding does not by
itself close the encoder/readout gap. Capacity and architecture both differ, so this
does not uniquely identify the cause or establish an optimized DeepONet limit.

## Stage3: matched attention-versus-gate operator control

18 fits completed and replayed,33256 active parameters each; exactly matching
initial tensors and minibatch sequences. Original Burgers gate0.376506 versus
attention0.288996; independent0.396253 versus0.332460, consistent with useful
token interaction on these repeated-development tasks. Allen-Cahn gate0.022733
versus attention0.027256; no universal gain. All remain worse than full FNO.
The contrast isolates the registered attention/gated-QKV block, not every possible
nonattention mixer. Equal capacity/steps does not establish equal compute or speed.
Full table and per-seed replay: docs/TOKEN_OPERATOR_V1_RESULTS.md.

## Stage4/5: completed100-shot physics/attention factorial

One Burgers PDE,12 fits, supervised100 trajectories at0/.1/.25/.5/1. Main outcome
is mean per-trajectory relativeL2 over noninitial times. Same32 continuous collocation
points and initial states/batches across gate/attention x physics weight0/.01.
No tuning of weight or checkpoints. Input256 fixed; native512 output queries secondary,
and newly queried odd512 positions are reported separately in confirmation.
Physics-off also computes residuals for diagnostic cost matching. All four arms
were sealed and committed BEFORE500 freshly reserved trajectory labels were read.
The conservative historical registry7829 IDs is disjoint from the reserved500.
Four registered primary contrasts use paired trajectory bootstrap10k, Bonferroni4
intervals and at least1% gain. Interactions are descriptive; uncertainty conditional
on three fits, not a full training-seed uncertainty model. Retained FNO600 final-time
control is contextual, not a matching100-shot/multitime baseline.

The pilot uses trajectory supervision and tested physical-unit u_t/u_x/u_xx; initial
condition is supervised at t0 and periodic trunk encodes boundary conditions.
Separate data/PDE losses and gradient norms are recorded. Architecture/hyperparameter
design used earlier larger data:100 is the current fitting label count, not the total
label access during research. No pretrained true-output encoder is used for prediction.

All126 fits in this five-stage first round were completed and independently audited.
Six stronger FNO100 controls were added before confirmation: widths16/32,five known
time heads, identical100 fitting trajectories/time labels/loss/steps/batch sequence.
Width16 has34261 real parameters versus33256 for token models (rough capacity bracket,
not exact equality); width32 has136613. Both widths remain fixed, no selection.
Training runtime observations had competing workstation jobs in some segments and
are not controlled efficiency benchmarks. Final inference timing was warmed and
synchronized after project trainers/tests stopped.

Seal commit05b8153 locked all12 factorial models,6FNO100 and3 historical FNO600
checkpoints, sources and rules BEFORE extracting500 new trajectories, disjoint from
the7829-ID conservative historical registry. These500 targets are now consumed.
Confirmation outcomes (primary: mean per-trajectory relativeL2 over four noninitial
times, averaged over three fitted seeds):

| Model | Confirm256 | Confirm512 | Parameters |
|---|---:|---:|---:|
| Gate, data-only | 0.486568 | 0.486562 | 33256 |
| Gate + physics | 0.458865 | 0.458862 | 33256 |
| Attention, data-only | 0.470762 | 0.470761 | 33256 |
| Attention + physics | 0.442270 | 0.442272 | 33256 |
| FNO100 width16 | 0.313237 | not supported | 34261 |
| FNO100 width32 | 0.318333 | not supported | 136613 |

Four predeclared contrasts pass>=1% gain and Bonferroni4 adjusted paired interval
upper<0: attention without physics3.249%; physics without attention5.694%; physics
with attention6.052%; combination versus gate9.104%. Uncertainty conditions on three
fixed fits. Physics effects improve each paired seed; attention-alone direction is
not unanimous across seeds. Interaction difference-0.000788 is descriptive and does
not establish a strong special synergy. No weights/checkpoints/seeds retuned.

Crucially, the best hybrid remains41.19% worse than near-capacity FNO16 in descriptive
mean error. Positive incremental module results are not evidence of state-of-the-art
architecture. There are no registered FNO superiority tests; FNO outcomes are stronger
descriptive controls on the same fresh trajectories. No claim of total training-data
optimality, universal performance, or novelty by stacking existing methods.

![Fresh confirmation accuracy and network latency](figures/physics_token_confirmation_20261008.png)

Figure:500 targets were unseen until the fixed seal, and are now consumed. Error bars
are three-seed SD, not confidence intervals. Batch64 network-only latency predicts
all five256-point fields;10 warmup/50 synchronized host-inclusive repeats per seed.
Attention+physics mean1.918ms versus FNO16 1.967ms is not a robust speed superiority
claim. Incremental inference allocation1.327 versus5.023MiB excludes live baseline;
full PyTorch-allocated training peaks69.614 versus70.970MiB are recorded separately.
These are allocator observations, not whole-device VRAM or reserved/context memory.

Native512 transfer retains256 input sensors. Extra odd512 query-position relative
error0.442270 for the combination; no full input-resolution transfer or improved
high-frequency reconstruction claim. All queried times were supervised, not temporal
extrapolation; physics has been tested on one Burgers PDE only. Allen-Cahn and
independent-split physics validation remain future work. Final confirmation replay
checks18 same100-shot model outputs, odd-query metrics, all summaries and4 decisions.
Complete metrics/intervals/costs: docs/PHYSICS_TOKEN_V1_CONFIRMATION.md.

Next priority: diagnose/generalize the strongest100-shot FNO, compare physics versus
matched derivative/smoothness regularization and isolate where benefits arise before
adding further blocks. Register any new independence/PDE test before fresh targets;
do not recycle these500 as an untouched holdout.

Related frameworks already exist: [DeepONet](https://www.nature.com/articles/s42256-021-00302-5),
[physics-informed DeepONet](https://arxiv.org/abs/2103.10974), and
[latent DeepONet](https://www.nature.com/articles/s41467-024-49411-w).
Their combination alone is not a novelty claim.
