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
baseline. Outcomes will be appended after completed replay audits.

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

## Stage4/5: registered100-shot physics/attention factorial

One Burgers PDE,12 fits, supervised100 trajectories at0/.1/.25/.5/1. Main outcome
is mean per-trajectory relativeL2 over noninitial times. Same32 continuous collocation
points and initial states/batches across gate/attention x physics weight0/.01.
No tuning of weight or checkpoints. Input256 fixed; native512 output queries secondary,
and newly queried odd512 positions will be reported separately in confirmation.
Physics-off also computes residuals for diagnostic cost matching. All four arms
will be sealed and committed before500 freshly reserved trajectory labels are read.
The conservative historical registry7829 IDs is disjoint from the reserved500.
Four registered primary contrasts use paired trajectory bootstrap10k, Bonferroni4
intervals and at least1% gain. Interactions are descriptive; uncertainty conditional
on three fits, not a full training-seed uncertainty model. Retained FNO600 final-time
control is contextual, not a matching100-shot/multitime baseline.

Stage4 needs trajectory supervision and tested physical-unit u_t/u_x/u_xx; constant
time training is insufficient. Periodic boundary conditions, initial condition,
data/PDE objective gradient scales, and native-resolution targets must be checked.
Stage5 locks a factorial attention x physics comparison before genuinely unused
trajectory extraction; no reused confirmation can be relabeled as unseen.

Related frameworks already exist: [DeepONet](https://www.nature.com/articles/s42256-021-00302-5),
[physics-informed DeepONet](https://arxiv.org/abs/2103.10974), and
[latent DeepONet](https://www.nature.com/articles/s41467-024-49411-w).
Their combination alone is not a novelty claim.
