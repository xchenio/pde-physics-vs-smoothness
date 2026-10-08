# Current conclusions

> Latest2026-10-09 diagnosis COMPLETE,36 frozen final models,80 tests pass.
> Allen-Cahn data-only curvature3.694:90.17% error energy at modes65..128,
> 98.25% at first/last4 nodes; denominator floor0% active. Periodic-seam artifact
> is strongly localized, but linear-coordinate causation is not established.
> Fixed cutoff16 counterfactual curvature0.325 with field0.042723 vs0.043007;
> NOT selected model, training gain or independent filter verification.
> Time JVP/own FD agrees~5e-5..2.3e-4, yet Allen u_t error~32%; reference RHS
> disagreement~1e-6. Learned dynamics, not derivative-code failure, is implicated.
> Burgers native FD width sensitivity44.40% full/12.34% resolved; do not attribute
> all full-PDE residual to space error or report sampled FD as exact derivative gold.
> No retraining/new module/fresh confirmation. Next periodicity/full-band smoothing
> control on development, then reliable time-constraint diagnosis; keep strong baselines.

> Latest2026-10-09 physics_stress_v1 COMPLETE:48 fits,48 independent checkpoint/
> label/batch/query/fork/summary replays. Fixed architecture and transferred0.3.
> Burgers sparse none/PDE/curvature0.644463/0.603849/0.589233;
> noisy dense0.273533/0.270520/0.267287; noisy sparse0.648667/0.611887/0.599397.
> PDE helps vs none but does NOT beat curvature in any Burgers stress condition.
> NEW nonlinear synthetic Allen-Cahn dense clean0.043007/0.042646/0.043966:
> PDE beats both in all3 seeds, but only0.839% benefit versus none on development;
> not second-PDE independent confirmation or fully tuned optimal smoothing.
> Allen-Cahn second-derivative errors~3.7 despite field errors~0.043; cannot
> claim physically faithful reconstruction from field loss alone. Both prior500
> consumed; new Allen-Cahn reserved500 not generated. No fresh confirmation opened.
> Next: development-only frequency/time-derivative diagnosis before new rules/
> independent confirmation, not additional architecture stacking.

> Earlier2026-10-09 protocol snapshot BEFORE training:
> physics_stress_v1, same34209-parameter time FNO; sparse-time/noisy-label/combined
> Burgers36 fits then new nonlinear Allen-Cahn12 fits. Independent observation-
> only warm starts, transferred0.3 penalty-gradient strength, curvature/none controls.
> Allen-Cahn data/residual solver QA passed;77 regression tests pass. No fresh
> stress confirmation opened, and no second-PDE learned success claimed yet.

> Latest strong-FNO follow-up COMPLETE:3 continuous-time baselines +30 paired
> continuations;30 checkpoint/batch/collocation/selection replays passed.
> Seal4d60f18 preceded a NEW500-trajectory extraction, disjoint from conservative
> historical8329 (including previous consumed500). This new500 is NOW CONSUMED.
> Confirmation none0.290253, PDE0.286260, gradient0.286076, curvature0.282820.
> All three penalties pass registered>=1%gain/Bonferroni5 intervals versus none;
> PDE-vs-gradient fails (interval crosses0), PDE-vs-curvature fails (PDE worse).
> Curvature gain2.561%, PDE1.376%: no unique physical-loss predictive advantage.
> All12 confirmation model predictions/five decisions replayed. Fixed2250-step
> budget/34209 params/100x5 labels; one PDE, seen times,256 sensors only.
> Keep curvature smoothing as a strong baseline; next test sparse-time/noisy
> supervision and a second nonlinear PDE. Do not justify adding blocks from this
> evidence or equate existing PINO ingredients with novelty.

> Earlier prerequisite snapshot (BEFORE new confirmation): training-only FNO
> physics discretization audit passed for
> FIXED modes0..16 (median resolved residual1.8%-4.2%), but almost-full256 band
> remains25%-51%. This supports testing a resolved constraint, NOT full-grid
> physics enforcement or a performance claim. A new continuous-time data-only
> FNO baseline is complete: development mean0.277170 vs historical0.287190;
> this is an architecture/control result, not a physical-loss or fresh-test gain.
> Historical five discrete heads do not define u_t. Thirty paired regularization
> continuations are registered, with training-only gradient calibration.
> No new confirmation targets opened; previous500 remain consumed.

> Latest2026-10-08 spatial latent five-stage FIRST ROUND COMPLETE:126 new fits
> audited (81 compression,9CNN/trunk,18attention,12physics,6matched-data FNO).
> Seal05b8153 preceded500 new Burgers target extraction; all500 NOW CONSUMED.
> Confirm100-shot primary gate0.486568,gate+P0.458865,attention0.470762,
> attention+P0.442270. Registered gains3.249%/5.694%/6.052%/9.104%, all4 pass
> >=1% and Bonferroni4 paired interval gates, conditional on3 fitted seeds.
> Same100-shot FNO16/32:0.313237/0.318333. FNO16 has34261 params vs token33256;
> best hybrid is41.19% worse than FNO16 in descriptive mean error. No competitive
> architecture/combination-alone novelty or strong interaction claim (interaction-0.000788).
> Native512 errors/extra query positions replayed at fixed256 input, not full input-grid
> transfer; all eval times supervised; one-PDE physics only, Allen physics untested.
> 100 labels refers to current parameter fitting, not total previous design/development access.
> 68 tests pass;18 current confirmation models/4 contrast decisions independently replayed.
> Timing/VRAM records and paper figures verified. Next PRIORITY: diagnose strong100-shot
> FNO physics/generalization and separate PDE-specific effects from matched smoothness
> regularization, before deeper stacks; additional independent split/PDE remains pending.
> Tables: PHYSICS_TOKEN_V1_CONFIRMATION.md; paper/FOLLOWUP_SPATIAL_LATENT_20261008.md.

> New2026-10-08 spatial latent stage1:81 AE fits independently replayed, POD fitted
> on same representation400, reused dev200, no fresh confirmation. Local CNN beats
> global MLP and exactly capacity/initialization-matched scrambled CNN in all9 mean
> field/gradient/curvature comparisons. This is registered neighborhood alignment,
> not universal global-vector inferiority. Nonlinear compression does NOT uniformly
> beat POD: field gains at independent Burgers/Allen rank16 only; POD better at32/64.
> Allen rank16 field0.017695 vs POD0.025837, BUT curvature8.101366 vs0.506683.
> Retain POD/FNO; field accuracy cannot certify physics or derivative quality.
> Stages2/3 and first Burgers100-shot stages4/5 subsequently completed (latest above).
> All old confirmation sets remain consumed. See
> MANIFOLD_RECONSTRUCTION_V1_RESULTS.md and paper/FOLLOWUP_SPATIAL_LATENT_20261008.md.

> Latest2026-10-08 shared FNO encoder/pointwise-head ablation:36 paired fits complete.
> Full/POD16/POD32/POD64 dev L2: original0.195846/0.191759/0.188328/0.190009;
> independent0.220706/0.218613/0.216032/0.215583;
> Allen-Cahn0.003315/0.026170/0.003653/0.003207. All136481 parameters; paired initial
> states and batches independently audited;9 full checkpoints reproduce old tensors bitwise.
> Allen-Cahn post-hoc projections0.026098/0.003552/0.003182 beat training projection
> in every rank/seed. Extra training-projection benefit is task/rank-dependent.
> The fixed POD output constraint alone cannot explain old pooled-head deficits; same-space
> shared pointwise outputs are much stronger. Prior pooled comparisons also change parameter
> count and initialization ordering, so head geometry is not a uniquely isolated cause.
> Next: seal full/post-hoc/training-projection controls before unused confirmation, retaining
> post-hoc projection as a strong baseline. No new confirmation opened here.57 tests pass;
> tables:FNO_PROJECTION_V1_RESULTS.md; manuscript:paper/FOLLOWUP_FNO_PROJECTION_20261008.md.

> Latest2026-10-08 matched-parameter spectral/POD study:54 new fits complete/audited.
> Best spectral: Burgers original rank16 0.273799, independent rank16 0.302838,
> Allen-Cahn rank64 0.006762. Best matched local convolution:0.373109/0.380088/0.004215.
> Spectral wins all tested Burgers ranks; local wins all Allen-Cahn ranks. More rank
> worsens Burgers generalization, while Allen-Cahn benefits mostly from16→32.
> No new model beats old full-field FNO0.195846/0.220706/0.003315. Rank64 has184608
> parameters vs FNO136481, so parameter efficiency is not established either.
> Exact squared Frobenius decomposition is audited, not inferred from captured energy.
> Next: identical FNO backbone/shared pointwise head, full output vs fixed POD projection
> during training, to isolate output restriction from pooled coefficient-head differences.
> That is proposed, not performed or established novelty. No new confirmation opened;
> reused development does not certify independent generalization. Tables:SPECTRAL_POD_V1_RESULTS.md.

> Latest architecture diagnosis:108 development-only fits completed and replay-audited.
> Direct FNO dev relative L2: original Burgers0.195846, independent0.220706,
> synthetic Allen-Cahn0.003315. Best MLP residual:0.363701/0.402240/0.287272.
> CNN same-POD helps Allen-Cahn (direct0.046134), not consistently Burgers.
> FNO projected to same POD:0.192454/0.217435/0.026098. Thus Burgers has a large
> mapping/encoder gap, while rank16 becomes a hard accuracy limit for Allen-Cahn.
> Parameter count also differs (8752/8848/136481); no isolated encoder causal claim.
> Next: parameter-matched spectral/POD controls and rank16/32/64, then lock rules
> before genuinely unused confirmation. No new confirmation opened in this round.
> Full tables: ARCHITECTURE_STAGES_V1_RESULTS.md; all52 regression tests passed.

> Latest2026-09-30 UNCERTAINTY_MODAL_V1: continuous uncertainty penalty fails to beat
> old continuous/all/matched-global controls. Independent Burgers error0.384878 versus
> old0.382451,0.6345% worse, adjusted difference interval[0.001133,0.003818]. Original
> Burgers has no confirmed primary gain; Allen-Cahn nonlinear arms are exact repeats.
> Four more500-ID confirmation sets are now consumed; sealbfd52cd preceded extraction.
> Next diagnostic: fixed penalty1 implies unconstrained correction r/2, not full r.
> Allen-Cahn train actual0.353639 vs penalty1 oracle0.229930 vs POD oracle0.026088.
> This supports investigating coefficient fitting/regularization before replacing the
> decoder; not uniquely network capacity or optimization. Then compare spatial branch
> and FNO with matched data/budgets. No architecture implementation started in this review.

> Latest2026-09-29 FIELD_MODAL_V1: whole-field relevance (including projection error)
> keeps original Burgers modes1/2/4, independent1/2, Advection none, synthetic Allen-Cahn
> 1–15. Sourceb2d554c, selectionseal8a4020b BEFORE four new500-ID confirmation sets.
> New confirmation gains versus tuned linear:3.0636%,3.1939%,21.6550% for the three
> nonlinear cases, passing predeclared effect and12-comparison adjusted intervals.
> Advection rejects correction. Allen-Cahn is custom numerically validated synthetic
> data, NOT a downloaded public benchmark. Eleven new fits; old public models reused.
> No new rule superiority over energy or old selector is confirmed. Allen-Cahn is
> slightly worse than old all-mode correction (0.0521%, difference interval above0).
> Low contribution does not imply a coefficient should be deleted. Prefer retaining
> old strong controls; investigate continuous field-aware shrinkage/uncertainty only
> under a new protocol, not by retuning on these2000 now-consumed confirmation targets.

> Latest2026-09-29: INCREMENTAL_MODAL_V1.md supersedes the next-step proposals below.
> Nested training-only tuning strengthens linear maps;33 neural fits compare identical
> residual predictions under none/all/stable/energy/shrinkage. Candidates committed at
> ad94e69 BEFORE three500-ID confirmation sets were opened. Original Burgers stable
> improves0.387273→0.375353 (3.0779%); independent Burgers shrink improves0.379517→0.368605
> (2.8752%). Both pass>=1% gain and Bonferroni-adjusted paired bootstrap criteria.
> This confirms incremental correction over the tested tuned linear comparator, not
> global linear optimality or superiority over FNO. Hard masks still equal energy order.
> Rule superiority over energy FAILS: original identical; independent gain0.2013% with
> interval crossing zero. Advection falls back to none, L2=0.024182229260. All three
> confirmation sets are now CONSUMED; no adaptive reuse as untouched holdouts.
> Next research: field-risk relevance/rejection with matched energy/shrinkage controls,
> and a genuinely second nonlinear PDE. These are proposals, not validated innovations.

> 2026-09-28 objective/modal follow-up: see OBJECTIVE_MODAL_V1.md and the dated paper
> addendum. Guarded normalized squared loss improves full-data development relative L2
> from0.366400 to0.358258 on original Burgers and0.407464 to0.398468 on a completely
> disjoint training/development split (five paired seeds improve on each). This is not
> the nonsquared primary metric itself, nor a fresh confirmation of the old model.
> Strict train-internal CV selects leading4 and2 modes respectively, identical to energy
> truncation. No incremental selection-rule novelty or superiority is established.
> The earlier500-ID confirmation failure below remains valid. New reserved labels stay
> unopened until fixed models, comparisons and success criteria are committed.
> Final status:123/123 records complete,48 ridge grid refits. Advection is a linear
> negative control: normalized loss worsens0.169283→0.175051; tuned pure ridge0.015378
> is far better. All16 modes are selected, making mask controls degenerate. The rule
> detects correctable error relative to a weak anchor, not uniquely nonlinear benefit.
> Next priority: cross-fit incremental benefit beyond tuned linear correction, with
> no-correction and shrinkage controls. No cross-PDE universal improvement is supported.

> 2026-09-28 frozen-ridge residual historical update: see RIDGE_RESIDUAL_V1.md.
> 40 runs +8 ridge refits completed; development selects width32/penalty1 with L2=0.366400
> versus frozen ridge0.373656. Selection was committed BEFORE opening500 fresh trajectories.
> Confirmation L2=0.405135 versus frozen ridge0.405354 (only0.054% gain; paired95% interval
> for difference[-0.007498,0.008923]). Predeclared primary success FAILS. Expanded ridge
> and initial mean are slightly better in confirmation L2. Secondary RMSE improves5/5
> seeds, mean0.147701 versus0.150931. Do not replace the failed primary endpoint with RMSE.
> Next: development-only objective/metric alignment diagnostics, then modal learnability
> shrinkage with cross-fitting. Both are proposals; reserve a NEW confirmation set before
> adaptive follow-up. The500 current confirmation IDs are now consumed, not reusable holdout.

> 2026-09-28 same-POD update (latest): 30/30 readout runs completed; see
> PUBLIC_POD_READOUT_V1.md. Identical POD bases / input normalization / mean offsets:
> nonlinear MLP improves train L2 and RMSE in 15/15 pairs versus Adam affine control,
> but worsens both validation metrics in 15/15. MLP validation means at ranks16/32/64
> are 0.398761 / 0.401114 / 0.402642, all worse than ridge and mean baseline.
> Coefficient mismatch contributes roughly 93–99% of MLP validation squared error,
> but train coefficients are also not fully fit. This localizes error to coefficients,
> not uniquely to insufficient nonlinearity or optimization. Next: restrained nonlinear
> correction to a frozen ridge map, regularization/early stopping, fresh confirmation split.

> 2026-09-28 follow-up: PUBLIC_SCALE1000_V1.md reports 20 completed n=1000 controls
> and 18 POD ridge candidates. Cosine improves validation relative L2 in 10/10 pairs,
> unlike n=128; full training fit remains unmet. MLP cosine mean 0.374807 is slightly
> worse than baseline 0.374509. Selected POD ridge is 0.373656, a tiny validation-tuned
> difference, not confirmed test superiority. Next: nonlinear readout on identical POD bases.

> 2026-09-24 stability update: 40/40 n=128 zero-init runs completed;
> PUBLIC_STABILITY_V1.md is the latest evidence. Cosine decay achieves stable final-tail
> projection fitting in 20/20 runs; constant LR achieves 0/10 (batch32), 1/10 (batch128).
> Validation does not generally improve, including RMSE: convergence is not generalization.
> Next: matched stable-training controls on 1000 samples before larger architectures.

> 2026-09-24 smallfit update: 60/60 runs complete; see PUBLIC_SMALLFIT_V1.md.
> Zero-residual initialization improves final validation in 29/30 matched pairs and all
> six group means. At n=32 all ten zero-initialized models reach their own projection-fit
> threshold; n=128 and n=1000 do not reliably do so. All group validation means still
> trail the initial-mean baseline. Keep zero initialization, then test optimization
> stability; these are pretrained-trunk diagnostics, not few-shot generalization claims.

> 2026-09-23 fixed-trunk update: all 40 predeclared conditioning / mean-lift runs
> completed. Whitening gives small average paired validation gains, not a successful
> performance repair: every arm remains worse than the initial-mean baseline. See
> PUBLIC_COEFFICIENTS_V1.md. Prioritize small-sample branch fit and zero-residual
> initialization checks; conditioning alone is not supported as the primary solution.

> Public-data update (2026-09-22): read `PUBLIC_BENCHMARK_V1.md` for the completed
> 15-run matched supervised pilot. Test mean relative L2: MLP DeepONet 0.39357,
> Fourier DeepONet 0.41231, FNO 0.22835; initial spatial-mean baseline 0.37577.
> Fourier has a better offline projection fit but not better learned predictions.
> The older Fourier recommendation below does not generalize to this public-data protocol.
> Prioritize controlled coefficient-learning / conditioning experiments, not model scaling.

> 2026-09-22 audit update: the legacy LB and optimization-gap claims below are not
> certified by their original implementation. Read `AUDIT_V2.md` first. Full-grid
> recomputation supports a large 2D projection component, but not pure causal attribution
> to architecture capacity. Historical p-values and variance-floor claims require regrouping.

**Authority.** The dated audit and public-data updates above override conflicting legacy claims
below and in the paper. The historical text of this document and `paper/main.md` superseded older entries in `README.md` and
`paper/short_paper.md`. Historical claims remain recorded in `docs/EXPERIMENT_LEDGER.md`, but
are not evidence for the current paper.

## Supported by the committed results

1. A shared-grid evaluator used with per-sample coordinates can silently broadcast a target from
   `(B, N)` to `(B, B, N)`. In this project it affected Burgers initial conditions,
   diffusion--reaction forcing, and Darcy coefficients. It can produce plausible losses and a
   coherent but false experimental story.
2. A supervised model's autograd PDE residual, compared with the trivial-solution residual, and
   explicit sampler shape assertions detect this failure mode. Finite-difference residuals of
   stored reference solutions do not detect it.
3. For a fixed trunk feature matrix, `LB = sqrt(1 - captured_energy)` is the projection-error
   lower bound for any linear DeepONet branch readout. Together with train/test error it is used
   here to separate representation, optimization, and generalization gaps.
4. In the corrected 1D Burgers experiments, representation-side alignment improves the mean in
   all ten reported cells, but is not the preferred practical intervention: random Fourier trunk
   features are cheaper and competitive or better.
5. In corrected 2D Burgers at width 32, the MLP baseline is representation-limited and collapses
   near the zero solution. Fourier features repair substantially more of this failure than the
   alignment loss. The analytic prior is not better than a random smooth target subspace.
6. The reported alignment mechanism is empirical: output-NTK effective rank increases and the
   measured loss sharpness decreases. It is evidence consistent with regularization, not a proof
   of a universal causal mechanism.

## Claims explicitly retracted or unsupported

- Trunk-basis collapse is the universal root cause of PI-DeepONet failure.
- Subspace alignment is necessary, optimal, or complementary to Fourier features.
- Alignment gain follows a linear law determined by baseline cross-seed standard deviation.
- Any single-seed 20k/50k result establishes a statistically reliable method ranking.

## Scope and next falsification tests

The evidence currently covers synthetic Burgers (1D/2D), diffusion--reaction, and Darcy
implementations in this repository. It does not establish transfer to public benchmarks, FNO/PINO
architectures, real-world problems, or high-accuracy regimes. The next protocol therefore tests:

1. the sampler-health checks on every in-repository PDE;
2. the lower-bound attribution on a non-DeepONet architecture; and
3. a public PDEBench dataset using a resumable, manifest-driven run layout.
