# Experiment ledger

2026-10-10 PERIODIC_CURVATURE_V1:48 fits complete, source seal0f2c912;
83 tests pass85.464s; audit48 prediction/observation/RNG/fork/summary replays and
18 historical exact tensor matches pass. Allen periodic/none field.037159,
curvature.336740 vs linear/none.043007/3.694110; one field seed worsens.
Extra smoothing not field-beneficial on periodic Allen; Burgers linear/low best
field. No fresh confirmation or extra module. Training segment sum921.28s,
recorded final-segment CUDA peaks95.94..98.97MiB; no inference cost benchmark.


2026-10-09 PHYSICS_DIAGNOSIS_V1/V2: source seal8d2d62e, scalar-only repaircf149dc;
36 frozen models diagnosed, zero new fits. V1 preserves9 Allen records before
NumPyfloat32 JSON failure;v2 reproduces9 exactly, completes36. Diagnostic-only
adjacent frames on32 existingdev IDs,200-case Fourier bands/boundary metrics.
Allen curvature~3.7 driven by high frequencies/boundary concentration, not floor;
JVP implementation numerically consistent, learned u_t stillpoor. Burgers native
FD sensitivity confounds full-band space/PDE attribution.80 regression tests pass.
No confirmation IDs opened, no output filter selected or architecture changed.

2026-10-09 PHYSICS_STRESS_V1:48 fits complete (12 own-observation warm starts,
36 none/PDE/curvature forks), seal600273d; all48 independent replays passed.
Same34209-param FNO,2250 total steps,100 trajectories, fixed transferred0.3.
Burgers sparse0.644463/0.603849/0.589233; noisy dense0.273533/0.270520/0.267287;
noisy sparse0.648667/0.611887/0.599397 (none/PDE/curvature). Curvature wins all.
Fresh nonlinear synthetic Allen-Cahn0.043007/0.042646/0.043966, PDE gain0.839%
vs none; development-only, reserved500 not generated. Gradient/curvature QA
diagnostics reveal field/derivative mismatch. No new confirmation consumed.

2026-10-08 FNO_PHYSICS_RESOLUTION_V1: training-only100 trajectories, native1024/
512/256 and centraldt0.01/0.02; resolved0..16 gate passed (median1.8%-4.2%),
almost-full coarse residual25%-51%.20 diagnostic snapshots per training trajectory
(16 additional to the five fitting labels); no fresh confirmation or development
targets. Seal08ddfd2.
TIME_FNO_BASELINE_V1:3 seeds of continuous-time data-only FNO, same100x5 fitting
labels and paired historical batch stream,1500 steps; seal7e4aa56. Complete,
development0.277170 +/-0.006826,34209 parameters;3 checkpoint/batch replays pass.
FNO_REGULARIZATION_V1:30 registered paired2250-total-step models (1500 shared
data warm start +750 continuation), none/PDE/gradient/curvature on fixed0..16
band, train-gradient strengths0.03/0.1/0.3; COMPLETE,30 independent replays.
FNO_REGULARIZATION_CONFIRMATION_V1: NEW500 after seal4d60f18, disjoint8329
historical IDs;12 models/five registered contrasts replayed. None0.290253,
PDE0.286260,gradient0.286076,curvature0.282820; all three vs-none pass, both
physics-vs-smoothing superiority gates fail. New500 NOW CONSUMED. No retuning.

2026-10-08 FNO_PROJECTION_V1:36 paired fits, full/POD16/32/64,3 tasks,3 seeds;
source seal3cac70a. Same136481-parameter FNO encoder/pointwise head; paired initial states
and independently regenerated batch hashes. Full dev0.195846/0.220706/0.003315;
training-POD32 0.188328/0.216032/0.003653. Allen-Cahn post-hoc32 0.003552 is better;
post-hoc is stronger than training projection at every Allen-Cahn rank/seed. Shared readout
largely closes prior pooled-head gap; prior parameter/initialization differences remain.
Audit36 predictions/closures/groups,9 bitwise historical full reproductions;57 tests.
Tables:FNO_PROJECTION_V1_RESULTS.md; paper figure:fno_projection_20261008.png/.svg.
No new confirmation labels opened; no confirmed method novelty or efficiency claim.

2026-10-08 SPECTRAL_POD_V1:54 fits (3 tasks ×3 ranks ×2 encoders ×3 seeds), source seal
de5cc93, direct coefficients, matched nominal real parameter count per rank. Best spectral
0.273799/0.302838/0.006762; best local0.373109/0.380088/0.004215. Both Burgers tasks choose16;
Allen-Cahn chooses64. Old full-field FNO remains stronger on all tasks. All54 checkpoint
replays and exact squared Frobenius decompositions pass;55 tests; all54 skip on rerun.
Protocol/results:SPECTRAL_POD_V1.md / SPECTRAL_POD_V1_RESULTS.md. No new confirmation IDs.

2026-09-30 architecture stages:72 MLP fitting controls +18 CNN/POD +18 full-field FNO,
three seeds, three nonlinear tasks, development only. Selections sealed8192e38 before
architecture transfer. Direct FNO0.195846/0.220706/0.003315; post-hoc same-POD projection
0.192454/0.217435/0.026098. All108 checkpoints replayed;52 regression tests pass.
Protocol/results: ARCHITECTURE_STAGES_V1.md / ARCHITECTURE_STAGES_V1_RESULTS.md.
No fresh confirmation consumption; prior consumed IDs remain consumed. Prior continuous
and all-mode controls retained; no established method novelty or equal-compute claim.

> 2026-09-22 authority update: older rankings and statistical claims below are historical.
> AUDIT_V2.md supersedes their error-decomposition and provenance interpretation.
> New public-data evidence is indexed below and must not be pooled with legacy runs.

| New family | Protocol / evidence | Limits |
|---|---|---|
| `public_uncertainty_modal_v1` | 0 new neural fits;2000 conditional bootstrap draws/task; continuous uncertainty, trajectory ablation and correction-energy-matched global control; sealbfd52cd before4 new500-ID sets | No superiority over strong controls; independent Burgers0.6345% worse than old continuous. Allen-Cahn coefficients unchanged. Bootstrap not a calibrated guarantee. All2000 targets consumed; active plan_v2.json. |
| `public_field_modal_v1` | 11 new synthetic Allen-Cahn fits;33 old public fits reused; whole-field risk gating and4 fresh500-ID confirmations, sealed8a4020b | Linear-baseline gains3.0636%/3.1939%/21.6550% on nonlinear cases, none on Advection. No superiority over energy/old rules; slight Allen-Cahn harm. Synthetic second PDE, not public benchmark reproduction. All2000 new targets consumed. FIELD_MODAL_V1.md. |
| `public_incremental_modal_v1` | 33 neural fits; nested linear tuning; sealed candidates confirm3.0779%/2.8752% gains on two Burgers splits; Advection no-correction fallback | Selection sealedad94e69 before1500 new targets. No confirmed superiority over energy truncation. Finite shared-alpha grid, not optimal linear oracle. All three confirmation sets now consumed; INCREMENTAL_MODAL_V1.md. |
| `public_objective_modal_v1` | 123 completed neural records +48 ridge grid refits; normalized loss helps two Burgers development splits but hurts linear Advection; strict internal-CV mask comparisons | 25 redundant panel records; Burgers masks equal energy selection, Advection selects all16. Tuned Advection ridge beats residuals. Three new confirmation sets unopened. OBJECTIVE_MODAL_V1.md. |
| `audit_v2` | Full-grid checkpoint projection recomputation and regrouped historical statistics | Unknown historical source hashes remain unknown. |
| `public_fixed_t1_v1` | 15 completed matched supervised runs; 3 architectures, 5 seeds | One PDE/subset/time, equal updates not equal compute; DeepONet below initial-mean baseline. |
| `public_coefficients_v1` | All 40 runs completed; small average whitening gains, all arms below initial-mean baseline; PUBLIC_COEFFICIENTS_V1.md | Extra branch training and relaxed constant term; same exploratory test split, no final-holdout claim. |
| `public_smallfit_v1` | 60 completed runs; n=32 zero-init fits oracle in 10/10 seeds; initialization improves validation in 29/30 pairs | Validation only, pretrained trunk/normalization; not few-shot. Larger subsets not sufficiently fit; no baseline superiority. |
| `public_stability_v1` | 40 completed n=128 zero-init runs; cosine decay stable-tail fit in 20/20, no general validation improvement | Equal updates not equal samples/compute; pretrained trunk, validation only. Smallfit inability to fit n=128 was optimizer/budget dependent. |
| `public_scale1000_v1` | 20 completed controls; cosine improves validation relative L2 in 10/10 pairs | Includes 10 old-control repeats; training still not fully fit; no new test evaluation. |
| `public_pod_ridge_v1` | 18 deterministic training-only residual-POD ridge candidates; selected validation error 0.373656 | Validation-tuned grid boundary, tiny baseline gain; not full POD-DeepONet or two-step reference. |
| `public_pod_readout_v1` | 30 completed same-POD neural runs; nonlinear improves training in 15/15 but worsens validation in 15/15 versus Adam affine control | Exact rank-matched POD/normalization; reused validation only; coefficient error is not uniquely optimization error. |
| `public_ridge_residual_v1` | 40 neural runs +8 ridge refits; sealed selection evaluated on500 new trajectories; primary gain not confirmed, RMSE improves5/5 | Baseline-inclusive early stopping is not an unseen-data guarantee; confirmation IDs are now consumed; no post-confirmation retuning. |

This is the permanent index of experiment families. Raw per-run JSON, logs, checkpoints, and
figures remain versioned under `results/`. A run is evidence only when its source fingerprint,
configuration, and status are recorded.

| Era / result family | Status | What it supports | What must not be reused |
|---|---|---|---|
| `expA`--`expJ`, early DR/Burgers alignment experiments | superseded | Reproduction of the original failure symptoms | Any pre-2026-09-16 representation or accuracy conclusion; these used the per-sample pairing bug. |
| `expK`--`expP`, corrected 1D controls | historical / partly superseded | Causal weighting and L-BFGS negative controls; early alignment/Fourier comparison | Single-seed rankings as statistical claims. |
| `expQ`, NTK checkpoints | active evidence | Paired output-NTK and sharpness mechanism measurements | A universal mechanism claim; the mechanism sample is small. |
| `expS`, `expT`, resolution/width sweeps | historical | Rank and resolution probes | The retracted rank-to-gain linear relation. |
| `expU`, early 2D | superseded | Development trace only | Incomplete seed groups or early 2D conclusions. |
| `expV`, five-seed phase diagram | authoritative | Main 1D and baseline/alignment 2D summaries in `results/final_stats.txt` | Uncorrected p-values as multiplicity-adjusted evidence. |
| `expW`, 2D intervention controls | authoritative | Fourier, random alignment, analytic alignment, orthogonalization, and IC-weight controls | Alignment-necessity claim. |
| `expDO*`, supervised references | active evidence | Budget-aware supervised reference curves | A claim of physics superiority: 50k physics comparisons have limited seeds. |
| `rep1d*`, `rep2d*` | active evidence | Bit-identical repeatability under a fixed code state | Historical code-state comparisons without matching hashes. |

## Statistical interpretation

- The ten-cell phase table has five seeds per cell. Eight uncorrected Welch tests are below 0.05;
  six survive Holm--Bonferroni correction. Both numbers must be reported together.
- The 20k/50k physics rows are exploratory unless and until multi-seed repeats are added.
- `results/phase_analysis.json` and `results/final_stats.txt` are the current summary artifacts;
  individual JSON files are the primary records.

## Change-control rule

Every new experiment directory must contain a `manifest.json` before training, append events to
`events.jsonl` or persist atomic `progress.json` plus checkpointed evaluation history while
running, and receive a terminal `completed`, `interrupted`, or `failed`
status. A result is never silently overwritten; resumed work uses the same manifest and skips
completed configuration keys.
# 2026-10-08 spatial latent representation study

- Family results/manifold_reconstruction_v1, protocol seal f1939ea,81 completed fits.
- All final checkpoint outputs replayed; POD9 controls, seed batch regeneration,
  local/permuted-local parameter/initialization pairing and Parseval closure passed.
- Reused development; local helps over tested global/permuted controls, does not
  uniformly beat POD. Allen-Cahn field/curvature trade-off prevents physics claims.
- Registered9 fixed-time CNN DeepONet fits e1402c3 and18 matched attention/gate fits
  0072261; their outcomes pending at this entry. Stages4/5 not yet executed.
# 2026-10-08 final spatial latent/token/physics first-round evidence

- 126 new fits completed/audited:81 representation +9CNN/trunk +18token +12physics +6FNO100.
- Seal05b8153 before fresh500 Burgers labels; all500 now consumed, zero recorded
  historical-ID overlap.18 current100-shot confirmation models and4 adjusted decisions replayed.
- Confirm gate0.486568,gateP0.458865,attention0.470762,attentionP0.442270;
  same100-shot FNO16/32 0.313237/0.318333. Four registered incremental contrasts pass;
  hybrid competitive superiority fails descriptively versus near-size FNO16.
- Scalar-budget/POD/derivative limits, output-only512 transfer and one-PDE physics
  evidence recorded. Training seconds not controlled; quiet inference/allocated-memory
  profiles and visually verified figures archived.68 regression tests passed.
