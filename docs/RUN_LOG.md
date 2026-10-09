# Run log

## 2026-10-10 Frozen temporal diagnosis complete

- ActiveV2 run exited0 with36 frozen model rows. V3 numerical audit exited0 with
  36 direct alternate-AD primary replays, exact18 pilot reproductions, strict
  parent field replay and independent stored-summary/screen aggregation.
  Artifact results/temporal_after_periodic_v2/audit_v3.json; old failed v2 raw
  precision criterion remains a failure. Full87 tests plus appended5 targeted
  tests4.424s pass. No numerical/model-selection criterion silently weakened.
- Allen periodic/none resolvedut.289333 vs linear.324480 (-10.83%), late.5
  .716034. TrueutRMS.6890, absolute errorRMS.4142 at.5, floors inactive.
  Time/state-RHS common energies.200847/.573608; dominance only1/3 seeds,
  registered all3 screen FAILS. Both temporal and state errors matter.
- Burgers native FD time uncertainty12.34% low/44.40% full and negative residual
  cross term prevent exact temporal-causal claims; keep old linear/low control.
- Decision: do NOT immediately add global time loss; next fixed existing-dev
  midpoint/short-interval diagnostic, then a matched temporal/integral-loss pilot
  only if justified. No new observation/ID/time access or training this turn.
- Sync full12group table, failed hypothesis, reference/precision limits and
  next-step decision to results report, paper, current assessment and ledger.
- Repeat SAME activeV2 run command: exit0, zero new model computations, summary
  SHA unchanged. Resumable skip verified;36 audited/18 repair matches confirmed.

## 2026-10-10 Frozen temporal diagnosis after periodic artifact control

- User requests time-bottleneck diagnosis before deciding on a temporal loss;
  fixed36 final models, no training/modules/new observations/confirmation.
- Register TEMPORAL_AFTER_PERIODIC_V1.md: existing32 dev IDs/native cache hashes,
  Allen semidiscrete RHS versus FD references; Burgers FD width sensitivity;
  native-coefficient low modes (not sampled-then-projected). All arms retained.
- Resolved time/state-RHS errors, signed cross term and oracle residuals use a
  common reference norm; descriptive3-seed gate, not causal allocation.
- Independent scientific review agrees with identity/reference design; highlights
  observed-time slopes do not establish between-label interpolation consistency.
- Three targeted tests passed10.675s; full regression86 tests passed86.383s.
  New tests cover native projection versus sampled aliasing, Nyquist weighting,
  analytic temporal/RHS oracle identity and CPU/CUDA alternate-AD agreement.
- Development QA found float32 constant-offset rounding in JVP primal would
  differ~1e-8 from parent field evaluation; use original16-case/all5time forward
  batches and float64 rescaling for field replay. Derivative remains unchanged.
- Synthetic float32 tiny-output FD test initially suffered cancellation at h.002;
  widened its test-only h to.02. Formal trained-model probes stay.002/.001/.0005.
  Attempted whole-model float64 conversion is invalid for complex FNO weights;
  discarded before sealing. No old source or historical result modified.
- V1 seal74a5bcd completed18 Allen records, stopped at first Burgers parent-field
  comparison: old trajectory_errors uses float32 target norm versus new float64
  diagnostic; maxdiff1.62573123e-8. Preserve source/plan/partial rows. NewV2 uses
  ORIGINAL trajectory_errors for parent gate, leaves all output metrics unchanged;
  require exact18-pilot reproduction. Add dedicated dtype distinction regression.
- V2 preflight plan had not been executed; preserve it as
  plan.preflight_without_metric_dependency.json, then add unchanged original
  physics_token_core.py metric dependency hash and register executable plan.
  No consumed plan changed or prior result deleted.
- V2 four targeted tests passed3.897s; complete87-test regression passed85.564s.
  V2 runner/auditor compile;11 source dependencies verified against registered
  plan before execution. No tolerance relaxed for parent metric precision repair.
- V2 diagnosis36 complete, all18 pilot metric dictionaries exactly reproduced.
- Original alternate-AD audit stopped on a near-zero element (1.1845e-7 absolute
  difference). Preserve failed v2 auditor. Source-hashed all36 precision probe:
  raw659/1179648 elements fail old absolute check; maxglobal relativeRMS1.3269e-7.
  ALL primary temporal/field scores pass old tolerance.28 own-FD case/time leaves
  across15 Burgers models differ when subtracting alternate AD from tiny FD error;
  worstdifference3.1824e-7. Aggregate own-FD scores pass; failures retained.
- Register separate v3 numerical QA policy AFTER precision failure/probe: global
  RMSrelative<=1e-6, peak/signalRMS<=1e-5, maxcase-timeRMSrelative<=1e-5.
  Not predictive screening, certified rounding bound or pre-outcome protocol.
  Primary temporal tolerances unchanged; own-FD scores replay against ORIGINAL
  reverse-AD reference, not silently redefined by alternate-AD precision.

## 2026-10-10 Periodic-curvature family complete and audited

- Background worker29572 completed all48 fits and exited; stderr empty.
- `D:/miniconda3/envs/pytorch/python.exe scripts/audit_periodic_curvature.py
  --require-complete` exited0:48 predictions/observation identities/RNG/forks/
  summary replays and18 historical linear tensor matches passed.
- Allen periodic/none improves mean field13.60%, curvature90.88%, removes boundary
  concentration; field improves in2/3 seeds only. Additional low/full smoothing
  worsens its field mean. Burgers old linear/low still best field, none meets its
  predeclared curvature screen. No fresh confirmation opened or rule exported.
- Sum saved training timings921.28s excluding eval/I/O; final-segment allocated
  CUDA peaks95.94..98.97MiB. No measured inference-time or end-to-end cost claim.
- Sync all12 final arm means, seam/frequency evidence, cost caveats and next
  dynamics diagnosis to PERIODIC_CURVATURE_V1_RESULTS.md, current conclusions,
  publication assessment, ledger and paper. Preserve all old/sealed results.

## 2026-10-10 Resume periodic-curvature task

- No existing training/test worker found. Prior interrupted regression session output
  unavailable; reran full regression rather than claiming an unobserved success.
- `D:/miniconda3/envs/pytorch/python.exe -m unittest discover -s tests -q`:
  all83 tests passed in85.464s. Protocol/source/plan committed before formal fitting.
- Training remains development-only; all48 fits and18 historical tensor matches
  must finish before reporting comparative conclusions. No confirmation opened.
- Source/protocol seal `0f2c912`. Hidden persistent worker PID29572 launched with
  `D:/miniconda3/envs/pytorch/python.exe scripts/run_periodic_curvature.py run`;
  stdout/stderr: results/periodic_curvature_v1/worker_20261010_001.*.log.
- Initial live check:2/48 fits completed (Burgers linear warm seeds0/1), no stderr.
  Seed0 saved1500-step checkpoint includes model, optimizer, step, random states,
  history and calibration; final trainable tensors exactly match historical
  time_fno_baseline_v1/s0. This is one control check, NOT the final18-control audit.
- Remaining fits run in background. Do not start a duplicate writer after sleep.
  After process termination, repeat the same run command (NO --resume flag).
  Final comparison/paper conclusions await48-run completion and full audit.

## 2026-10-09 Paired periodic-position and low/full curvature controls

- User authorizes controlled retraining, no added modules. Newperiodic_curvature_v1
  factorial2 encodings x3 smoothing choices x3 seeds x2 dense-clean PDE tasks;
  12 own-encoding warm starts +36 shared-state final forks,48 fits planned.
- Replace only scalar coordinate with grid-statistic-matched periodic sine;
  same3 lift channels and34209 real parameters, trainable initial tensors paired.
  Single feature is not injective fullsin/cos or a hard periodic-output guarantee.
- None/low0..16/full0..128 curvature at TRAIN-only calibrated gradient ratio0.3,
  fixed1500+750 budgets/paired label/query streams. Full Nyquist energy correct.
  Hist linear controls must reproduce18 previous corresponding state tensors.
- Three new tests passed17.567s: state/capacity/linear equivalence, periodic scalar,
  exact historical low penalty, full Parseval/Nyquist, CPU/CUDA exact resume,
  calibration/pairing and delegated-core global restoration. No old source edits.

## 2026-10-09 Frequency artifacts, time derivative and discretization diagnosis

- User requests diagnosis only, no extra network modules. Newphysics_diagnosis_v1
  freezes all36 final stress models; no new training, architecture or confirmation.
- Register actual Fourier error bands0..4/5..16/17..64/65..128, Nyquist-correct
  Parseval closure, boundary8 energy concentration and denominator-floor activity.
  Fixed cutoff16/32/64 are counterfactual diagnostics, not selected output models.
- Temporal checks: JVP versus own FD0.001/0.0005 and numerical native truth on
  fixed first32 EXISTING development IDs. Discretization grids1024/512/256,
  native FD0.01/0.02 Burgers and0.0005/0.001 Allen-Cahn; residual band16/full.
  Compare raw/filtered coarse targets and exact nonlinear-closure coefficient gap.
- Extra adjacent development snapshots, including immediately after final label,
  explicitly disclosed and diagnostic-only. Not unseen trajectory validation,
  extrapolation success, additional training labels or prior500 reuse.
- Two new analytic diagnostics tests passed0.172s; source/rules committed before
  execution. Old checkpoints/results/seals remain unchanged.
- V1 seal8d2d62e;9 Allen-Cahn records persisted before float32 NumPy floor scalar
  caused JSON serialization failure on first Burgers result. No training/model
  affected. Preserve sealedv1 and partial artifacts; newv2 scalar-only adapter
  delegates unchanged numerics with new output/cache roots and source fingerprints.
  Add float32 JSON compatibility regression; do not modify sealed numerical files.
- JSON-onlyv2 sealcf149dc;36 final model diagnostics complete, ninev1 Allen records
  reproduce exactly. Audit36 source/model/output identities, energy shares and
  native FD sensitivity passed; no training or fresh confirmation.
- Allen data-only curvature3.6941:90.17% error energy in modes65..128,98.25% at
  first/last4 grid nodes, denominator floor never active. Fixed cutoff16 diagnostic
  reduces curvature0.3250 with field0.04272 vs0.04301; NOT selected validation gain.
  Linear coordinate/local paths are candidate causes, not proved by this diagnosis.
- Model JVP/own FD mean differences5e-5..2.3e-4, but Allen time derivative error
  0.3252/0.3219/0.3274 (none/PDE/curvature); native reference RHS disagreement1.17e-6.
- Burgers native FD h0.02 versus0.01 disagreement44.40% full,12.34% resolved;
  do not call full prediction-vs-FD~95%-99% a certified physical derivative error.
  Large full-band residual even at1024 confounds temporal and spatial effects.
- New evidence changes prior interpretation: coarse full residual alone does NOT
  identify spatial inconsistency; resolved nonlinear closure RMS0.03739 vs native
  FD residual0.94883 is not dominant here. Allen reference residual~1e-6 on all
  grids rules out target inconsistency as an explanation for its seam curvature.
- Final full regression80 tests passed55.497s. Diagnostic artifact/source/model/
  share invariants and nine unchanged scalar-repair records audited; not claimed
  as a second independent replay of all36 temporal predictions. No background
  diagnostic/training worker remains; numerical reference and all final checkpoints
  retained. Paper/current assessment updated to distinguish temporal/spatial effects.

## 2026-10-08 Fixed-architecture sparse-time/noisy-label and second nonlinear PDE

- User authorized fixed architecture, strong curvature baseline, sparse-time/noise
  and a second nonlinear PDE. New familyphysics_stress_v1; do not edit old seals.
- Register48 fits:3 Burgers conditions x12 fits, then Allen-Cahn dense-clean12.
  Each condition own1500-step OBSERVATION-only warm start, none/PDE/curvature
  paired750-step forks; transfer strength0.3 without per-condition retuning.
- Sparse labels0/1 only; withheld0.1/0.25/0.5 primary. Noise only future labels,
  fixed10% observed-label RMS. Inputs/IC clean; omitted labels cannot affect
  corruption amplitude or fitting statistics (unit-tested). No dense warm leakage.
- Nonlinear Allen-Cahn fresh train/dev IDs200000..200299, reserved500 not generated;
  period2 and reaction5, dealiased cube. Dedicated numerical QA passes time/space/
  independent DOP853/residual checks; existing synthetic generator reused with
  new trajectories, no public dataset download or already-proven physics success.
- Three new tests passed15.288s, covering analytic PDE signs/period/dealiasing,
  observation/noise isolation, constant trajectories, CPU/CUDA exact warm/fork
  resume and calibration pairing for both PDEs. Old confirmation sets consumed.
- 2026-10-09: full regression77 tests passed48.218s before registered training.
  Launch48-fit worker after committing source, immutable plan and solver QA;
  hidden background worker writes persistent stdout/stderr, all models atomic
  full checkpoints every50 steps. Resume same run command, one writer only.
- Source seal600273d; worker40956 launched hidden, stdout/stderr recorded in new
  result family.48/48 finished without errors; process check confirms no writer.
- Independent audit48/48 verifies final weights/identities, development predictions,
  observation/noise isolation, regenerated labeled/query streams, paired warm
  states/calibrations and summary means. Added exploratory full-grid spectral
  gradient/curvature diagnostics with train INITIAL-only denominator floor.
- Burgers sparse none/PDE/curvature0.644463/0.603849/0.589233; noisy dense
  0.273533/0.270520/0.267287; noisy sparse0.648667/0.611887/0.599397.
  PDE helps none but never beats curvature in these transferred-strength tests.
- Allen-Cahn new nonlinear dense-clean0.043007/0.042646/0.043966. PDE improves
  both controls in all3 paired seeds but gain vs-none only0.839% on development.
  No fresh confirmation opened; reserved synthetic500 not generated; prior two
  Burgers500 sets remain consumed. Full-grid Allen curvature errors~3.7 warn
  against field-only physical-fidelity claims. No posthoc strength search.
- Next proposed work: develop frequency/time-derivative diagnosis before spending
  further confirmation; stronger smoothing/physics scale controls remain required.

## 2026-10-08 Strong FNO physics prerequisite (new versioned families)

- After user interruption, process check found no remaining training worker;
  saved progress and summary show30/30 complete. No duplicate training started.
  Complete independent replay:30 checkpoints, predictions, batches, collocations,
  same starts/calibrations and family-selection rule all passed.
- Development none0.270949, physics0.267574, gradient0.267647, curvature0.265248;
  all selected strengths0.3 (upper searched boundary). Relative improvements
  1.246%/1.219%/2.104%. Physics does NOT show an advantage over smoothing here.
- Archive commitc392f39 preserves all30 fits. A NEW confirmation registry excludes
  conservative8329 historical IDs, explicitly including previous consumed500.
  New500 selected from1671 available, seed2026100808;12 frozen models, five
  predeclared contrasts,10000 paired trajectory bootstrap, Bonferroni5 intervals,
  >=1%gain AND upper<0. No new target extraction until seal committed.
- Confirmation effect-gate/reproducibility/pairing unit test passed0.036s.
- Seal4d60f18 committed before fresh500 extraction. New500 disjoint from8329
  conservative history (previous500 included); NOW CONSUMED from first extraction.
- Confirm none0.290253, PDE0.286260, gradient0.286076, curvature0.282820;
  gains1.376%/1.439%/2.561%. Five predeclared Bonferroni5 bootstrap contrasts:
  all three vs-none pass; PDE-vs-gradient fails (interval contains zero);
  PDE-vs-curvature fails (PDE worse, interval wholly positive).
- Independent confirmation audit replays all12 predictions and all five decisions.
  No post-confirmation weight search/model selection. Follow-up priority changes
  to retain curvature as the strong control and examine sparse-time/noise settings,
  then a second nonlinear PDE. These are proposals, not new demonstrated results.
- Final full regression74 tests passed40.445s after confirmation; no training or
  confirmation worker remains. Paper/main/follow-up, current conclusions,
  publication assessment, ledger and runbook synchronized to the negative
  physics-specific result. All binary warm/continuation checkpoints remain local.

- User: continue. Preserve previous126 fits and consumed500 confirmation IDs.
- Added continuous-time FNO, query-time JVP and periodic spectral conservative
  residual; do not infer temporal derivatives from historical five output heads.
- Source/gate seal08ddfd2 before training-only resolution audit.100 existing
  training IDs,20 diagnostic time frames (16 additional to fitting labels), no development or
  confirmation extraction. Diffusion0.001/pi, exact retained cell centers.
- Registered resolved modes0..16 passed feasibility gate: median relative
  residual0.042138/0.036492/0.031825/0.018282 at four times; almost-full spectrum
  medians0.507245/0.511209/0.421037/0.254306. This is scale-limited evidence;
  full-frequency residual enforcement is not supported by this diagnostic.
- Three derivative/dealiasing/JVP tests passed4.058s, including CPU/CUDA parameter
  gradients; one additional CPU/CUDA exact-resume/identity test passed3.714s.
- Seal7e4aa56 before time_fno_baseline_v1 training:3 seeds, same100 trajectories,
  same5 labels per batch,1500 fixed steps, atomic full checkpoints every50.
  Development continuation gate<=1.10*historical FNO16 mean0.2871897, fixed before
  outcomes. No physical loss yet; training-only PDE probes descriptive.
- Commands: pytorch python -m unittest tests.test_time_fno -q;
  -m unittest tests.test_time_fno_baseline -q;
  scripts/audit_fno_physics_resolution.py plan/run;
  scripts/run_time_fno_baseline.py plan/run. Re-run the last command to resume;
  completed identities/checkpoint hashes are checked before skipping.
- Time FNO3/3 complete: dev0.277170 +/-seedSD0.006826, per-seed
  0.269737/0.283157/0.278617,34209 real parameters. Gate passed; no physics used.
  Independent audit3/3 predictions and regenerated rolling batches passed;
  batches equal historical matched100-shot FNO16. Source seal7e4aa56.
- Register fno_regularization_v1:30 paired750-step continuations of frozen1500
  warm starts; data-only versus resolved PDE/gradient/curvature strengths
  0.03/0.1/0.3, training-only gradient-norm calibration. Common optimizer/labels/
  query times/LR restart; all penalty forwards, different backward cost. Fixed
  final checkpoints only; per-family mean-of3 development selection, all grid
  entries preserved. No fresh confirmation or second-PDE claim.
- New continuation test passed7.296s: CPU/CUDA bitwise resume for all4 families,
  calibrated initial gradient ratios and paired starts/batches/collocations.
- Full regression73 tests passed40.976s before continuation training. Resume
  command: pytorch python scripts/run_fno_regularization.py run. No --resume flag.

## 2026-10-08 Shared FNO pointwise head and output projection

- Added FNO_PROJECTION_V1 protocol:36 paired fits, full/POD16/32/64,3 tasks,3 seeds; identical136481-parameter encoder/head, optimizer, initialization and batches. No new confirmation data.
- Coefficient loss equals projected-field squared loss up to a fixed target projection constant. Raw projected-arm fields outside the POD subspace are diagnostic only. Saved initialization and rolling batch hashes make pairing auditable.
- Preflight: gradient/subspace equivalence, CPU/CUDA exact resume, matching initialization/batch hashes, full-arm bitwise equivalence to historical training core. Source and immutable plan to be committed before formal training.
- Source seal3cac70a; full regression57 tests passed (46.908s). Original Burgers12 fits completed/replay-audited. Full0.195846; training-POD16/32/64:0.191759/0.188328/0.190009; paired full-training post-hoc POD:0.192454/0.192368/0.194286. Historical full final tensors match bitwise for all3 seeds. Initialization and all1500 batch hashes agree across arms. Differences remain exploratory reused-development evidence.
- Independent Burgers12 fits completed/audit24 total. Full0.220706; training-POD16/32/64:0.218613/0.216032/0.215583; post-hoc0.217435/0.218254/0.218886. Rank16 training projection slightly worsens mean post-hoc comparison;32 mixed seed signs;64 improves all3 seeds. All six full-arm checkpoints match historical tensors bitwise. Allen-Cahn12 fits remain in progress.
- Interim30/36 completed: Allen-Cahn full0.003315 ±0.000286, POD16 0.026170 ±0.000049. Archived these6 completed runs before high-rank completion. Full/POD32/POD64 comparisons and final audit pending; no protocol changes.
- Completed36/36; Allen-Cahn POD32 0.003653 ±0.000202, POD64 0.003207 ±0.000226. Same-space full-training post-hoc16/32/64 0.026098/0.003552/0.003182 is better in every paired seed. Training projection therefore has no extra gain here, even where it beats unprojected full output.
- Final `scripts/audit_fno_projection.py --require-complete`:36 checkpoint hashes/identities/predictions/projection closures/groups verified, independently regenerated every batch fingerprint, same initial states and136481 parameters. All9 full-arm checkpoints match historical tensors bitwise. Re-running the driver skips all36 and preserves summaries;57 full regression tests previously passed46.908s.
- Generated paper figure with standard plotting (`scripts/plot_fno_projection.py`); PNG and SVG visually inspected via PNG render. Shading is seed SD, oracle is label-dependent, all curves use reused development data. Full generated Markdown report matches saved/audited summaries. Updated manuscript/conclusions/ledger/publication assessment.
- No new confirmation labels opened; all old consumed IDs remain consumed. Full optimizer/RNG checkpoints remain local/Git-ignored. New code/immutable plan/JSON histories/figures/docs are versioned. Next proposed action: seal full/post-hoc/training-projection controls and assess genuinely unused confirmation data; not executed this round.

## 2026-10-08 Spectral/POD matched-parameter study

- New54-fit development-only protocol in SPECTRAL_POD_V1.md. Historical sources/results unchanged. Two encoders share heads and exactly match nominal real parameter counts; rank16/32/64 basis uses representation400 only, correction600 fitting,dev200; three seeds.
- CPU/CUDA resume, zero head, parameter matching and nested-basis tests passed (`-m unittest tests.test_spectral_pod -q`,3 tests). Real-data basis preflight succeeded for all3 tasks and ranks, preserving historical q16. No confirmation data loaded.
- Source/plan committed before outcomes; run command `D:/miniconda3/envs/pytorch/python.exe scripts/run_spectral_pod.py run`. Checkpoints every50 steps; all54 fits and rank selections will be archived. Selection is development-adaptive, not unseen validation.
- Source seal de5cc93. Full regression55 tests passed (37.685s). Original Burgers18 fits completed/replay-audited: spectral ranks16/32/64 dev0.273799/0.288870/0.296425; local matched0.373109/0.393550/0.398890. Both select16. Nominal real parameters147696/160000/184608. Increasing rank improves training/projection floor but worsens final development error; all tested spectral models remain weaker than old direct full FNO0.195846.
- Independent Burgers18 fits completed; checkpoint audit36 total. Spectral ranks16/32/64 dev0.302838/0.318006/0.322749; local matched0.380088/0.391903/0.394451. Both select16. Old full FNO0.220706 remains stronger. Allen-Cahn18 fits now running, with no protocol changes or new confirmation opening.
- Interim48/54 completed: Allen-Cahn rank16 spectral0.027097/local0.026244; rank32 spectral0.006965/local0.004486. Archived these12 completed runs before rank64 finishes. Their final replay audit awaits completed task summary. Local control is stronger here, unlike Burgers; no universal spectral superiority claim.
- Completed54/54. Allen-Cahn rank64 spectral0.006762 ±0.000606/local0.004215 ±0.000146; both select64. No configuration beats historical full-field FNO0.003315 on Allen-Cahn or its Burgers scores. Full tables in SPECTRAL_POD_V1_RESULTS.md; paper and conclusions updated without changing historical results.
- `scripts/audit_spectral_pod.py --require-complete` replayed all54 final checkpoints, identities, exact anchors, predictions, counts, group metrics and selections; exact normalized squared Frobenius closure checked. Added per-seed development errors and decomposition to audit.json. Diagnostic percentages are not percentages of mean relativeL2.
- Re-running `scripts/run_spectral_pod.py run` skipped all54 fits and reproduced summaries. CPU/CUDA exact-resume tests and full55-test regression verified. No new confirmation read; all prior consumed IDs remain consumed. Checkpoints remain local and Git-ignored; all source/plans/history/JSON evidence are versioned.
- Next proposed (not executed): same FNO backbone and shared pointwise readout, full output versus projected output during training. Test the current pooled coefficient-head confound before any larger architecture search. No method novelty or fresh-validation success claimed.

## 2026-09-22 — second audit and actual recovery implementation

- Completed `python scripts/probe_coefficients.py`: four fixed-trunk supervised pilot runs,
  1000 steps each. Results and limitations recorded in AUDIT_V2.md.
- Ran `python tests/test_fno_resume.py`: CPU FNO final tensors were bit-identical after
  simulated interruption and restart versus continuous training.
- Started pinned PDEBench download using `python -u -B scripts/download_pdebench.py`.
  Expected 8,232,968,312 bytes; revision and publisher SHA-256 embedded in the script.
  Live state: external_data/pdebench/download_status.json (local, ignored by Git).
  Resume by rerunning that command after interruption. Completion requires checksum validation.

- Ran `python tests/test_projection_audit.py`: three tests passed (known omitted-mode
  counterexample, bias/Pythagoras, rank deficiency).
- Ran `python scripts/audit_checkpoints.py`: 35 checkpoints completed, each independently
  persisted with data/source provenance. All three data splits evaluated on full grids.
- Ran `python scripts/audit_statistics.py`: 13 configuration groups, 11 exploratory
  comparisons, one conflicting repeat quarantined. Unknown source hashes remain unknown.
- Ran `python tests/test_resume.py`: forced CPU Adam interruption/resume matched uninterrupted
  parameters and histories bit for bit. See AUDIT_V2.md for limits and measured results.
- Corrected legacy claims: original LB proxies are not certified bounds; supervised value
  fit does not guarantee small derivative residual. The old phase aggregation CLI is retired.

This append-only log records work performed after Git adoption. Raw manifests and event streams
for future long runs live under `results/runs/<family>/`.

## 2026-09-22 — repository stabilization and validation extension

- Created Git repository and committed the pre-existing project baseline as `b31d444`.
- Verified the requested conda interpreter:
  `D:\\miniconda3\\envs\\pytorch\\python.exe`, Python 3.11.16, PyTorch 2.14.0+cu130,
  CUDA available, NumPy 2.4.6, SciPy 1.17.1.
- Added `h5py 3.16.0` to support official PDEBench HDF5 files; exact environment packages are
  frozen in `requirements-pytorch.lock`.
- Ran `tests/test_sampler_contracts.py`: passed for Burgers, diffusion--reaction, and Darcy.
  The test compares batched paired evaluation to an independent one-sample-at-a-time oracle and
  rejects the historical `(B, B, N)` broadcast shape.
- Ran a synthetic-HDF5 end-to-end FNO smoke test: dataset discovery, two optimizer steps,
  checkpoint serialization, and JSON result serialization all completed. Synthetic artifacts are
  deliberately local/ignored and are not scientific evidence.
- Ran the existing `scripts/check_residual.py` in the CUDA environment. The process completed;
  rerun it before a paper release through a logged experiment family if its numeric residual
  output is needed as a fresh evidence artifact.

## Next queued run (not started)

The public PDEBench 1D Burgers file has not been downloaded automatically because the official
training file is multi-GB. Once placed at the path in `docs/PDEBENCH_PROTOCOL.md`, execute the
committed `experiments/pdebench_burgers_fno_smoke.json` plan with `--resume`. The plan writes a
manifest before launching and the FNO job checkpoints every 25 steps.

## 2026-09-22 — downloaded public data and matched protocol

- Supersedes the queued-run note above: the public file is present. Independently verified
  SHA-256 `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d`.
- Inspected the actual HDF5 and official generator: 10000 x 201 x 1024 solutions,
  with one unused final time-coordinate entry; the generator uses epsilon/pi diffusion.
- Executed `scripts/prepare_public_benchmark.py`: immutable fixed-t=1 subset and train-only
  normalization persisted in `external_data/pdebench/fixed_t1_v1`.
- `tests/test_public_benchmark.py` passed (2 tests). `tests/test_matched_public_resume.py`
  passed for both DeepONet and FNO: forced CPU interruption restores bitwise-identical
  final model tensors and metrics, and changed configurations are rejected.
- Added matched supervised three-architecture/five-seed protocol, actual cell-centered FNO
  coordinates, source/data/environment identities, atomic checkpoints every 50 steps,
  progress JSON and append-only events. See PUBLIC_BENCHMARK_V1.md for the locked protocol.
- Formal CUDA command queued after implementation commit:
  `D:/miniconda3/envs/pytorch/python.exe -u -B scripts/run_public_benchmark.py`.
  No public-data model performance claims are made before that run completes.

## 2026-09-22 — completed public CUDA pilot

- Executed the formal command above at source commit `4a11b63`; 15 / 15 runs completed.
  Every model trained 1500 updates; each has identity, atomic status, event log and local
  resumable checkpoint. Historical results were not overwritten.
- Added and executed `scripts/summarize_public_benchmark.py`; validates common source,
  environment, data and matched configurations before aggregating all five seeds.
- Test mean relative L2 (mean +/- sample standard deviation): MLP 0.39357 +/- 0.01014,
  Fourier 0.41231 +/- 0.01344, FNO 0.22835 +/- 0.00680. Initial spatial-mean baseline
  is 0.37577. Report negative DeepONet baseline comparisons explicitly.
- Fourier offline projection error 0.04770 (0.05937 at rcond 1e-6) contrasts with its
  actual 0.41231; next hypothesis concerns coefficient learning / conditioning, not an
  established pure-optimization cause. Full limitations and next controls are in
  PUBLIC_BENCHMARK_V1.md. Updated CURRENT_CONCLUSIONS.md to prevent stale extrapolation.
- `python -B -m unittest discover -s tests -p "test*resume.py"`: 3 tests passed.
- `PUBLIC_TEST_DEVICE=cuda python -B tests/test_matched_public_resume.py`: both model
  subcases passed bitwise parameter and exact metric equality after forced interruption.
- Re-executed the formal command: all 15 completed identities checked and skipped.
  No experiment was retrained. Source files used by training were unchanged during the run.
- This is a supervised, fixed-time, single-split pilot, not a physics-informed or official
  full-dataset benchmark claim. Reserve unused trajectory IDs for final confirmatory tests.

## 2026-09-22 — fixed-trunk factorial protocol locked

- Added PUBLIC_COEFFICIENTS_V1.md and a separate resumable runner; 2 trunk families x
  5 seeds x conditioning/mean-lift factorial = 40 runs, predeclared 1500/3000 budgets.
- Both conditioning arms retain exactly the same SVD subspace at rcond 1e-6, 65 branch
  outputs, unit element RMS, common branch initialization and minibatch sequence.
- New runner checkpoints parent/source/data/environment identity, model/optimizer/RNG,
  basis, history, cumulative timing and milestone results every 50 steps.
- Tests added for equal span/scale and forced interruption plus identity rejection.
  Actual test/run outcomes will be appended after execution.

- Fixed-trunk CPU and CUDA tests both passed (2 tests per device, including two resume
  subcases). Full unittest discovery passed 10 tests. No claim of physical power-loss testing.
- Formal 40-run CUDA command started from commit `f3d7685`; checkpoints and results are
  written incrementally. Training source is frozen while running.

## 2026-09-23 — fixed-trunk factorial completed and audited

- On continuation, inspected all progress files and active Python processes: all 40 runs
  had completed 3000 steps; no training process remained. No restart was necessary.
- Executed `python -B scripts/summarize_public_coefficients.py`: all common identities,
  matched parent hashes and configurations validated; recorded eight groups, both budget
  milestones and paired effects. Generated and visually inspected validation_curves.png.
- Executed `python -u -B scripts/run_public_coefficients.py` again: all 40 were skipped
  after identity checks, with no repeated updates.
- Whitening alone reduced mean validation error by 0.00626 (MLP) and 0.01831 (Fourier),
  with 4/5 seeds improved in each. Mean-lift effects were mixed. All eight group test
  means remain above the 0.37577 no-training baseline; no practical improvement claim.
- Every group mean validation error improved between 1500 and 3000 updates. This supports
  budget sensitivity, not eventual attainment of the offline projection oracle.
- Preserved all negative results and historical families. Updated current conclusions and
  ledger; next pre-experiment checks should target small-sample branch fitting and residual
  initialization before adding architecture or physics complexity.

## 2026-09-23 — small-subset / initialization protocol

- Locked PUBLIC_SMALLFIT_V1.md: two trunks, five seeds, nested n=32/128/1000 subsets,
  random versus zero residual initialization; 60 runs, 3000 updates, validation only.
- Reuse whitened bases and input-mean offsets; compare actual subset fit with its own
  affine projection oracle. Predeclare excess relative Frobenius <= 0.01 fit criterion.
- Explicitly not a few-shot benchmark: pretrained trunks and normalization already saw
  original 1000 training samples. No new final holdout is consumed.
- Added dedicated runner and tests for subset selection, exact zero-init baseline,
  forced resume equality and configuration rejection. Verification outcomes follow.

## 2026-09-24 — resume the interrupted smallfit study

- Inspected process command lines and all saved progress: 37 completed runs, one stopped
  at step 1350, no active smallfit training process. Other Python services were untouched.
- Restarted the identical `python -u -B scripts/run_public_smallfit.py` command, preserving
  source commit `05f5879`, identities, subset rows and hyperparameters; completed runs skip.
- Before the original launch, CPU and CUDA smallfit tests passed, including exact initial
  mean-baseline predictions, forced resume equality, subset identity and config rejection.
- Full regression suite and final evidence aggregation are being verified on continuation.

## 2026-09-24 — smallfit results finalized

- Actual events confirm resume at step 1350 for deeponet_fourier_s1_n32_z1. All 60 runs
  reached 3000 steps without modifying training sources or hyperparameters.
- `python -B -m unittest discover -s tests`: 11 tests passed.
- `python -B scripts/summarize_public_smallfit.py`: validated all identities, parent hashes,
  nested row indices and configurations. Saved paired summaries and inspected fit_curves.png.
- At n=32 zero initialization reaches excess relative Frobenius <=0.01 in 10/10 runs.
  At n=128 only 1/10, n=1000 0/10. Fourier random n=32 previously reached but did not retain
  the threshold in every seed; do not confuse final instability with inability to fit.
- Zero initialization lowers final validation mean-relative-L2 in 29/30 matched pairs;
  all six group averages and seed standard deviations are lower. No significance claim.
- Every group mean remains worse than validation initial-mean baseline 0.374509; the
  best group mean is MLP n=1000 zero init 0.389815. No test data evaluated this round.
- Updated conclusions and ledger. Next proposed, not executed: fixed small-subset learning-
  rate/batch stability controls, with MSE and relative-error metrics reported together.

## 2026-09-24 — stability factorial protocol locked

- User requested preserving zero residual initialization and testing decay / batch size.
- Locked 40-run n=128 protocol: 2 trunks x 5 seeds x constant/cosine LR x batch 32/128.
  Predeclared final and last-four-logged-point stability thresholds; no test evaluation.
- Explicitly report unequal sample exposure, plus descriptive full-batch 750 versus
  mini-batch 3000 comparison. Learning-rate position is checkpoint-step derived.
- Added independent runner and PUBLIC_STABILITY_V1.md; verification/run outcomes follow.

- CPU and CUDA stability tests passed all four schedule/batch resume subcases; endpoints,
  exact prior-control metrics and changed-config rejection also verified.
- Full unittest discovery passed 16 tests, including stability/rebound summary boundaries.
- Formal run launched from source commit `672c9d9`. Regression checks overlap part of the
  run, so update timings are operational measurements, not isolated performance benchmarks.

## 2026-09-24 — completed optimization stability controls

- All 40 runs reached 3000 steps. Summary validated identities and exact final metric
  reproduction for all 10 previous constant-LR batch32 controls.
- Cosine decay: 20/20 stable fits over steps 2250/2500/2750/3000. Constant LR: 0/10
  stable with batch32 and 1/10 with batch128. Every constant/full-batch run previously
  crossed the fit threshold and rebounded; final fit alone hides this instability.
- With batch32, decay worsened validation relative L2 and RMSE in 5/5 seeds for each
  trunk despite better training fits. MLP/full-batch decay improves validation relative
  to constant/full-batch; Fourier/full-batch does not. No overall performance claim.
- Saved same-sample-exposure descriptive comparisons and all raw paired differences;
  equal updates do not equal compute. No test data evaluated.
- Generated and inspected stability_curves.png. Re-executed original command: all 40
  identities checked and skipped, no repeated training. Results and next steps documented
  in PUBLIC_STABILITY_V1.md; historical families preserved.

## 2026-09-24 — manuscript synchronized with audited evidence

- User requested updating the paper plus publication-readiness and next-direction advice;
  no new experiments or external submission were authorized or performed this turn.
- Archived four pre-audit drafts under paper/archive/pre_audit_20260924. Rewrote main.md
  around the supervised public-data intervention evidence, and synchronized short Markdown
  and legacy-named LaTeX versions. Original data, checkpoints and figures were preserved.
- Removed active claims of pure optimization attribution, universal variance floor,
  confirmed causal mechanism and low-value-error guaranteeing low PDE residual. Removed
  unverified submission/template assertions; new public evidence is not physics-informed.
- Checked primary sources for Lee/Shin two-step training, Williams et al. basis diagnostics,
  Lu et al. POD comparisons, PDEBench, ReZero, SGDR and large-batch generalization work.
  References and novelty boundaries are documented in the manuscript and
  PUBLICATION_ASSESSMENT.md; this is targeted screening, not an exhaustive novelty review.
- Added evidence tests for numeric tables, run/control counts, citation keys, local links
  and basic TeX balance. No TeX compiler was detected; no PDF or page-count certification.
- `tests/test_paper_evidence.py`: 2 tests passed. Full unittest discovery: 18 tests passed.
  No training source or experiment result was changed by this manuscript revision.

## 2026-09-28 — larger-subset controls and POD ridge protocol

- Locked PUBLIC_SCALE1000_V1.md: 20 n=1000 zero-init/batch32 constant/cosine runs and
  an 18-case training-only residual-POD ridge grid. Validation selection; no test loading.
- Separate runners preserve previous source hashes and identities. Neural runs checkpoint
  every 50 updates; ridge saves each complete candidate and model before continuing.
- Tests added for closed-form ridge normal equations, basis orthogonality, nonmutating
  prediction, neural resume equality, config rejection and old constant-control parity.

- CPU and CUDA new tests passed (two tests per device). Full regression discovery passed
  20 tests. Formal source commit `5cb07f5`; 20 neural runs and 18 ridge cases completed.
- Summary validates all shared identities and ten exact old-control metric reproductions.
  Cosine improves validation relative L2 in 10/10 pairs; RMSE in 9/10. No full train fit.
- MLP cosine validation mean 0.374807 versus initial-mean baseline 0.374509; selected
  rank16/alpha10 ridge is 0.373656 after 18-candidate validation selection, not test evidence.
- Re-executed both runners: all completed items skipped after identity checks. Preserved
  local checkpoints and historical results. No test arrays were loaded.
- Added dated paper follow-up without replacing the core study's original tables; records
  now total 175 neural runs including 20 repeated controls, plus 18 deterministic ridge fits.
  Next not yet executed: nonlinear readout on the identical training-only POD basis.

## 2026-09-28 — same-POD readout protocol

- Locked PUBLIC_POD_READOUT_V1.md: ranks16/32/64, five seeds each, zero-output MLP
  and matched Adam affine control, 3000 cosine updates / batch32. Reuse exact stored
  ridge Q and per-position normalization; reproduce rank-matched ridge metrics first.
- New runner saves atomic full-state checkpoints every50 updates and evaluations every250;
  float64 reconstruction checks squared projection/coefficient error closure. No test loading.
- CPU and CUDA readout tests both passed (2/device): zero prediction, objective identity,
  validation-target isolation, uninterrupted/resumed bitwise weights and history equality,
  completed-run skip and changed-config rejection. Full regression:22 tests passed.
- Command: D:/miniconda3/envs/pytorch/python.exe scripts/run_public_pod_readout.py

- Formal source commit `5ad1676`; all30 runs completed on the RTX5060 Laptop GPU.
  Independent audit reproduces group statistics from per-run files, checks exact shared
  POD checkpoint hashes/oracles, final-step histories, completion manifests and protocol.
  Three selected ridge checkpoints reproduce their original train/validation metrics exactly.
- Audit initially rejected a 1.12e-12 historical-baseline difference: old spatial means
  were float32, new means float64. Adjusted only the audit tolerance to1e-10; no training
  code or results changed. Maximum squared projection closure error is1.32e-16.
- Nonlinear versus matched Adam affine: train L2/RMSE improves15/15, validation L2/RMSE
  worsens15/15. Final MLP beats ridge0/15, initial-mean baseline0/15; full projection fit0/15.
  Validation means at ranks16/32/64:0.398761/0.401114/0.402642. Coefficient mismatch explains
  about93–99% of squared validation error, not a certified pure optimization gap.
- Added paper/FOLLOWUP_POD_READOUT_20260928.md and linked the active manuscript. Historical
  core tables remain unchanged. Combined records:205 neural (including20 old repeats),
  plus18 deterministic ridge fits reused here. No new test arrays were loaded.
- This runner uses atomic progress plus checkpointed history rather than events.jsonl;
  ledger now explicitly documents both persisted logging formats. Local checkpoints remain
  ignored by Git; configuration, metrics, histories and provenance are versioned.
- Final regression:23 tests passed, including numeric report/paper tables, exact shared
  basis identities and squared-error closure. Re-ran the formal runner:all30 completed
  items skipped after identity checks; no training was repeated. `git diff --check` passed.

## 2026-09-28 — frozen ridge residual and fresh-confirmation protocol

- User requested analysis and implementation. Targeted primary-source check found prior
  POD/DeepONet residual learning (Demo et al.2302.12682) and ReZero; no new residual-method
  novelty claim. Proposed future contribution is development-only modal learnability
  selection, not implemented or verified in this experiment.
- Locked docs/RIDGE_RESIDUAL_V1.md and results/public_ridge_residual_v1/plan.json:
  frozen rank16/alpha10 map + zero-output one-layer width16/32 MLP, four correction penalties,
  five seeds, baseline-inclusive early stopping. Eight-alpha expanded ridge control.
- Reserved500 IDs outside every historical split plus conservative smoke prefix0–127.
  Plan stage read only existing protocol/checkpoint metadata, not new trajectory labels.
  Confirmation command requires the selected model hashes/selection lock committed to Git.
- Atomic full training state every50 steps, flushed event stream, selected/latest checkpoints;
  confirmation data cached in50-trajectory atomic chunks for interruption recovery.
- CPU and CUDA tests passed(3 per device): deterministic continuation including patience and
  best model, frozen baseline/zero equality, disjoint IDs, paired bootstrap, commit gate
  preventing label access and completed-confirmation skip. Full regression:26 tests passed.
- Commands: run_public_ridge_residual.py plan; after source commit, train; after selection
  commit, confirm. All use D:/miniconda3/envs/pytorch/python.exe. No confirmation results yet.

- Training at source commit4e2e3e2 completed40 residual runs/60200 updates and8 expanded
  ridge fits. Selected width32/lambda1 by five-seed development mean0.3663995661;
  frozen ridge0.3736562256, expanded ridge alpha100=0.3735998498.
- All40 frozen anchors are array-identical to the original; independent audit replays
  patience/best-step rules and verifies model/result identities.17 runs retained step0.
  Added direct reproduction/training-only test for fixed-basis ridge; CPU4 tests pass.
- Selection lock and all development evidence are being committed BEFORE the confirmation
  stage. No500-ID confirmation pair has been read, and no confirmation score is known yet.

- Selection sealed in Git8881a9c before running `run_public_ridge_residual.py confirm`.
  Full8.23GB source SHA256 matched;500 disjoint pairs extracted in10 resumable chunks.
  Only five selected residual checkpoints and predeclared controls evaluated, with no tuning.
- Fresh relative L2:residual0.4051353931±0.0018933127, frozen ridge0.4053541806,
  expanded ridge0.4045177708, mean baseline0.4043734692. Primary difference=-0.0002187875,
  paired95% CI[-0.0074978679,0.0089227175], relative gain0.0539744%; primary_success=false.
- Secondary RMSE improves5/5:mean0.1477012653 versus frozen ridge0.1509313323 (~2.14%).
  Kept the negative primary conclusion; metric disagreement is not proof of a specific
  low-energy/outlier mechanism. No retraining or retuning followed confirmation.
- Independent audit reproduced bootstrap intervals from versioned per-trajectory errors.
  Updated current conclusions, publication assessment and a dated paper addendum while
  preserving earlier tables and counts. Combined public records:245 neural including20
  historic repeats,26 deterministic ridge fits including six new repetitions of old settings.
- Full regression before confirmation:27 tests passed. New follow-up work must reserve
  another untouched confirmation set; the current500 IDs are now consumed.
- Final regression:28 tests passed, including numeric paper/report tables and paired
  confirmation statistics. Re-executing train/confirm checked identities and skipped both;
  no retraining and no confirmation labels reloaded. `git diff --check` passed.

## 2026-09-28 — objective alignment and modal learnability follow-up

- Development-only energy diagnostic completed; train-only quintile thresholds, no old
  test or500-ID confirmation labels loaded. Lowest-energy development group has47 samples,
  ridge L2=0.698359, selected residual=0.677225; highest group42 samples,0.118216→0.121598.
  This does NOT support claiming low-energy samples caused the previous confirmation failure.
- Training target-energy5th percentile=0.0155348191; proposed normalized-loss denominator
  floor is fitted from training only. Squared relative loss remains a surrogate, not equal
  to the mean nonsquared relative-L2 evaluation metric.
- Preparing strict disjoint representation-fit versus branch-CV sets to prevent POD/ridge
  label leakage into held-out CV folds. Equal-data all-mode and random-mode controls required.
- Second public PDE chosen: PDEBench Advection beta0.4, a linear transport negative control
  (not a second nonlinear task). Publisher HF revision33d90a245d98b8a250884eab613c6f9c26a13b68,
  file8232966952 bytes, SHA256d973ff2bb3c2a5edf42957ff029b78672451532ebd6bf475848ff23cfd3ee3b6.
  Added separate resumable downloader/status path, preserving the original Burgers file.
- Locked OBJECTIVE_MODAL_V1.md and machine-readable plan with disjoint new Burgers training
  split and three reserved500-ID confirmation lists (not evaluated in this development stage).
  Per task:10 full-data objective comparisons,6 strict fixed-step CV probes,25 equal-data
  correction panels;123 neural records planned over three tasks. Extra pure-ridge grids
  distinguish residual improvements from a weak alpha10 anchor, especially on Advection.
- CPU/CUDA new tests passed(2 per device): bounded normalized weights, modal criteria,
  masked continuation equality and held-CV-label perturbations leave final weights unchanged.
  Full regression30 tests passed. Formal runs follow the source/protocol Git commit.
- Source commitf807853; original Burgers41 runs completed,79500 updates. Full-data guarded
  loss improves development L2:0.366400→0.358258; RMSE0.145048→0.143010.
  Strict CV selects modes0–3; stable mask equals top-energy mask, both0.360077 vs all
  normalized0.362092 and random0.375807. Do not claim superiority over energy selection.
- Independent audit checks disjoint samples, exact frozen anchors, fixed-step CV selection,
  masks and group statistics. Added rejection of numerically degenerate modal denominators
  rather than certifying gains caused by the denominator floor; original case passes.
- Targeted primary-source novelty check: Koike et al.2605.27756 uses learned sparse POD
  mode selection for nonlinear decoding. Our procedure differs, but broad mode-selection
  originality is unsupported. Independent split is running; second-PDE download continues.
- Independent Burgers41 runs completed,67500 updates; aggregate82 records complete.
  Full-data normalized L2=0.398468 vs MSE0.407464;10/10 paired L2 improvements across
  the two independent training splits,9/10 RMSE improvements. New strict mask selects0/1,
  equals top-energy again; masked0.403133 vs all0.404991, random0.412182. Only3/5 paired
  gains over all modes on this split; do not turn group means into universal superiority.
- Ten stable/energy pairs across the two tasks have exactly identical histories/metrics.
  New post-training energy strata use development data only; both low-energy bins improve
  with the normalized objective, while the middle bin worsens slightly. This is descriptive.
- Full regression33 tests passed. Advection remains downloading; its experiment is not
  yet reported as completed. All three newly reserved confirmation sets remain unopened.
- Final Advection download completed and matched the pinned SHA256. Ran
  `D:/miniconda3/envs/pytorch/python.exe scripts/run_objective_modal.py advection`:
  41 additional records completed,89600 updates. All123 records now complete,236600
  updates total, plus48 deterministic ridge grid refits. Training source remainsf807853;
  Advection launch Git checkpointcd0eaeb; no source edits during training.
- Advection full-data normalized loss worsens L2 from0.169283 to0.175051 and RMSE
  from0.137108 to0.139769 in5/5 pairs. All five energy-bin means worsen. Stable selection
  keeps16/16 modes, so normalized/all, stable, random and energy are exact repeats.
  Across three tasks25 panel records are redundant; record counts are not independent N.
- Expanded Advection ridge selects alpha0.0001 (lower grid boundary): full1000 L2
  0.015378 and anchor400 L2=0.019756, much better than residuals. Improvement relative
  to the original alpha10 anchor0.341140 does not imply nonlinear correction is needed.
- Ran `scripts/summarize_objective_modal.py` and `scripts/diagnose_objective_energy.py`:
  audit checks all123 records, frozen anchors, strict CV, selected checkpoint hashes,
  mask identities and aggregates; stratified aggregation reproduces development scores.
  No reserved confirmation labels loaded. All three500-ID reserved sets stay unopened.
- Synced current conclusions, experiment ledger, publication assessment and dated paper
  addendum. Proposed next step is incremental nonlinear benefit beyond tuned linear
  correction, with rejection/continuous-shrinkage controls, not an implemented success.
- Final `D:/miniconda3/envs/pytorch/python.exe -m unittest discover -s tests`:34 tests
  passed, including all-three-task completion and numeric paper-table checks. Resume
  command `scripts/run_objective_modal.py all` verified identities and skipped all three
  completed tasks without retraining. `git diff --check` passed (CRLF notices only).
- Local datasets and full optimizer/RNG/model checkpoints remain ignored by Git;
  protocol, provenance, selected hashes, progress logs and results are versioned. Local
  sleep/interruption recovery is supported; a Git clone alone is not a checkpoint backup.

## 2026-09-29 — incremental nonlinear gain beyond tuned linear correction

- Locked a new nested-CV protocol in INCREMENTAL_MODAL_V1.md. Keep representation400
  disjoint from correction600 and all outer folds. Tune shared-alpha linear maps with
  3 inner folds, two objectives and18 alpha settings including pseudoinverse; correctly
  centered weighted intercept. Final masks share the same fitted residual predictions.
- Planned33 neural fits, fixed1500 steps for CV/final so no mask gets a development-
  selected stopping advantage. Compare none/all/stable/energy/continuous shrinkage.
  Three existing500-ID reserved sets remain unopened at protocol creation. Candidate,
  checkpoints and criteria must be committed before confirmation; six comparisons use
  Bonferroni paired-trajectory bootstrap intervals and a1% effect threshold.
- New tests: weighted ridge stationarity, exact linear recovery, numerical-zero modal
  exclusion, continuous shrinkage, no-effect inference and linear-tuning resume:3 passed.
  Created machine-readable plan with immutable source fingerprint before formal runs.
- Formal source/protocol commite352c0b. `scripts/run_incremental_modal.py develop`
  completed33 neural fits,49500 updates. Nested linear grids run for4 mapping fits per
  task (3 outer+1 final),3 inner folds each,36 settings per fold. These1296 candidate
  fits plus12 selected refits reuse SVDs; they are not1296 independent experiments.
- Development means (none/selected): original0.373252/0.362262 stable; independent
  0.409382/0.402073 shrink; Advection0.019756125100/itself after no-correction fallback.
  Final linear alphas100/10000/1e-6; hard Burgers masks remain energy-equivalent.
  Advection modal probes1/7 pass, but field benefit is negligible; no performance claim.
- `scripts/audit_incremental_modal.py` independently replays checkpoints and risks,
  nested split/selection rules, masks and development metrics for all33 fits. Full
  regression37 tests passed. Next seal and commit immutable selection before any
  reserved target extraction. Source/training rules will not change after this point.
- Selection artifacts and checkpoint hashes committed atad94e69 before confirmation.
  `scripts/run_incremental_modal.py confirm` opened the three reserved500-ID sets.
  Original candidate stable:0.387273→0.375353,3.0779% gain, adjusted interval
  [-0.018826,-0.005641]. Independent candidate shrink:0.379517→0.368605,2.8752%,
  [-0.016384,-0.005921]. Both pass predeclared primary criteria. Advection chooses none
  and remains0.024182229260; no improvement claim. All1500 new targets are now consumed.
- Versus energy: original exactly equal; independent shrink gain0.2013%, interval
  [-0.001847,0.000219]. Neither passes the incremental selection-rule claim. No tuning,
  retraining, checkpoint changes or post-confirmation candidate selection took place.
- Extended the separate audit to replay confirmation predictions using already-opened
  chunk caches and verify every per-trajectory error/RMSE and bootstrap statistic.
  Source/plan/model/masks remain unchanged. Synchronized current conclusions, ledger,
  publication assessment and new paper addendum; annotated previous unopened-set claims
  as historical. Later research requires newly reserved confirmation data.
- Added regression checks that the seal existed atad94e69, confirmation retained
  those candidates/IDs, paired statistics reproduce and paper tables match raw JSON.
  Initial table test used10-decimal formatting against12-decimal Advection values;
  corrected the test precision (no data or model changes). Final full suite39 passed.
- Re-running develop and confirm checks immutable identities and skips all completed
  tasks; no retraining and no new target extraction. Git diff whitespace checks pass.
  Training-internal modal table added with one-based ordering and explicit caveat:
  separate POD bases do not identify the same physical mode, and confirmation validates
  selected model performance rather than each modal hypothesis independently.

## 2026-09-29 — whole-field relevance and synthetic Allen-Cahn

- Checked public dataset provenance; PDEBench1D reaction-diffusion has an unresolved
  equation/data issue94. Chose an explicitly synthetic, reproducible Allen-Cahn second
  nonlinear PDE, not a claimed public benchmark reproduction. User-facing scope stated.
- Implemented exact reaction/Fourier-diffusion Strang splitting and independent DOP853
  QA on separate IDs. Time error2.471e-7, space error5.721e-14, independent3.091e-7;
  constant/equilibria/pure-diffusion tests pass. Three new unit tests pass.
- New field-gain protocol fixes per-mode threshold0.1% of total weighted FIELD baseline
  risk in each fold, including projection error, plus each-fold1% joint relative-L2
  gate. Old public CV/final networks are reused without retraining;11 new Allen-Cahn
  fits planned. New500-ID confirmation lists/task exclude every consumed public split.
- Before training/inspection: plan and source will be committed. No confirmation
  targets generated or opened at protocol creation. All previous records preserved.
- Source commitb2d554c; development completed. Public models reused; Allen-Cahn11 fits,
 16500 updates. New masks: Burgers1/2/4 and1/2, Advection none, Allen-Cahn1–15.
  Allen-Cahn linear0.458885→field0.359218, equal to same-k energy. Original Burgers
  field0.362597 versus energy0.364460 but old rule0.362262. New rule is not tuned further.
- Independent audit initially mixed float32 input means with float64 coefficient
  replay, causing approximately1e-11 relative differences. Corrected audit conversion
  by calling the existing coefficient helper; training/selection source unchanged.
  All development predictions and field-risk/gate computations then replayed exactly
  within original tolerances. Full regression42 tests passed before adding fresh-ID test.
- Added explicit fresh-ID exclusion regression (4 field tests passed). Committed
  complete development/model hashes/selection lock at8a4020b, THEN ran confirmation.
  Original Burgers linear0.405879→field0.393444; independent0.402369→0.389518;
  synthetic Allen-Cahn0.467579→0.366325. Gains3.0636%/3.1939%/21.6550%, all primary
  criteria pass. Advection none0.020532878061. All four new500-ID sets are now consumed.
- None of the8 predeclared comparisons versus energy/old controls establishes new-rule
  superiority (the12-comparison family also includes4 linear comparisons). Allen-Cahn is0.0521%
  worse than old all-mode correction, interval[0.000153,0.000233]. Preserve this small
  negative effect; do not adjust gates after confirmation. No new training followed.
- `scripts/audit_field_modal.py` replays every CV/development/confirmation prediction,
  full-field risks/gates, source artifacts and bootstrap inference from cached targets.
  Updated paper addendum, current conclusions, ledger and publication assessment;
  explicitly distinguish synthetic Allen-Cahn from public benchmark reproduction.
- Final full regression44 tests passed, including fresh-ID exclusions, exact seal
  from8a4020b, paper numeric tables,12 comparison outcomes and raw-error aggregation.
  Re-running develop/confirm skipped all4 completed tasks and all4 confirmations;
  no retraining, new extraction or regeneration. `git diff --check` passed. Full
  binary checkpoints and chunk caches remain local/ignored; Git versions hashes,
  protocol, per-trajectory errors and experiment history, not a binary-data backup.

## 2026-09-29 — continuous uncertainty penalty, strong nonlinear controls

- New protocol UNCERTAINTY_MODAL_V1.md retains old continuous shrinkage/all outputs,
  trajectory-only uncertainty ablation and CV-energy-matched global shrinkage. Main
  coefficient=N/(D+1.96*bootstrap SD(N)), clipped to existing[0,1] bounds. No new hard
  deletion threshold or tuning grid. This is a quadratic uncertainty penalty, NOT a
  calibrated95% confidence/safety constraint; multiplier1.96 is fixed regularization.
- Hierarchical bootstrap samples fitted fold-model/seed blocks and held trajectories;
  only3 folds/two probes exist, overlapping train sets and unmodeled refitting limit
  interpretation. Models remain frozen, zero new neural fits. Deterministic bootstrap
  replicates checkpoint every100, permitting exact continuation after interruption.
- Reserved another500 IDs/task excluding every prior public split/confirmation;
  synthetic Allen-Cahn uses1700–2199, not previously generated targets. Candidates
  must be committed before extraction;16-comparison Bonferroni confirmation criteria.
- Three new unit tests pass: continuous/no-new-zero coefficients and stationarity,
  matched correction energy, deterministic resumed bootstrap with seed dependence.
- Initial launch at07d7ca4 stopped before bootstrap or evaluation: artifact enumeration
  required omitted cv_seeds metadata. Preserved original plan.json, added explicit[0,1]
  and versioned plan_v2.json rather than overwriting the immutable initial record.
  Scientific settings unchanged; no development outcomes or new targets examined.
- Repaired preflight verified13 model artifacts/task; active source/protocol0e3217f.
  Development completed with zero new neural fits. All2000 bootstrap replicates/task,
  coefficient masks, nonzero support and energy matching replay exactly in independent
  audit. New/old continuous L2: original0.362695/0.362297, independent0.402668/0.402073.
  Neither development comparison favors new uncertainty versus old continuous.
  Allen-Cahn remains all1 and predictions coincide; Advection selects none.
- No coefficient newly becomes zero. Added fresh-ID exclusion/preflight metadata test.
  Freeze these results and commit selection before new target extraction; no changes
  to penalty strength or candidate rule follow the unfavorable development comparison.
- Full preconfirmation regression48 passed. Candidate sealbfd52cd preceded all four
  new500-ID confirmations. Confirmation completed before the interrupted turn ended;
  raw files persisted. On2026-09-30 resumed archival and architecture-direction review,
  without changing models or starting any new architecture training.
- No uncertainty candidate beats old continuous/all/matched-global controls under
  predeclared criteria. Independent Burgers uncertainty0.384878 versus old0.382451,
 0.6345% worse, adjusted difference interval[0.001133,0.003818]. Original primary fails
  despite1.4590% mean gain (interval crosses0). Independent primary3.4022% and synthetic
  Allen-Cahn21.4176% pass versus linear; Allen-Cahn is unchanged old all-mode prediction,
  not uncertainty gain. Advection deploys none. All2000 additional targets are consumed.
- Read-only fitting diagnostic on existing training/development data: penalty1 free
  coefficients minimize squared objective at delta=r/2. Allen-Cahn training actual
 0.353639, free-coefficient penalty1 oracle0.229930, POD oracle0.026088; development
 0.359048/0.230826/0.025842. Label-dependent diagnostic, not a deployable model, and
  not a pure optimization attribution. No confirmation labels used for this diagnostic.
- Historical FNO pilot test0.22835 is informative but from a different protocol; do not
  compare directly with current residual results as a new controlled superiority claim.
  Recommended next experiment is fitting/penalty/budget control before spatial branch
  and same-protocol FNO, not an unsupported new-architecture novelty claim.
- Independent audit replayed all bootstrap draws, development/confirmation predictions,
  masks, matched-energy controls and16 inference comparisons. Final full regression49
  passed, including historical candidate sealbfd52cd, raw-error statistics and paper
  tables. Previous-turn results/documentation now archived; no new architecture training.

### 2026-09-30 Ordered architecture preflight

- Added architecture_core and run_architecture_stages; protocol in ARCHITECTURE_STAGES_V1.md.
- CPU/CUDA resume test covers three architectures and both mappings (12 combinations), zero initialization and identity rejection; passed with exact model/history equality.
- Initial CUDA CNN preflight failed because adaptive pooling backward is nondeterministic. Replaced pooling with equivalent bin-average matrix multiplication; retained preflight_plan.json, regenerated active plan before real training. No development outcome was used for this repair.
- Commands: `D:/miniconda3/envs/pytorch/python.exe -m unittest tests.test_architecture_core -q`; `D:/miniconda3/envs/pytorch/python.exe scripts/run_architecture_stages.py plan`.
- Next: fit (72), commit choices, spatial (18), FNO (18); all development-only, resumable every50 steps. Existing confirmations remain consumed.

- First completed task: original Burgers24 fits; checkpoint replay audit passed24. Selected residual penalty1/1500 (dev0.363701) and direct1500 (0.369034). No-penalty residual train improves0.349670 to0.310076 with6000 schedule, while dev worsens0.370425 to0.386662. Penalty1/6000 dev0.365952 also loses to1500. These are development observations, not a fresh validation claim.
- Full regression50 tests passed before two added equivalence checks; those additional checks passed separately. Audit/report commands: `scripts/audit_architecture_stages.py`, `scripts/report_architecture_stages.py` using the pytorch interpreter.
- Independent Burgers24 fits also completed and replayed (48 total): selected residual penalty1/1500 dev0.402240 and direct1500 dev0.412777; frozen linear0.409382. No-penalty6000 dev0.439150. Full regression including equivalence checks now52 passed. Allen-Cahn fitting remains in progress.
- All72 fitting runs completed and replayed. Allen-Cahn selects residual penalty0.1/6000 dev0.287272 (seed SD0.008826), direct6000 dev0.094198 (SD0.000262), linear0.458885. Direct mapping beats the tested residual family substantially; target form and initialization both differ, so not a pure causal isolation. Choices are committed before spatial/FNO transfer; no new confirmation data used.
- CNN Burgers controls complete, audit84 total: original residual0.367876/direct0.389238; independent residual0.401304/direct0.423261. Only independent residual slightly improves on matched MLP; no consistent cross-partition advantage. CNN8848 vs MLP8752 parameters. Allen-Cahn CNN in progress.
- Spatial stage complete18/18, audit90 total. Allen-Cahn CNN residual0.063771 ±0.001256, direct0.046134 ±0.000290. Same-POD spatial structure substantially improves this task; direct remains better than residual under transferred settings. Commit before FNO stage; no additional tuning or confirmation opening.
- FNO Burgers12 runs complete, audit102 total. Original residual0.251721/direct0.195846; independent residual0.276705/direct0.220706. Direct full-field FNO is substantially stronger on both development partitions. It has136481 real parameters and a different output space; cannot attribute the difference to encoder or decoder alone. Allen-Cahn FNO still running.
- All108 runs complete; Allen-Cahn FNO residual0.043046 ±0.000211, direct0.003315 ±0.000286. Final `scripts/audit_architecture_stages.py --require-complete` replayed108 hashes/predictions/aggregates/selections. `-m unittest discover -s tests -q`:52 passed (29.638s).
- Added exploratory post-hoc FNO→same-POD diagnostic after seeing outcomes: direct projected dev0.192454/0.217435/0.026098 vs full0.195846/0.220706/0.003315. Not a separately trained candidate; not preregistered evidence. Burgers advantage remains in same output space; Allen-Cahn becomes rank-limited.
- Updated manuscript, conclusions, ledger, publication assessment and full data-only Markdown tables. No confirmation IDs opened. All binary checkpoints remain local/ignored by Git; JSON histories, plans and summaries are versioned. Resume is exact at latest atomic checkpoint, at most49 unsaved updates after an abrupt interruption; not a claim of automatic process restart.
- Completed this three-stage round. Next proposed work, not executed: parameter-matched spectral/POD encoder controls, rank16/32/64 and efficiency/sample-size curves, then lock rules before new confirmation. Existing strong shrinkage/all-mode history retained.
# 2026-10-08 manifold/token operator roadmap and representation stage

- Seal commit05b8153 preceded ANY fresh500 target extraction; confirmation_opened.json
  marks consumed before extraction. All4 arms and6 same100-shot FNO fixed; no retuning.
- Confirm gate0.486568→gateP0.458865;attention0.470762→attentionP0.442270;
  registered gains3.249%/5.694%/6.052%/9.104%, all4 pass Bonferroni4 interval/effect gate.
  Same100-shot FNO16/32 0.313237/0.318333; hybrid41.19% worse than near-size FNO16.
- Independent confirmation replay18 primary models, per-trajectory and odd512-query
  metrics, summaries and4 bootstrap decisions passed; original ID overlap0. These
 500 targets now consumed. No additional tuning/checkpoint/seed selection occurred.
- Native512 output query transfer only, same256 input; no temporal extrapolation or
  Allen-Cahn physics evidence.100 fitting labels does not count earlier larger-data
  design/hyperparameter access. Conditional bootstrap on3 fits, not seed uncertainty.
- Network latency10 warmup/50 synchronized repeats, batch64, five256/512-point fields;
  incremental allocation and inclusive process peak reported separately. No project
  trainer/test active during timings. Figures rendered and visually checked.
- Commands: confirm_physics_token.py seal/confirm; audit_physics_confirmation.py;
  report_physics_confirmation.py; plot_physics_confirmation.py. Source/CP hashes frozen.

- All126 new fits complete (81 reconstruction +9CNN/trunk +18token +12physics +6FNO100).
- Full regression68 tests passed64.876s. Physics12 final models independently replayed,
  initialization/parameter/labeled-batch AND collocation hashes independently regenerated.
  Dev gate0.463866→0.441088, attention0.454306→0.426508 with physical loss0.01.
- FNO100 width16/32dev0.287190/0.286988, both substantially below token/physics pilot;
  same labels/times/steps/batches,34261/136613 parameters. All6 retained before holdout.
- Training seconds are raw workstation observations: some late physics runs overlapped
  FNO/regression jobs, so NOT isolated efficiency evidence. Confirmation network profiling
  will run after all project trainers/tests stop, with warmed/synchronized timing.
- Next: commit audits/artifacts, seal every final model, commit seal, THEN fresh500 read.

- Full regression67 tests passed58.003s. Same100-shot/time FNO16/32 controls registered
  before confirmation (6 fits, fixed final, exact labeled batch sequence), not replacing
  registered4 factorial contrasts. New FNO trajectory CPU/CUDA resume/shape test1
  passed4.631s. Width16 rough capacity bracket only; all widths reported.
- Confirmation profiles record synchronized host-inclusive network latency, total live
  CUDA allocation AND incremental inference peak; no inference speed claims based on
  concurrent training. No fresh target opened at this entry.

- Stage3 all18 fits independently replayed;33256 params in both arms, full initial/
  batch pairing audited. Attention improves original/independent Burgers dev to
  0.288996/0.332460 vs gate0.376506/0.396253; Allen worse0.027256 vs0.022733.
  All remain worse than full FNO. TOKEN_OPERATOR_V1_RESULTS.md records full table.
- Stage4 launched only after stage3 replay. All100-shot labels drawn from registered
  training rows; time and native coordinate metadata verified; no confirmation read.
- Confirmation tooling tests1 passed0.030s (effect threshold, adjusted interval gates),
  with checkpoint/context/source sealing, partial per-model resume and consumed marker.

- Stage2 all9 runs audited: devCNN0.383864/0.405222/0.023343 vs fullFNO
  0.195846/0.220706/0.003315; CNN24356 params, FNO136481. No superiority/capacity-isolated
  attribution. Full results SPATIAL_DEEPONET_V1_RESULTS.md. Stage3 launched afterward.
- Stage4/5 registered: one-PDE Burgers100-shot multitime factorial,12 fits; physical
  JVP derivative tests2 passed5.821s, including differentiable analytic u_x/u_t/u_xx,
  correct Nu/pi, full model/optimizer/RNG CPU/CUDA bitwise resume and collocation hashes.
- Conservative historical-ID audit7829 IDs; reserved500 have zero overlap. No new
  target read. Physics stage uses ONLY training100 and historical development200.

- Stage1 audit complete:81 final models replayed,9 POD controls checked, regenerated
  batch hashes, model initialization and exact local/scrambled trainable parameter
  pairing; train/dev means and per-sample band Parseval closure passed.
- Figures generated/visually inspected; one plotting keyword syntax error corrected
  before generation. PNG/SVG and full metric table agree with replay audit.
- Stage2 training launched after stage1 replay. No new confirmation opened.

- Stage1 all81 runs completed; full replay audit starting. Full regression62 tests
  passed50.673s after stage2 implementation.
- Stage3 exact-capacity token interaction registered before training:18 fits, same
  CNN/trunk, attention versus active QKV gate, fixed64-scalar endpoint bottleneck.
  Tests2 passed8.746s: all identical initial tensors/count, every parameter gradient
  active, CPU/CUDA exact resume. No physics/temporal confirmation in stage3.

- While stage1 runs, register fixed rank64 CNN/smooth periodic-trunk DeepONet
  baseline (spatial_deeponet_v1),9 fits queued sequentially after stage1. No attention
  or physics; same600 correction rows,200 dev, guarded field loss/batches/budgets as FNO.
- Stage2 tests2 passed4.421s: first/second physical-coordinate derivatives against
  finite differences, exact periodicity, native coarse coordinates,64-scalar bottleneck,
  CPU/CUDA exact resume and configuration-change rejection.
- Time alignment is ALREADY registered in src/public_benchmark.py/protocol.json:
  discard trailing2.01 time coordinate. Upstream generator confirmed it_tot solution
  entries versus it_tot+1 coordinates and epsilon/pi diffusion coefficient:
  https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py

- User authorized five ordered stages; register them in MANIFOLD_OPERATOR_ROADMAP_V1.md.
- New stage1 family manifold_reconstruction_v1,81 AE fits:3tasks x3ranks x3arms x3seeds;
  train representation400, evaluate reused development200 only. Independent POD per rank.
- Local/permuted-local exact parameter/initial-trainable-state/batch controls; global
  MLP is same scalar bottleneck but different capacity. No physics/attention claim yet.
- HDF5 metadata read:1024 cell centers0.00048828125..0.99951171875,periodic length1;
  t-coordinate202 entries while tensor has201 slices (alignment must be separately
  resolved BEFORE spatiotemporal stage2). Existing fixed_t1 convention retained here.
- Tests3 passed including analytic spectral derivatives, Parseval bands, latent
  shapes/parameter pairing and bitwise CPU/CUDA interruption-resume (22.091s).
  Full regression60 tests passed54.721s before registered training. Checkpoints50.
- Commands: python -m unittest tests.test_manifold_reconstruction -q;
  python scripts/run_manifold_reconstruction.py plan/run (conda pytorch interpreter).
