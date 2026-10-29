# Allen--Cahn fresh-trajectory confirmation V1

Protocol locked 2026-10-29, before generating or scoring the confirmation trajectories. This is a frozen-model evaluation, not a new training or tuning study.

## Question and scope

Check whether the previously observed interval-versus-time-pointwise comparison persists on genuinely new Allen--Cahn initial conditions. The existing Allen--Cahn pool `200000..200299` was reused during development and is excluded from this confirmation. The new set contains 200 trajectories, IDs `310000..310199`. Separate solver QA uses IDs `300000..300015`; none of those QA cases enter model evaluation.

The evaluation reuses the 24 already audited checkpoints from `allen_pointwise_control_v1`: four arms (`none`, `full`, `interval_uniform`, `pointwise_nodes`), dense and sparse supervision, and seeds 0, 1, 2. Model weights, input normalization, target scale, equation, architecture, and training protocol are frozen. No new fit, early stopping, model selection, loss recalibration, or hyperparameter change is allowed.

## New trajectory generation

- Equation: `u_t = 0.01 u_xx + 5u(1-u^2)`, periodic on length 2, over `[0, 0.5]`.
- Grid: 256 equally spaced periodic points; output times `[0, .05, .125, .25, .5]`.
- Initial conditions: the existing 12-mode random Fourier family. Each case uses the existing `SeedSequence([2026092904, trajectory_id])` construction, with IDs disjoint from all prior training/development cases and the solver-QA IDs.
- Solver: the project's Strang-splitting Fourier solver with `dt=0.0005`. Before evaluation, the versioned QA checks timestep refinement (`.001`, `.0005`, `.00025`), 256/512-point spatial refinement, a separate DOP853 method-of-lines solve, and the registered low-mode residual diagnostic on the separate QA IDs.
- The generated initial states and five solution snapshots are stored with a SHA-256 receipt. Generation is chunked into ten 20-case files and can resume after interruption.

## Frozen evaluation

Each checkpoint receives the new initial field and the five fixed query times. It is evaluated in batches of 16. Dense-condition primary field error averages times `.05, .125, .25, .5`; sparse-condition primary error averages the three unobserved interior times `.05, .125, .25`. All-five-time and initial-condition errors are also retained. Gradient and curvature diagnostics use the same Fourier definitions and training-derived denominator floors as the prior Allen--Cahn analysis; floors are computed from the original training initial fields only.

The primary comparison is `interval_uniform - pointwise_nodes` separately for dense and sparse supervision. The unit for paired resampling is the trajectory: per-trajectory errors are first averaged over the three frozen model seeds, then 10,000 paired trajectory-bootstrap replicates are used. The two primary intervals use a Bonferroni-adjusted 97.5% percentile interval. Per-seed contrasts and every arm's metrics are retained. Comparisons with `none` and `full` curvature are reported as secondary descriptive controls. No equivalence margin or automatic success threshold is defined; close means will not be called statistically equivalent.

The new trajectories are generated from the same declared initial-condition distribution, so this tests generalization to independent cases from that distribution. It does not test a shifted distribution, another PDE, or solver-independent physical truth. Once the frozen models are scored, this confirmation set is consumed and will not be used to select further methods.

## Resumable commands

Run from the repository root in the existing `pytorch` conda environment:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py plan
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py prepare
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py evaluate
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py summarize
```

`evaluate` writes one atomic result per frozen checkpoint and updates `progress.json` after each model. After interruption, rerun `evaluate`; completed result files are hash-checked and skipped. `prepare` similarly resumes by verified 20-case chunks. Results and the confirmation data are recorded under `results/allen_cahn_fresh_confirmation_v1/`.
