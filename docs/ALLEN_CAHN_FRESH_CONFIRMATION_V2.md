# Allen--Cahn fresh-trajectory confirmation V2

Protocol locked 2026-10-29 before generating V2 trajectories or scoring any confirmation case. This is a frozen-model evaluation, with no new fitting or tuning.

## Version history and scope

V1 generated IDs `310000..310199` but the first evaluation attempt stopped before model scoring: its training-reference loader assumed the current training split was contained in two original cache shards. Inspection showed that the frozen model split draws its 100 training IDs from the full prior 300-case pool. V1's data are retired and never scored. This V2 plan pins all six source cache shards, selects only the exact frozen 100 training IDs for metric floors and training-scale checks, and uses a new confirmation ID block `320000..320199`.

The reused development pool is `200000..200299`; separate solver QA uses `300000..300015`. Neither is part of evaluation. V2 evaluates the 24 already audited checkpoints from `allen_pointwise_control_v1`: four arms (`none`, `full`, `interval_uniform`, `pointwise_nodes`), dense and sparse supervision, and seeds 0, 1, 2. Weights, architecture, equation, input normalization, target scale, and training are frozen. There is no new fitting, early stopping, model selection, recalibration, or hyperparameter change.

## New trajectory generation

- Equation: `u_t = 0.01 u_xx + 5u(1-u^2)`, periodic on length 2 over `[0, 0.5]`.
- Grid: 256 equally spaced periodic points; output times `[0, .05, .125, .25, .5]`.
- Initial conditions: the existing 12-mode random Fourier family, with `SeedSequence([2026092904, trajectory_id])` and IDs `320000..320199`.
- Solver: the Strang-splitting Fourier solver with `dt=0.0005`. The separate QA cases test timestep refinement (`.001`, `.0005`, `.00025`), 256/512-point spatial refinement, a DOP853 solve, and the registered low-mode residual diagnostic.
- Ten 20-case chunks and the assembled dataset have SHA-256 receipts; generation resumes from individually verified chunks after interruption.

## Frozen evaluation and metrics

Each checkpoint is evaluated on all 200 new initial fields at all five fixed times, in batches of 16. Dense primary field error averages `.05, .125, .25, .5`; sparse primary error averages the three unobserved interior times `.05, .125, .25`. All-time, common-interior, initial, gradient, and curvature metrics are retained. Derivative denominator floors use only initial fields for the exact 100 IDs in the frozen training split. The dense/sparse target scales are checked using only those IDs and the corresponding times observed during training. The plan binds all six cache files because those source shards encode the old contiguous split; the loader selects only rows matching the frozen training IDs before computing any training reference.

The primary comparison is `interval_uniform - pointwise_nodes`, separately for dense and sparse supervision. Each trajectory's error is averaged over the three fixed model seeds, then 10,000 paired trajectory-bootstrap replicates are used. The two primary intervals use Bonferroni-adjusted 97.5% percentile intervals. Per-seed contrasts and all arm metrics are retained; comparisons with `none` and `full` are descriptive. There is no equivalence margin or automatic success threshold.

The confirmation samples are independent cases from the declared initial-condition distribution. This does not test distribution shift, a second PDE, or solver-independent physical truth. Once evaluation starts, these data are consumed and cannot be used for method selection.

## Resumable commands

Run from the repository root in the existing `pytorch` conda environment:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py plan
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py prepare
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py evaluate
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_cahn_fresh_confirmation.py summarize
```

One atomic result is written per frozen checkpoint and `progress.json` is updated after each model. Rerunning `evaluate` verifies and skips completed model results. All V2 artifacts are stored in `results/allen_cahn_fresh_confirmation_v2/`.
