# Resumable experiment runbook

## Strong FNO physics/smoothness follow-up (2026-10-08)

The continuous-time data-only baseline3/3 is complete and independently replayed.
Do not rerun completed training to change its budget. The continuation grid is
new and resumes automatically, including full optimizer and both RNG streams:

```powershell
Set-Location 'F:\Desktop\temp\pinode_align'
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_fno_regularization.py run
```

Run only ONE copy. If a process survived computer sleep, let it continue; do not
start a duplicate writer. After shutdown, rerun the command; at most<50 unsaved
steps of the current arm are repeated. Completed result/checkpoint hashes are
checked before skipping. Progress files: `results/fno_regularization_v1/progress.json`
for completed groups, and each arm's `progress.json` for its latest saved step.
All30 planned final models use2250 total steps, not the baseline1500-step budget.
No confirmation IDs are selected/opened by this runner. After training stops:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_time_fno_runs.py fno_regularization_v1 --require-complete
```

Git does NOT include binary checkpoints; back up the local `checkpoints` folders
to preserve trained weights. Do not edit sealed core/protocol files in place.

## Spatial latent/operator first round (2026-10-08)

Five new standalone immutable scientific runners resume automatically from their
latest atomic checkpoints; rerun the SAME `run` command, without `--resume`:

```powershell
Set-Location 'F:\Desktop\temp\pinode_align'
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_manifold_reconstruction.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_spatial_deeponet.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_token_operator.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_physics_token.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_fno_small_trajectory.py run
```

Completed runs skip training. Model/optimizer/step/RNG/history checkpoints every50;
evaluation every250. Config/source/environment mismatches fail rather than silently
mixing results. Never delete local `results/**/checkpoints`: Git tracks source and
result evidence but intentionally excludes model/optimizer binary weights; back them
up separately if moving computers. Preserve the validated pytorch environment.

The first round is COMPLETE (126 fits,68 tests). Fresh confirmation500 IDs are NOW
CONSUMED. Never reseal/retune this family or call those IDs untouched. For interrupted
confirmation only, rerun `scripts/confirm_physics_token.py confirm`: it verifies the
committed seal and skips saved per-model results; completed confirmation skips entirely.
No new confirmation label extraction or new training is needed to read the results.
See docs/PHYSICS_TOKEN_V1_CONFIRMATION.md and the spatial-latent paper follow-up.

The machine may sleep at any time. Long jobs must therefore be *configuration-resumable*, not
dependent on a live terminal.

## Rules

1. Use the experiment runner introduced in `scripts/experiment_runner.py`, not an ad-hoc shell
   loop.
2. The runner writes `manifest.json` before the first command and emits one JSON event per state
   transition. It skips a job only after its declared completion artifact exists.
3. Use a distinct output directory per experiment family; never reuse an old result directory
   for a different configuration.
4. Commit the manifest, event log, summary, and code changes after each completed experiment
   family. Do not commit transient checkpoints unless they are needed for a published analysis.
5. After a sleep/interruption, rerun the exact same command with `--resume`; the runner restarts
   only jobs lacking their completion artifact.

## Environment

```powershell
& 'D:\miniconda3\Scripts\conda.exe' run -n pytorch python --version
```

The validated environment is documented in `environment.yml`. The current interpreter is
`D:\miniconda3\envs\pytorch\python.exe` with CUDA-enabled PyTorch 2.14.0+cu130.

## Required evidence per run family

- command line and immutable configuration;
- Git commit and source/environment fingerprints;
- dataset provenance and checksum;
- job-level start/finish/exit events;
- raw result files and an aggregated summary;
- an explicit status when a run is interrupted or fails.
