# Resumable experiment runbook

## Allen-only exploratory joint interval pilot (2026-10-16)

New family `results/allen_interval_pilot_v1`; fixed15 continuation fits, not a
confirmation study. Protocol: `docs/ALLEN_INTERVAL_PILOT_V1.md`. Only one writer.
After process exit, resume from repository root using:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_allen_interval_pilot.py run
```

First check existing Python workers. A sleeping PC may retain the process;
do not launch a duplicate. Training atomically saves complete state every50
steps, evaluates250; training-only reference stages save every8cases. Finished
results are verified/skipped, not retrained. Sources/protocol/identities must
match the sealed plan. No new confirmation is generated or read. After15/15:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_allen_interval_pilot.py --require-complete
```

## Sparse-time/noisy-label and nonlinear Allen-Cahn pilot (2026-10-09)

New48-fit familyphysics_stress_v1. Use exactly one worker; after shutdown rerun:
UPDATE:48/48 completed and independently replayed. No background worker remains;
do not restart or change the completed training budget. Commands below are only
for recovery of incomplete local copies, not new experiment initiation.

```powershell
Set-Location 'F:\Desktop\temp\pinode_align'
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_physics_stress.py run
```

Background stdout/stderr files are inresults/physics_stress_v1. If the hidden
worker survives sleep, let it continue; do not launch a duplicate. Per-model
progress and atomic full checkpoints persist every50 steps; the aggregate
progress updates after each completed three-seed fork group. Completed model
hashes/identities are verified and skipped. Backup checkpoint binaries separately
from Git. No confirmation generation occurs in this runner. Once it completes:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_physics_stress.py --require-complete
```

## Strong FNO physics/smoothness follow-up (2026-10-08)

The continuous-time data-only baseline3/3 is complete and independently replayed.
The continuation grid30/30 and fresh500 confirmation are NOW COMPLETE; all12
confirmation predictions/five decisions replayed. Do not restart completed
training, change its budget, or reseal the consumed confirmation. The following
command is only for restoring incomplete copies from their atomic checkpoints:

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

Completed midpoint/interval diagnosis (2026-10-10), no temporal loss/training:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_interval_diagnosis.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_interval_diagnosis.py
```

Check no surviving interval worker first. References resume8-case/stage chunks;
finishedmodel rows skip by identity; no --resume. Plan/results interval_diagnosis_v1,
binary references external_data/interval_diagnosis_v1. Both predeclared strict
screens failed; no follow-up constraint is currently trained. New intermediate
development timestamps must not later be presented as untouched validation.

Completed frozen temporal diagnosis (2026-10-10): source protocol
docs/TEMPORAL_AFTER_PERIODIC_V1.md, active results temporal_after_periodic_v2.
No training or reference regeneration. Repeating the command below skips
identity-matched completed model records; audit_v3.json is the active audit.

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/diagnose_temporal_after_periodic_v2.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_temporal_after_periodic_v3.py
```

Keep V1 pilot and failed V2 alternate-AD precision checks; V3 numerical policy
is an explicitly post-failure engineering QA revision, not predictive tuning.

Current periodic-curvature study (2026-10-10): use its versioned, full-state
resumable runner directly; the generic `--resume` rule below does NOT apply.
Check for an existing `run_periodic_curvature.py run` process first. If absent:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_periodic_curvature.py run
```

Protocol: docs/PERIODIC_CURVATURE_V1.md. Results/checkpoints/logs:
results/periodic_curvature_v1. Every50 steps saves model, optimizer and RNG states.
After48 fits, run `scripts/audit_periodic_curvature.py --require-complete` with
the same interpreter before updating comparative conclusions. No confirmation
dataset is accessed by this experiment.

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
