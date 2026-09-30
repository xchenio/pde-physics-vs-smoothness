# Ordered architecture diagnosis (development only)

2026-09-30. Retain all previous experiments. No confirmation set is reopened as unseen data.

1. Fixed rank-16 POD / MLP32: residual penalties 0, 0.1, 1 and budgets 1500, 6000; direct coefficients with penalty 0 at both budgets. Three paired seeds on original Burgers, independent Burgers and synthetic nonlinear Allen-Cahn: 72 fits.
2. Select residual penalty/budget and direct budget separately by mean fixed-final development relative L2. Commit choices, then transfer both settings to periodic CNN/POD: 18 fits.
3. Transfer the same settings to full-field FNO: 18 fits. Report parameters and optimization time, not equal-compute claims.

All models share the frozen linear anchor, normalization and representation information; train on the same 600 correction rows, evaluate on 200 development rows. Normalized squared error has a training-derived denominator floor. It is not exactly the mean relative L2 selection metric. No early stopping/checkpoint selection in this diagnostic. Each budget uses its own cosine horizon, so a budget comparison also changes the schedule. Direct mapping also changes initialization versus frozen-ridge residual mapping. Architecture-specific tuning is deliberately not included, so this is a transferred-setting comparison, not a best-achievable architecture ranking.

Full model, optimizer and random states are atomically saved every 50 updates. Evaluation every 250 updates; completed runs are reused only when identities match. Resume commands from repository root:

```powershell
D:/miniconda3/envs/pytorch/python.exe scripts/run_architecture_stages.py fit
# Commit fit_selection.json before the next stage.
D:/miniconda3/envs/pytorch/python.exe scripts/run_architecture_stages.py spatial
D:/miniconda3/envs/pytorch/python.exe scripts/run_architecture_stages.py fno
```

The label-dependent penalized free-output oracle is a diagnostic squared-objective optimum, not a deployable predictor or proof of an optimization bottleneck. The POD oracle is not a lower bound for full-field FNO.
