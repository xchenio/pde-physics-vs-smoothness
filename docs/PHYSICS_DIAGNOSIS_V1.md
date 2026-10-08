# Development-only derivative/consistency diagnosis (2026-10-09)

UPDATE: sealedv1 stopped after nine persisted records due to NumPy float32 JSON
compatibility. V1 is preserved; scalar-onlyv2 repair completed all36 models.
Results: PHYSICS_DIAGNOSIS_V2_RESULTS.md. Usev2 runner for recovery, notv1.

No architecture changes, training or new confirmation access. All36 final
stress models (three arms x three seeds x four conditions) remain frozen.
Existing200 development fields quantify Fourier error; fixed FIRST32 existing
development IDs per PDE quantify temporal/discretization effects.

Four fixed bands0..4/5..16/17..64/65..128 decompose actual spectral squared
errors with Parseval closure, including proper Nyquist treatment. This is not
a truncated POD captured-energy claim. Field/gradient/curvature scores and
boundary8/256 error-energy concentration are recorded. Denominator floor uses
0.001 clean TRAIN INITIAL RMS for that derivative; floor-active fraction is
reported, to distinguish metric denominator effects from actual artifacts.

Fixed posthoc Fourier cutoff16/32/64 outputs are DIAGNOSTIC counterfactuals,
not selected/retrained models, and no winner is exported. Retain all outcomes.
Time JVP holds input/grid fixed and is compared to model's own central differences
at half-width0.001/0.0005 and to native numerical-data derivatives.

NEW diagnostic development access: Burgers adjacent snapshots at0.01/0.02 around
the four future times on existing32 IDs, including1.01/1.02; Allen-Cahn same32
development IDs regenerated at1024/512/256, dt0.00025 and neighbors0.0005/0.001,
including0.5005/0.501. Numerical differentiation around the last supervised time
is not an extrapolation test or new independently untouched dataset.

Native truth PDE residuals are compared across grids, time widths and band16
versus almost-full spectrum. Lowpassed256 truth is compared to raw retained256
truth. The unresolved nonlinear-term coefficient difference is checked against
the EXACT low-mode residual difference; do not confuse it with a complete learned
field error attribution. Allen-Cahn native time differences are also compared to
the spectral method-of-lines RHS. Extra frames are never passed to model fitting,
normalization, weight selection or calibration. No confirmation IDs are generated.

Two analytic tests pass: Fourier derivative amplification/Parseval band closure/
lowpass reconstruction, independent NumPy Burgers residual and cubic dealiasing.
Source and immutable diagnostic rules must be committed before running.

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/diagnose_physics_stress_v2.py run
```

Each model's diagnostic result is atomic and skipped on valid rerun. Native
reference arrays are cached atomically outside Git, including explicit visited
development IDs/time snapshots. This is exploratory diagnosis, not causal proof
or a confirmatory success test.
