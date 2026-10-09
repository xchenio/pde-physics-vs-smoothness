# Periodic position and curvature-band paired results (2026-10-10)

Complete:48 fits (12 warm starts,36 final models),83 regression tests passed;
audit48 predictions/label identities/RNG streams/forks/summary replays passed,
including18 exact historical linear-control tensor matches. Source seal0f2c912.
Protocol: PERIODIC_CURVATURE_V1.md. Raw results: results/periodic_curvature_v1.
No fresh confirmation access. All comparisons use reused development data.

Same34209-parameter FNO,100 training trajectories, five clean labels,1500+750
steps and three seeds. Only one existing scalar coordinate feature changes;
no layers, channels or attention added. Low curvature uses modes0..16; full
uses0..128 with Nyquist-correct energy. Fixed train-gradient ratio0.3, not a
fully optimized regularization strength. Final checkpoints, no best-step selection.

## All final arms

Metrics are three-seed means; field/gradient/curvature are sample-relative errors,
lower better. Edge is fraction of curvature error energy in boundary8 nodes;
seam is field seam-error RMS, not a hard endpoint-periodicity certificate.

| PDE | Coordinate | Smoothing | Field | Gradient | Curvature | Edge fraction | Seam RMS |
|---|---|---|---:|---:|---:|---:|---:|
| Burgers | linear | none | .270949 | 1.007247 | 1.484856 | .036423 | .108043 |
| Burgers | linear | low | .265248 | .984412 | 1.451139 | .036027 | .107280 |
| Burgers | linear | full | .266738 | .983520 | 1.222072 | .034825 | .105504 |
| Burgers | periodic | none | .273427 | .971355 | 1.004199 | .034287 | .100669 |
| Burgers | periodic | low | .268437 | .957691 | .996243 | .034276 | .102498 |
| Burgers | periodic | full | .268264 | .955127 | .990694 | .034225 | .105045 |
| Allen-Cahn | linear | none | .043007 | .210859 | 3.694110 | .982545 | .038424 |
| Allen-Cahn | linear | low | .043966 | .213937 | 3.699850 | .984081 | .038418 |
| Allen-Cahn | linear | full | .043561 | .163701 | 1.821634 | .910503 | .016970 |
| Allen-Cahn | periodic | none | .037159 | .119583 | .336740 | .025672 | .002443 |
| Allen-Cahn | periodic | low | .038679 | .126077 | .334329 | .026019 | .002916 |
| Allen-Cahn | periodic | full | .038553 | .123806 | .317558 | .026321 | .002886 |

## Interpretation and limits

Allen-Cahn periodic/no-smoothing versus linear/no-smoothing improves mean field
13.60% and curvature90.88%. Curvature high-band65..128 share falls90.17% to
0.0313%; boundary concentration falls98.25% to2.57%. This paired intervention
supports the coordinate-path hypothesis, but does NOT isolate an exclusive cause:
single sine is non-injective and changes absolute-position information as well.
It does not hard-enforce continuous periodic outputs or their derivatives.

Field gains are NOT stable across all seeds: relative improvements are
-5.58%,+36.43%,+5.50%. The three periodic arms pass the predeclared exploratory
field/curvature screen, not a significance test or independent verification.
Adding low/full smoothing to the periodic model worsens mean field error;
full smoothing improves curvature modestly. Linear/full halves curvature but
misses the field screen (field degrades1.29%). Do not select full smoothing
solely from a derivative metric or claim a universal combined-method gain.

Burgers retains linear/low as best mean field control. Periodic features improve
derivative means but not the field mean versus their corresponding linear arms;
curvature remains near1, high-band share~86%, and none passes the predefined
50% curvature-reduction screen. This is not successful shock/multiscale recovery.
No temporal-derivative or cross-resolution success was tested in this family.

## Cost evidence

Sum of saved training-segment timings over48 fits:921.28s (~15.35min), excluding
evaluation, checkpoint I/O and launch time; not end-to-end elapsed time. Per750-step
continuation seed-mean16.36..18.18s without smoothing,21.41..22.29s with smoothing.
Recorded peak allocated CUDA memory in final segments95.94..98.97MiB; not total
GPU device occupancy or a fresh isolated memory benchmark. All continuations
compute both penalty forwards, even none; backward work differs. No standalone
inference timing was measured here; parameter parity does not prove cost parity.

## Next controlled work, before new modules

1. Freeze these results. Keep Allen periodic/data-only and periodic/full alongside
   Burgers linear/low as task-specific strong controls; don't erase losing arms.
2. On development only, recheck learned time derivatives after removing the seam
   artifact. For Allen use converged numerical time references; for Burgers retain
   native time-step sensitivity and resolved-band comparisons, not full FD gold.
3. Only if dynamics remain the bottleneck, compare a time-consistent constraint
   against matched spatial curvature controls with fixed architecture/budgets.
   Sparse-time tests must not leak omitted labels into normalization or calibration.
4. Lock a task-specific rule before genuinely unused trajectory confirmation.
   Reserve IDs remain unopened. Cross-resolution requires its own registered test.

Publishable contribution remains a candidate diagnostic/controlled methodology,
not novel FNO architecture, unique PINN combination or universal physics gain.
