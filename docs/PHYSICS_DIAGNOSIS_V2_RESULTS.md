# Frozen-model diagnosis results (2026-10-09)

36 final stress models diagnosed; no training, architecture change, selected
postprocessor or fresh confirmation. Rules/source seal8d2d62e before diagnosis,
scalar-only JSON repaircf149dc before completedv2. Thev1 numerical files and nine
saved Allen-Cahn results remain unchanged; v2 reproduces those nine exactly.
Output/source/model hashes, spectral energy shares and temporal FD stability
checks pass. Extra time snapshots are existing32 development IDs, not new cases.

## 1. Allen-Cahn: boundary-localized high-frequency error is real

Three-seed means of the data-only arm:

| Diagnostic | Value |
|---|---:|
| Field relative error | 0.043007 |
| Gradient relative error | 0.210859 |
| Second-derivative relative error | 3.694110 |
| Second-derivative error energy in modes65..128 | 90.17% |
| Second-derivative error energy at first/last4 nodes | 98.25% |
| Second-derivative denominator-floor active fraction | 0% |

The PDE/curvature arms show essentially the same pattern (~90.2%-90.3% highest-
band share,98.3%-98.4% boundary concentration). The eight edge nodes occupy only
3.125% of the grid. Thus large curvature error is NOT just tiny denominators;
it is actual error energy concentrated at the periodic seam and high modes.
Data-only seam-error RMS is0.0230/0.0514/0.0408 across seeds.

Fixed Fourier projection is a counterfactual diagnostic, not a selected method:

| Data-only output | Field error | Gradient error | Curvature error |
|---|---:|---:|---:|
| Original | 0.043007 | 0.210859 | 3.694110 |
| Keep modes0..16 | 0.042723 | 0.129282 | 0.324997 |
| Keep modes0..32 | 0.042906 | 0.146842 | 0.554428 |
| Keep modes0..64 | 0.042966 | 0.167407 | 1.177370 |

This supports diagnosing unnecessary high-frequency contamination. It does NOT
establish a general useful cutoff, filtered-model independent gain, or the causal
source of that contamination. Current TimeFNO lift includes a nonperiodic linear
coordinate feature and local pointwise paths; this is a plausible seam mechanism,
but only a controlled positional-encoding/periodicity ablation can establish it.
No encoding or checkpoint was changed here.

Burgers differs: boundary concentration~3.5%-3.7%, close to grid fraction, despite
~86% curvature-error energy in the highest band. Do NOT transfer the Allen-Cahn
seam explanation to Burgers. Missed fine-scale shock structure and temporal
reference limitations remain distinct candidate contributors.

## 2. Time derivative: implementation is consistent; dynamics are not learned well

Across36 models, JVP versus model's own central FD at h0.001 has mean relative
disagreement~5e-5..2.3e-4; h0.0005 also passes the numerical audit sanity bound0.001.
The derivative code is supported numerically. This does not certify learned
dynamics against the PDE or temporal discretization.

| Task/condition | No penalty | PDE | Curvature |
|---|---:|---:|---:|
| Allen-Cahn dense: native FD reference | 0.325175 | 0.321884 | 0.327393 |
| Burgers sparse: resolved0..16 FD reference | 0.973229 | 0.969842 | 0.967967 |
| Burgers noisy dense: resolved0..16 FD reference | 0.905365 | 0.897595 | 0.889424 |
| Burgers noisy sparse: resolved0..16 FD reference | 0.976137 | 0.973296 | 0.971647 |

These use first32 development cases and registered primary times. Native time
differences are sampled onto the model256 grid before scoring; the model's
resolved band is thus the Fourier band of the retained samples, not an alias-
free projection of all native1024 temporal coefficients. Allen-Cahn
native FD agrees with spectral semidiscrete RHS to1.17e-6 mean relative error,
so its~32% prediction derivative error is not explained by reference noise.
PDE provides a small improvement there, not accurate learned time dynamics.

Burgers reference derivative has MATERIAL step-size sensitivity: full-field
native h0.02 versus0.01 mean disagreement44.40%; resolved0..16 disagreement
12.34%. Therefore full-field~95%-99% prediction-versus-FD errors cannot be
reported as precise physical derivative errors against an exact ground truth.
Low modes are more stable, but not perfectly certified. No temporal extrapolation
claim: numerical probes extend just beyond the final label only for differencing.

## 3. Discretization: separate space, time and nonlinear closure

For the fixed32 development IDs, Burgers median resolved residuals at h0.01:
native1024 about1.3%-2.3%, raw256 about1.7%-4.0%, across four times. At h0.02,
native1024 rises to5.4%-9.0%. Almost-full-band residuals remain large EVEN at1024
(~26%-52%), and change with differencing width. The coarse full-band residual
does NOT isolate a spatial-discretization inconsistency; temporal undersampling
is confounded. Earlier full-band enforcement caution remains valid, but do not
attribute all observed residual to spatial underresolution.

The dealiased nonlinear low-mode closure gap between native1024 and filtered256
truth has RMS0.03739 versus native resolved FD residual RMS0.94883. Its coefficient
identity closes exactly, but it is not the dominant aggregate discrepancy in
this diagnostic. This does not rule out local/other-resolution closure effects
and is not a complete learned-error causal decomposition.

Allen-Cahn is numerically much cleaner: median resolved and almost-full residual
~5e-7..7e-7 at all1024/512/256 grids at h0.0005; h0.001 residual~2e-6..3e-6.
Nonlinear closure gap~6.5e-16. Its large predicted curvature error is therefore
not explained by numerical target/residual inconsistency under these checks.

## Next scoped experiment, NOT implemented in this diagnostic turn

First isolate periodicity/coordinate-path effects on Allen-Cahn using paired
retraining, same capacity/budget and strong data/curvature controls; do not
replace coordinates in already-trained models and call it a fair intervention.
Compare full-band smoothing to current resolved-band smoothing to test whether
high-frequency contamination is the relevant regularization bottleneck.
For Burgers, do not escalate full-band physical loss on coarse-time targets;
first separate reliable resolved-time constraints from fast shock motion and
reference aliasing. New architecture blocks are not supported by this evidence.

All results are exploratory development diagnostics. Both earlier500 Burgers
confirmation sets remain consumed; reserved synthetic500 was not generated.
No filter/encoding/loss rule is selected on these results for independent testing.
Final regression80 tests pass55.497s; audit36 artifact/source/model identities and
nine repair reproductions pass. This audit checks invariants and repair identity,
not a second independent recomputation of all36 temporal predictions.
