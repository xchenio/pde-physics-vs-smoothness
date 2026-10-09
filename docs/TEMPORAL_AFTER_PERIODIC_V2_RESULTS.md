# Time dynamics after periodic artifact control (2026-10-10)

36 frozen final models diagnosed; no training/modules/new snapshots/confirmation.
87 regression tests passed85.564s, appended5 targeted tests passed4.424s.
V3 audit completes all36 field/alternate-AD temporal/energy/summary replays,
including18 EXACT old pilot reproductions; original numerical failures retained.
Sourcev1 seal74a5bcd; parent-metric precision repairv2 seald61c886.
V1 preserves18 Allen rows, which V2 reproduces exactly. All previous data remain
development/consumed; none is reclassified as unseen validation.

## Scope and reference

All six final arms per PDE, three seeds each. Same first32 EXISTING development
trajectories, four supervised future times. Field32 below is NOT the200-case
field primary from the parent experiment; do not cross-compare different cohorts.
Temporal normalization is different from field normalization; their percentages
cannot directly rank causal contributions to field error.

Allen uses native1024 semidiscrete RHS as the primary reference. Native FD widths
.0005/.001 disagree3.417e-6 in resolved modes; FD/RHS discrepancy1.161e-6.
Burgers uses native FDh.01: width sensitivityh.02 versus.01 is12.34% resolved
and44.40% full. Native FD/RHS resolved discrepancy4.86%, full49.91%, so full
derivative errors near97% cannot be read as exact physical truth.

Resolved0..16 scores project ORIGINAL native Fourier coefficients, before spatial
subsampling; fixes older sampled-then-projected aliasing caveat. Full scores use
sampled native256 reference and remain secondary. Model-JVP/own-FD discrepancies
over registeredh.002/.001/.0005 never exceed4.58e-4 in relative mean. Agreement
checks differentiation implementation, not correct learned time dynamics.

## All model groups (three-seed means)

| PDE | Coordinate | Smoothing | Field32 | Time full, sampled | Time resolved, native |
|---|---|---|---:|---:|---:|
| Allen-Cahn | linear | none | .040307 | .325175 | .324480 |
| Allen-Cahn | linear | low | .041264 | .327393 | .326835 |
| Allen-Cahn | linear | full | .040698 | .325554 | .325094 |
| Allen-Cahn | periodic | none | .035631 | .289702 | .289333 |
| Allen-Cahn | periodic | low | .037256 | .293664 | .293433 |
| Allen-Cahn | periodic | full | .037106 | .293545 | .293362 |
| Burgers | linear | none | .269659 | .980399 | .926772 |
| Burgers | linear | low | .263438 | .970968 | .911743 |
| Burgers | linear | full | .264721 | .975378 | .917541 |
| Burgers | periodic | none | .274860 | .982390 | .933275 |
| Burgers | periodic | low | .269332 | .971844 | .917030 |
| Burgers | periodic | full | .268631 | .971171 | .915139 |

Allen periodic/none reduces resolved derivative error10.83% relative to linear/
none, but still28.93%. Per-seed31.53%,26.63%,28.64%; not all-seed improvement
over linear (seed0 worsens). At times.05/.125/.25/.5 its resolved errors are
18.38%,11.68%,14.07%,71.60%. Late-time discrepancy is the largest relative one;
time-dependent signal magnitudes and absolute errors must accompany this fact.
Spatial smoothing does not fix it, and full smoothing is not better than none.

For periodic/none, native sampledut RMS by time is1.3144/1.3112/1.1591/.6890;
predictedut RMS1.2216/1.2993/1.1689/.6115; absolute error RMS
.2610/.1532/.1637/.4142. The late-time error is real in absolute units too,
not only shrinking denominators: minimum native case RMS at.5 is.1307 and
denominator floors are inactive for all seeds. Sampled full and native-resolved
metrics differ slightly; do not mix their by-time averages.

## Is time error the dominant resolved bottleneck?

Use exact coefficient identity r_model=r_native+E_time-E_rhs_field. All energies
below use the SAME per-case native-ut norm. Include the signed cross term;
these are not additive positive causal contributions to field error.

Allen periodic/none three-seed mean time energy.200847, state-RHS energy.573608,
cross+.365822. Time energy exceeds state-RHS energy only in seed1, not seeds0/2.
Mean resolved residual RMS.369600; oracle replacing only predictedut reduces it
to.210220, while oracle replacing only state RHS gives.269524. Both errors matter.
The predeclared ALL3 dominance screen FAILS despite significant error relative to
the empirical reference discrepancy and oracle-time residual reduction in all3.
This is a descriptive failed research screen, not a significance test; reference
width sensitivity itself is empirical, not a mathematical error bound.

Burgers linear/low mean time/state-RHS energies.958837/1.165757, cross-1.426865:
substantial cancellation. Model residual RMS3.022547 is smaller than oracle-time
5.520323 or oracle-state-RHS5.599685. A smaller residual can coexist with inaccurate
derivatives and fields; do not optimize a residual alone or treat oracle changes
as guaranteed improvements. Reference uncertainty further limits causal claims.

## Decision and next step

Time derivatives remain inaccurate after artifact reduction, but evidence does
NOT establish time derivative as the unique/main bottleneck. Do not add a global
pointwise time-derivative loss immediately or claim that it will improve fields.

Next diagnostic: register fixed midpoint and short-interval checks on EXISTING
development IDs, keep the same frozen controls. Test whether local slope errors
persist between labels and whether integrated increments/PDE evolution agree.
Allen numerical references should retain timestep convergence checks. Burgers
should use native/resolved interval increments with explicit time-resolution
uncertainty, not force a questionable full-frequency derivative target.

Only if that test supports an identifiable temporal inconsistency, compare a
small temporal/integral consistency loss with matched strong spatial curvature
controls at fixed architecture, budgets and training-only calibration. Judge by
field/derivative accuracy together, not residual reduction alone. Include both
none and full/low smoothing; preserve Allen periodic/none and Burgers linear/low.
Do not use development diagnostic derivatives as new labels without declaring
that changed supervision. Lock rules before opening any reserved trajectories.

This turn tests supervised-time local slopes only, not interpolation, semigroup
consistency, cross-resolution, noise or few-shot temporal success. No temporal
loss has been implemented or trained. Paper conclusion remains exploratory.

## Numerical QA and audit history

The original parent-field replay failed because float32 truth norms in legacy
trajectory_errors differ~1.6e-8 from new float64 Fourier diagnostics. V2 preserves
V1 and uses the ORIGINAL metric for strict parent replay; no criterion relaxed.

V2 alternate-AD audit then stopped on one near-zero JVP element differing1.18e-7.
An explicit all36 precision probe retained659/1179648 raw element failures and
28 tiny own-FD case/time metric failures across15 Burgers models. All primary
field/temporal diagnostics passed their original tolerances. Maximum normwise
relative AD difference1.327e-7; maximum absolute difference1.144e-5, which is
3.933e-6 of model derivative signal RMS. No numerical predictive benefit implied.

V3 numerical-quality policy was registered AFTER this probe, transparently:
global RMSrelative<=1e-6, peak/signalRMS<=1e-5, case/time RMSrelative<=1e-5.
Primary metric tolerance remains rtol2e-5/atol1e-7, strict field replay1e-10/1e-11.
Own-FD diagnostics replay against their ORIGINAL reverse-AD reference; alternate
AD is used for primary temporal/decomposition replay. The original failed
elementwise protocol did not pass and is not retroactively called successful.
This is engineering numerical QA, not a certified error bound or model-selection
threshold. Audit artifact: audit_v3.json; probe/policy/old sources retained.
