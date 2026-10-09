# Frozen time-dynamics diagnosis after periodic-position control (2026-10-10)

No training, new network module, physical loss, confirmation, new ID or new
time snapshot. All36 final periodic_curvature_v1 models, same existing first32
development cases/four supervised future times. Hash and reuse native1024/512/256
reference caches from physics_diagnosis_v2; source and plan commit BEFORE run.

Allen primary temporal reference: native semidiscrete PDE RHS, independently
compared with existing central FDh.0005/.001. Burgers primary: native FDh.01,
retain h.02 sensitivity; it is not exact full-band time-derivative truth. Resolve
modes0..16 by projecting original native Fourier coefficients before sampling,
correcting the older sampled-then-projected reference aliasing caveat. Full256
score remains explicitly sampled and secondary; no alias-free full-field claim.

Compare model time JVP against own FDh.002/.001/.0005 (code consistency only).
Independent auditor recomputes with forward-mode torch.func.jvp versus the
runner's reverse-over-reverse functional.jvp. Physical output scale included;
known initial mean constant must not contribute to query-time derivative.

Resolved identity: r_model = r_native + E_time - E_rhs_field. Report time and
state/PDE-right-hand-side error energies in the SAME native-ut normalization,
including signed cross term, and oracle-time/oracle-field residual magnitudes.
Oracle replacements are diagnostic, not implemented corrected models or causal
field-error allocations. Retain all seeds/arms, especially Burgers linear/low.

Descriptive future-experiment gate on Allen periodic/none: in all3 seeds,
native-resolved time-relative mean>=.1, >=10x max(native FD width sensitivity,
FD-vs-RHS discrepancy), common normalized time-error energy>rhs-field-error
energy, and oracle-time replacement reduces residual. These gates are neither
significance nor necessary/sufficient proof that time error causes field error.
Failure does not imply dynamics are accurate. No hyperparameter search here.

Observed-time local slopes do NOT test interpolation, temporal integration or
semigroup consistency between labels. If slopes remain problematic, first
register a fixed existing-development midpoint/short-interval test before
training a temporal constraint. No reserve access until model/rule locked.

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/diagnose_temporal_after_periodic.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_temporal_after_periodic.py
```

Per-model JSON atomically saved, resume SAME command without --resume; skip only
identity-matched finished rows, one writer. Results: results/temporal_after_periodic_v1.
No optimizer/checkpoint needed for frozen inference; original checkpoints unchanged.
