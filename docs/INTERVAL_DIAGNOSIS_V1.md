# Fixed midpoint and short-interval development diagnosis (2026-10-10)

No network module, loss or training. All36 frozen final models; same first32
existing development IDs per PDE. NEW adjacent/midpoint development snapshots
are explicitly accessed this round, but never fitting targets, normalization or
calibration. No new IDs, reserve access or independent-verification claim.

Exact Allen midpoints.025/.0875/.1875/.375, secondary late-interior.475;
half widths.005/.01. Native1024 Strang reference dt.0000625 versus.000125 and
independent512 fine-grid QA; all times align and reference generators retain
old supervised labels as anchors. Save/resume each8-case chunk, preserve all
reference identities. No changes to old solver/caches or training observations.

Burgers original native1024 HDF frames every.01. Fixed centers.05/.18/.38/.75,
secondary.95; .18/.38 have+.005 offset from exact mids.175/.375. Half widths
.02/.04. Native Simpson.01 versus.02; model.005 versus.01. No invented finer
truth, derivative gold or time extrapolation. All intervals stay within labels.

Field/gradient/curvature and time derivatives at centers; native Fourier modes
0..16 BEFORE sampling for resolved signals. Primary4 centers, secondary late
point separately. Field metrics have same cohort but new times versus old study.
No direct field/time-percent causal ranking; record absolute incrementRMS and
floor activity, which matters when intervals shrink.

Main interval identity:

`r_pred = r_native + (delta_pred-delta_true) - (integral_rhs_pred-integral_rhs_true)`

Common native-increment norm for both error energies, signed cross term and
oracle replacements. Error cancellation and reference defects retained; small
residual is not accuracy. Model/reference quadrature and solver/space discrepancies
reported separately, empirical checks NOT mathematical uncertainty bounds.

Descriptive pilot gate for strong Allen periodic/none and Burgers linear/low:
ALL3 seeds x BOTH windows x ALL4 primary centers: incrementmean>=.1,
>=10x max(native closure, reference quadrature, solver/spatial increment or RHS
integral discrepancies); model quadrature<=.1*incrementmean; flooractive0.
No searching favorable centers. Pure increment dominance needs its common error
energy to exceed state-RHS-integral energy everywhere, reported separately.
Screen is not significance/causality; it informs discussion, not automatic training.

Extra fixed counterfactual: one full low-mode endpoint increment repair using
integrated predicted RHS (gamma1, no future truth as input); retain uncorrected
baseline and all outcomes. This is a local diagnostic, NOT an exported predictor,
global trajectory method, loss implementation, selected winner or confirmed gain.
No semigroup refeed in this round: evolved branch-input distribution shift could
confound that optional test; interval consistency does not imply semigroup success.

Candidate future innovation: continuous interval/frequency constraint weighting
from TRAIN-INTERNAL numerical reliability, not developer benefit or hard deletion.
Needs uniform-integral and curvature controls, matched budgets, locked rules and
unused confirmation. Novelty and benefit are unproven. Prior art already covers
[temporal consistency](https://arxiv.org/abs/2301.13262),
[weak integral residuals](https://arxiv.org/abs/1912.00873), and
[midpoint/Fourier numerical consistency](https://arxiv.org/abs/2512.04863).
These papers motivate the diagnostic but do not establish a unique contribution.

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_interval_diagnosis.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_interval_diagnosis.py
```

Same command resumes references/model rows, no --resume, one writer; compare
source/plan/reference/model hashes before skipping. Sealed code/protocol commit
before new references/results. Binary caches ignoredGit, JSON records retained.
