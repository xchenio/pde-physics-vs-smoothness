# Midpoint and short-interval results (2026-10-10)

COMPLETE:36 frozen models,360 windows,8 reference chunks/16 solver stages.
Audit36 direct-forward/alternate-AD replays,2160 independent SciPy Simpson
checks, signed-energy/oracle identities and exact group/screen aggregation pass.
Source/protocol seal584b5bf, auditor seal0b7d524. Full94-test regression passed
63.898s; after two pre-execution fixes,6 targeted tests passed0.222s.
No new fit, module or temporal loss. Source checkpoints unchanged.

Protocol: INTERVAL_DIAGNOSIS_V1.md. Raw results: results/interval_diagnosis_v1.
Additional times on SAME32 existing development trajectories per PDE are NEW
diagnostic access, not extra training labels or independent confirmation.
Reserved trajectory sets remain unopened. No selected corrected model exported.

## All arms

Three-seed means. Field and time errors at four primary centers; interval error
and endpoint repair average both windows/all4 centers. Do not compare these
cohorts/times directly to the prior200-case supervised-time primary.

| PDE | Coordinate | Smoothing | Midpoint field | Midpoint resolvedut | Interval increment | Endpoint field | Fixed repair field |
|---|---|---|---:|---:|---:|---:|---:|
| Allen-Cahn | linear | none | .035076 | .215073 | .215037 | .035001 | .032876 |
| Allen-Cahn | linear | low | .035876 | .217539 | .217506 | .035794 | .033719 |
| Allen-Cahn | linear | full | .035390 | .216741 | .216704 | .035316 | .033140 |
| Allen-Cahn | periodic | none | .030986 | .192852 | .192823 | .030853 | .029247 |
| Allen-Cahn | periodic | low | .032440 | .197160 | .197132 | .032329 | .030690 |
| Allen-Cahn | periodic | full | .032250 | .197136 | .197108 | .032139 | .030510 |
| Burgers | linear | none | .275060 | .795348 | .745161 | .270510 | .344307 |
| Burgers | linear | low | .265815 | .787068 | .737582 | .262542 | .318389 |
| Burgers | linear | full | .267718 | .790057 | .739940 | .264005 | .327828 |
| Burgers | periodic | none | .278596 | .827939 | .773273 | .274312 | .355927 |
| Burgers | periodic | low | .269343 | .815061 | .762996 | .267045 | .329841 |
| Burgers | periodic | full | .267273 | .810635 | .760154 | .265383 | .325121 |

Allen exact midpoints.025/.0875/.1875/.375; secondary latepoint.475:
periodic/none field.046350 (from stored late_field), resolvedut.599472.
Burgers centers.05/.18/.38/.75: .18/.38 are+.005 from exact mids; late.95
linear/low resolvedut.995970. All intervals lie inside the training time range,
but midpoint/interval targets were never used for fitting. This is new-time,
reused-trajectory development evidence, not temporal extrapolation or untouched
trajectory validation.

## Reference reliability and actual interval mismatch

Allen native1024 dt.0000625/.000125 and independent512 references agree well.
Largest per-window empirical increment/RHS/quadrature/closure uncertainty across
allcenters2.55e-9..3.96e-8; primarymean6.87e-9. Model Simpson17/9 difference
~1.11e-7 versus periodic/none increment error.192823. The mismatch is not
explained by reference or model quadrature accuracy. Late increment error~.599.

Burgers native dt.01/.02 integral/closure comparisons give3.41%..5.08%
empirical uncertainty (primarymean4.07%). Model quadraturemean2.63e-5 versus
linear/low increment error.737582. Increment error is clearly substantial, but
native PDE-integral reference is less reliable; no precise fullband FD truth.
Native Fourier modes0..16 are projected BEFORE subsampling. Full coarse field
metrics and repairs use the declared256 point-sampled targets.

Common native-increment normalization:

| Strong control | Increment error energy | RHS-integral error energy | Signed cross |
|---|---:|---:|---:|
| Allen periodic/none | .054371 | .049259 | +.014801 |
| Burgers linear/low | .618222 | 1.093467 | -1.106186 |

State/operator and temporal increment errors coexist; pure increment dominance
does not hold everywhere. Burgers errors strongly cancel. A small PDE defect
cannot be interpreted as accurate evolution or guaranteed beneficial correction.
These exact algebraic terms are not positive causal field-error contributions.

## Predeclared screen: failed, not retuned

Each strong control passes22/24 primary center/window/seed checks, but BOTH
strict all-center/all-window/all-seed screens FAIL. Do not change thresholds or
drop unfavorable windows after outcomes.

- Allen periodic/none seed1 at center.0875: increment errors.096949/.096988 are
  below fixed.10 minimum; reference/quad/floor checks otherwise satisfactory.
- Burgers linear/low seeds0/1 at center.05,h.04: errors.370282/.374952 are below
  10 x empirical reference uncertainty.037945. Both failures preserved.

The screen is descriptive, not significance or a formal error bound. Failure
does not mean dynamics are accurate; eligibility cannot be presented as passed.

## Fixed integral endpoint repair: a useful positive AND negative result

The preregistered one-shot correction sets only low-mode endpoint increment to
the integral of predicted RHS, without true future inputs. Gamma1 was fixed,
all arms/seeds/windows retained; no trained module or selected postprocessor.

Allen periodic/none primary endpoint field.030853 -> .029247, mean5.204%
improvement. Three-seed averages all improve:
.037002 -> .034517; .026152 -> .024849; .029404 -> .028376.
But the earliest center.025 worsens for BOTH windows (three-seed means):
.027382 -> .029016 (h.005), .026477 -> .029754 (h.01).
Do not describe repair as stable for every interval/sample or universally valid.

Burgers linear/low .262542 -> .318389, mean21.272% degradation; all3 seed
averages degrade. This counterexample rejects blindly enforcing the same
integral correction across tasks. Existing prediction errors and cancellation
can make a physical-defect removal hurt field accuracy.

Repair is local and anchor-dependent: overlapping windows do not define one
consistent global prediction. No global rollout, corrected derivative accuracy,
noise robustness or independent benefit is established. Not a deployable new
operator from these results alone.

## Decision: what is worth a controlled follow-up?

No temporal/integral loss trained in this round. Keep strong controls unchanged.
Allen has a reasonable EXPLORATORY case for a small newly registered JOINT
interval-PDE comparison, not a pure time-derivative-only loss: reliable references,
real between-label mismatch and modest counterfactual field benefit. This is
scientific judgment despite a failed conservative screen, NOT a screen pass or
authority to delete early windows. Report the failed screen in that future study.
Burgers full-strength interval correction/constraint is not justified; defer
until reference/closure and error-cancellation limitations are addressed.

Future minimal comparison: same periodic FNO, data-only + existing low/full
curvature controls + uniform interval constraint. Only THEN test a continuous
reliability weighting, using TRAIN-INTERNAL discretization/quadrature diagnostics,
not developer benefit, fresh confirmation values or hard window deletion. Hold
labels, initialization, streams, optimization/compute budgets and early-stopping
rules fixed; assess field plus gradient/curvature/increment and cost. Extra
training reference access used for weights must be disclosed, not hidden as
unchanged data. Allen's numerical uncertainty is almost uniform/tiny here, so
it alone is a weak test for a reliability-weighting novelty claim.

Weak/integral and midpoint ideas already exist in
[Temporal Consistency Loss](https://arxiv.org/abs/2301.13262),
[VPINN](https://arxiv.org/abs/1912.00873) and
[PENCO](https://arxiv.org/abs/2512.04863).
Potential contribution lies in a well-defined reliability/identifiability rule
and independently demonstrated gains, not assembling known losses. Novelty is
unproven; broad literature search/ablation remains required before such a claim.

All independent-test reserves remain unopened. Midpoints accessed now are
development forever, not re-labeled unseen temporal validation in future work.
