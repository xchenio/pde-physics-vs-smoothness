# Conditional uncertainty shrinkage does not improve the strong controls

Source/protocol0e3217f, candidate sealbfd52cd, four new500-trajectory confirmation sets.
No neural models were retrained. All new sets are now consumed. The historical initial
plan at07d7ca4 failed preflight because of missing CV-seed metadata; the active plan_v2
repairs that key before any development outcome. Neither strength nor criteria were
adapted after development or confirmation.

For held-fold residual coefficients r and frozen branch correction delta, estimate
N=mean(w*r*delta)/256,D=mean(w*delta^2)/256. Old shrinkage clips N/D to[0,1]. New
shrinkage clips N/(D+1.96*s_N) to[0,1], where s_N is the standard deviation over2000
fold/seed/trajectory bootstrap replicates. This minimizes a quadratic risk plus an
uncertainty amplitude penalty; it is NOT a calibrated confidence/safety constraint.
Only three fitted folds/two probes exist; overlapping training sets and unrepresented
refitting uncertainty limit interpretation. No new mode threshold is introduced.

Strong controls retain all outputs and old continuous shrinkage. An additional global
scalar on old shrinkage matches the new rule's CV correction energy. Trajectory-only
uncertainty is descriptive. The locked main candidate falls back to no correction if
development improvement over linear is below1%. Sixteen confirmation comparisons
(four tasks versus linear/old/all/global) require>=1% relative gain and a Bonferroni
two-sided99.6875% paired-trajectory interval below zero, with10000 resamples.

| Confirmation task | All outputs | Old continuous | Raw uncertainty | Matched global |
|---|---:|---:|---:|---:|
| Original Burgers | 0.420848 | 0.419806 | 0.419982 | 0.419670 |
| Independent Burgers | 0.382781 | 0.382451 | 0.384878 | 0.384784 |
| Advection | 0.017884071348 | 0.017884069382 | 0.017884124078 | 0.017884132110 |
| Synthetic Allen-Cahn | 0.369091 | 0.369091 | 0.369091 | 0.369091 |

Advection deploys the locked no-correction fallback0.017884195185. No candidate passes
superiority against old continuous, all outputs or matched global shrinkage. On the
independent Burgers split uncertainty is0.6345% worse than old continuous, with adjusted
difference interval[0.001133,0.003818]. The original split has no confirmed gain even
over linear (mean1.4590% gain, interval[-0.018638,0.018631]). Independent Burgers and
Allen-Cahn still improve over linear by3.4022% and21.4176%, respectively, but Allen-Cahn
coefficients remain all1 and all nonlinear arms coincide. That is not uncertainty gain.

This negative result concerns the specified conditional moment penalty, not all possible
uncertainty modeling. Do not adjust its strength on these consumed samples or imply
its bootstrap standard deviation is calibrated epistemic uncertainty. The full evidence
and limitations are in [UNCERTAINTY_MODAL_V1](../docs/UNCERTAINTY_MODAL_V1.md).

## Next diagnostic rather than an architecture novelty claim

The existing fixed residual penalty1 has a nontrivial fitting bias: for unconstrained
coefficients, minimizing ||delta-r||^2+||delta||^2 yields delta=r/2. This is an exact
squared-objective statement, not a minimum of nonsquared relative L2, and the actual
network additionally shares weights and has finite optimization/capacity constraints.
On Allen-Cahn development, actual mean relative L2 is0.359048, the label-dependent
penalty1 free-coefficient solution is0.230826 and POD projection is0.025842. Training
counterparts are0.353639,0.229930 and0.026088. These oracle diagnostics are NOT deployable
models or predictions on unseen labels; they do not uniquely identify optimization error.

Before claiming architectural necessity, test penalty/budget and direct versus residual
coefficient mapping on identical POD axes. Then compare a spatially structured branch
with unchanged decoder and a full-field FNO at matched data and tuning costs. The
historical FNO pilot is encouraging but uses another protocol and cannot be pooled
as a present head-to-head result. Changing architecture alone is not novelty.
