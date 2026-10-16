# A small Allen–Cahn interval-consistency gain and a degenerate weighting contrast

Exploratory follow-up, 2026-10-16. This addendum preserves the earlier failed
eligibility screens and historical experiments. It is not a submitted manuscript,
an independent confirmation, or evidence for a new architecture.

## Scope and paired design

A uniform joint interval constraint produced small development improvements with
the existing periodic-coordinate FNO. The proposed numerical-reliability weights,
however, all became exactly one in the implemented precision. The experiment
therefore distinguishes a potential benefit of the uniform constraint from an
unidentified benefit of heterogeneous reliability weighting.

The earlier frozen-model all-window eligibility screen failed. This subsequent
trial was separately authorized and registered as exploratory before its new
reference solves and continuations. Neither that screen nor a success threshold
was changed retrospectively to qualify these results.

We retained the 34,209-real-parameter FNO, the original 100 training trajectories,
and the five observed times `[0, .05, .125, .25, .5]` for the synthetic periodic
Allen–Cahn equation `u_t = .01 u_xx + 5 u(1-u²)` on a spatial period of two.
The same 200 development trajectories were reused. For each of three seeds, an
existing 1,500-step data-only warm checkpoint supplied the model, AdamW state and
random states to five 750-step continuations: data only, low-band curvature,
full-band curvature, uniform interval consistency, and reliability-weighted
interval consistency. No layers, channels, attention or fitted intermediate-time
labels were added. Final checkpoints were fixed; there was no early stopping,
seed selection, strength search or development-selected checkpoint.

The query stream, batch size 16, learning-rate schedule and training steps were
paired. The initial penalty/data gradient-norm ratio was fixed at .3 using the
same warm training probe, with the resulting multiplier held constant thereafter.
These are transferred controls, not proof of architecture-specific optimal tuning.

## Joint constraint and additional training information

For a derived interval center `c` and alternating half-width `h=.005` or `.01`,
the dimensionless low-mode rate residual was

`r_k = (T/s_train) * [(u_hat_k(c+h)-u_hat_k(c-h))/(2h) - Simpson5_average(F_k(u_theta))]`,

where `T=.5`, `s_train` is the original observed training-target standard
deviation, and `k=0..16`. Both the endpoints and the dealiased predicted Allen
RHS belong to the same trainable model. The uniform loss is the batch mean of
`|r_0|² + 2*sum_{k=1..16}|r_k|²`. This is not fitting a frozen oracle derivative
or replacing the predictor by a reference-dependent repair.

Candidate reliability weights used extra numerical solves for the first 16
training trajectories only. At five fixed centers, two widths and 17 modes,
the maximum of seven case-RMS numerical rate discrepancies defined `epsilon`.
The fixed rule was `w=tau²/(tau²+epsilon²)`, with
`tau=.001*s_train/T`, followed by temporal interpolation. It is a numerical
tolerance policy, not a calibrated uncertainty probability. Those solves are
additional training information and computational work, even though their
intermediate fields are not fitted supervision targets. No development or
reserved trajectory was used to construct the map.

## Development results and counterexamples

The following are means over three seeds. Field, gradient and curvature use the
original 200 development trajectories and four noninitial supervised times.
Increment error uses the already-consumed 32-trajectory interval references and
the first four registered centers; it is not a 200-case field metric.

| Continuation | Field | Gradient | Curvature | Interval increment |
|---|---:|---:|---:|---:|
| Data only | .03715870 | .11958300 | .33673974 | .19282287 |
| Low-band curvature | .03867859 | .12607679 | .33432885 | .19713169 |
| Full-band curvature | .03855306 | .12380561 | .31755783 | .19710809 |
| Uniform interval | .03654534 | .11759923 | .32957541 | .18970521 |
| Reliability-weighted interval | .03654534 | .11759923 | .32957541 | .18970521 |

Uniform interval consistency reduces mean field error by 1.6507%, gradient error
by 1.6589%, curvature error by 2.1276%, and interval-increment error by 1.6169%
relative to data only. Per-seed field improvements are 0.69294%, 4.04777% and
0.75282%. All directions agree, but their magnitudes do not: seed 1 supplies
68.62% of the total absolute field improvement. Three model seeds are not three
independent datasets or data partitions. Cross-seed field-error standard deviation
increases from .00711132 to .00751456; this neither establishes reduced training
fluctuation nor supports a training-stability claim.

There is no Pareto domination of the strong spatial controls: full-band smoothing
still has lower curvature error (.31755783 versus .32957541), although its mean
field error is higher. The transferred strength does not establish the best
achievable version of either regularizer.

On the reused midpoint references, mean field error falls from .03098636 to
.03066936. The separately reported late interval-increment error falls from
.59927116 to .58314540, but remains large. Improvements are not universal in time.
At the earliest center `.025`, mean endpoint field error worsens by 0.66449% for
`h=.005` and 0.60112% for `h=.01`. At the original initial time and `.05`, field
errors also rise slightly (.04265217 to .04292113 and .02525239 to .02534944).
These unfavorable observations remain part of the result; later-time or average
gains do not erase them. Lower closure residual alone is not improved prediction,
and this study does not establish temporal extrapolation or a unique temporal
cause of the remaining field error.

## The reliability comparison is degenerate

All 170 effective float32 weights are exactly one. The float64 values range from
.9999999990656812 to .9999999999991117, so the small numerical heterogeneity is
absent in the implemented weights. The final weighted and uniform models have
exactly identical tensors for each of the three seeds, not merely similar
aggregate scores.

Consequently, the two implemented objectives are the same treatment here. Their
equal scores are neither additional evidence for reliability weighting nor a
successful test of heterogeneous weighting. They also do not prove that such
weighting is ineffective on a different, genuinely uncertain numerical problem.
The five registered arms contain only four effective loss objectives; duplicated
weighted models are not additional independent seeds. Likewise, historical
control replays are reproducibility evidence, not new independent experiments.
The rule was not rescaled, thresholded or retuned to manufacture contrast after
this degeneracy was observed. Initial gradient calibration cancels a globally
constant positive weight in exact arithmetic; the verified all-one contrast is
also tensor-identical in this implementation.

## Audit and cost boundaries

All 15 final records passed source, reference, normalization, checkpoint,
optimizer/RNG, stream, calibration and prediction replay checks. The nine
data-only/curvature controls reproduce historical model tensors exactly. The
full regression suite passed 117 tests, including CPU/CUDA interrupted-resume
checks. Replaying shared equation and scoring helpers is not an independently
derived PDE solver or independent data verification. No reserved confirmation
set was opened.

Persisted numerical-reference compute time was 3.203 seconds. Timed optimizer-step
segments sum to 390.73 seconds for all 15 continuations, excluding the reused
4,500 warm steps, calibration, evaluation and other untimed work. Uniform
continuations average 27.18 seconds versus 23.22 seconds for data-only controls
(about 17.06% more). All arms computed the same extra penalty forwards, so this
is the shared-forward experimental protocol, not a minimum-cost data-only
implementation or an end-to-end benchmark. Recorded final-segment peak allocated
CUDA memory spans approximately 119.7–135.8 MiB; it is neither whole-device usage
nor a cumulative peak across interruptions. Median inference batches of 16
trajectories at five times took approximately 2.29–2.66 ms, excluding transfers
and postprocessing. Small timing differences between identical architectures do
not establish an inference-speed improvement.

## Claim boundary and proposed next step

The supported claim is a small, controlled development gain from the registered
uniform joint-interval constraint on this Allen–Cahn setup, together with a
documented degenerate weighting contrast. Weak/variational physics losses have
prior art, including [Variational Physics-Informed Neural Networks](https://arxiv.org/abs/1912.00873).
Integral losses, Simpson quadrature and this unchanged FNO do not by themselves
constitute a new architecture or establish methodological novelty.

This trial has no pointwise PDE/AD-loss comparator with the same periodic FNO,
warm checkpoint and 750-step continuation budget. Earlier pointwise-PDE results
used the linear-coordinate model and are not a matched comparator here. Thus
these results cannot establish that an interval constraint outperforms a
pointwise physics constraint or PINN-style training.

The next proposed work is an additional development split and a fixed sparse-time
test, retaining the data-only and strong spatial controls. These have not been
executed in this trial. Any changed split requires a warm model trained only on
its new training subset, with training-only normalization, calibration and
reference construction; moving old warm-training cases into validation would
leak information. Omitted supervision times must not enter normalization or
calibration. Repartitioning previously consumed cases still creates development
evidence, not a genuinely unseen confirmation. Only after locking the rule,
budget, outcomes and comparison criteria should genuinely reserved trajectories
be used once for confirmation. No present result establishes sparse-time,
noise, cross-resolution, cross-PDE or independently confirmed gains.

Full registered results and limitations:
[Allen interval pilot report](../docs/ALLEN_INTERVAL_PILOT_V1_RESULTS.md).
Raw audit: [all 15 records and controls](../results/allen_interval_pilot_v1/audit.json).
