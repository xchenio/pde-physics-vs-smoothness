# Same-node pointwise physics weakens the case for an interval-specific advantage

Exploratory follow-up, 2026-10-16. All 24 registered continuations and their
audit are complete. The interval and pointwise controls produce very small
observed prediction differences whose direction changes with the supervision
condition. This is not evidence for a stable interval advantage or a statistical
equivalence result. Earlier interval gains over data-only fitting must not be
recast as evidence for an innovative integral method.

## The missing matched control

The preceding development replication compared uniform joint interval
consistency with data-only fitting and full-frequency curvature smoothing. Its
positive mean field results did not establish an advantage over pointwise PDE
regularization. This study adds that comparator without changing the periodic
FNO, output head, observed data, warm state, query windows or training budget.

The equation remains periodic Allen–Cahn, `u_t=.01u_xx+5u(1-u²)`, with spatial
period two and `T=.5`. The FNO retains width 16, 16 spectral modes, four layers
and 34,209 real parameters. The pointwise control is pointwise in **time** but
projected onto spatial Fourier modes 0–16. It is not a full-frequency spatial
residual, an unrestricted PINN baseline or a new architecture.

Both physics objectives use the same five nodes
`t_j=c+h*[-1,-.5,0,.5,1]`, the original alternating half-widths `.005/.01`, and
Simpson weights `alpha=[1,4,2,4,1]/12`. Let `F_hat` denote the existing spectral,
4x-dealiased Allen RHS and `E(v)=|v_0|²+2*sum_{k=1..16}|v_k|²`. The new objective
is

`r_j=(T/sigma_observed_train)*(u_hat_t(t_j)-F_hat(u_theta(t_j)))`,

`L_pointwise=mean_cases(sum_j alpha_j*E(r_j))`.

Time JVPs hold the initial input and spatial grid fixed; their parameter
gradients remain differentiable during training. The retained interval objective
instead squares the joint endpoint-rate/integrated-RHS defect:

`R=(T/sigma_observed_train)*[(u_hat(c+h)-u_hat(c-h))/(2h)-sum_j alpha_j*F_hat(u_theta(t_j))]`,

`L_interval=mean_cases(E(R))`.

Thus averaging squared point residuals is not confused with squaring an average.
The independent initial penalty/data gradient-norm ratios are fixed at `.3`
using the same observed-training warm probe and frozen thereafter. Matching this
rule does not mean using identical numerical multipliers or fully tuning either
loss. No reliability map, tau, solver target or new supervision is introduced.

## Data scope and training accounting

This study reuses the preceding fixed split of the already-consumed 300 cases:
100 training and 200 development trajectories from IDs `200000..200299`.
The current development subset contains 68 cases that belonged to a still
earlier training partition. All six permissible warm models had therefore been
trained from scratch for the current split in the preceding replication; only
those current-split full-state warm checkpoints are reused here. No older,
incompatible warm model is inherited.

Dense supervision observes `[0,.05,.125,.25,.5]`; sparse supervision observes
only `[0,.5]`. Although complete cached trajectories are physically read for
integrity and reindexing, training statistics and fitting receive selected labels
only. The observed-target scales remain `.6293242455978698` and
`.6857900440848712`, respectively. The earlier end-to-end omitted-label poison
protections remain in force. Model query nodes are not additional true labels.

There are four objectives per condition—none, full curvature, interval and
pointwise—each with three seeds. Every new continuation runs 750 steps, restoring
its own condition/seed's model, optimizer and random states from the same warm.
These 24 continuations contribute 18,000 new optimizer steps. The six reused
1,500-step warms contribute 9,000 historical steps, not new training. Final
checkpoints are fixed; no strength, window, seed, time or checkpoint is selected
using these outcomes. All objectives compute the same penalty forwards, while
their selected backward work differs.

## The direct interval–pointwise comparison

The dense primary metric averages four future times `.05,.125,.25,.5`; the
sparse primary metric averages the three omitted times `.05,.125,.25`. Those
condition-specific primary metrics cannot be directly compared with each other.
Within each condition the interval and pointwise definitions are identical.
All scores below use the same 200 development trajectories and three-seed means.

| Supervision | Interval primary | Pointwise primary | Interval minus pointwise | Relative signed change |
|---|---:|---:|---:|---:|
| Dense | .02885936944 | .02885891553 | +.00000045392 | +.0015729% |
| Sparse | .11631309294 | .11632142866 | −.00000833572 | −.0071661% |

Negative changes mean lower interval error. Dense primary paired changes for
seeds 0/1/2 are `+3.22494e-7`, `+8.73607e-7`, `+1.65651e-7`: pointwise is lower
in all three. Sparse changes are `−9.27391e-6`, `−1.61559e-5`, `+4.22686e-7`:
interval is lower in only two of three. The signed magnitudes are small, and
neither the condition reversal nor the unfavorable seed may be hidden by
reporting only interval-versus-none gains. No equivalence margin, power analysis
or statistical equivalence test was registered or performed.

The complete primary field/structure means retain both strong controls:

| Condition | Objective | Primary field | Primary gradient | Primary curvature |
|---|---|---:|---:|---:|
| Dense | None | .02980754 | .11059764 | .31122602 |
| Dense | Full curvature | .03054326 | .11120644 | .29038237 |
| Dense | Interval | .02885937 | .10625808 | .29740449 |
| Dense | Pointwise | .02885892 | .10625728 | .29740175 |
| Sparse | None | .12884243 | .27462137 | .67546866 |
| Sparse | Full curvature | .12675533 | .24008005 | .53499381 |
| Sparse | Interval | .11631309 | .24856337 | .61127615 |
| Sparse | Pointwise | .11632143 | .24858345 | .61132896 |

For a common temporal evaluation, both conditions also use the same three
interior times `.05,.125,.25`:

| Condition | Objective | Interior field | Interior gradient | Interior curvature |
|---|---|---:|---:|---:|
| Dense | None | .02554231139 | .08483010 | .24908317 |
| Dense | Full curvature | .02607461145 | .08667182 | .24402349 |
| Dense | Interval | .02490435641 | .08285142 | .24220711 |
| Dense | Pointwise | .02490388441 | .08285149 | .24220735 |
| Sparse | None | .12884243047 | .27462137 | .67546866 |
| Sparse | Full curvature | .12675533341 | .24008005 | .53499381 |
| Sparse | Interval | .11631309294 | .24856337 | .61127615 |
| Sparse | Pointwise | .11632142866 | .24858345 | .61132896 |

Dense common-time interval field error is approximately `.0018953%` higher than
pointwise; the common gradient and curvature rankings differ by very small
amounts in the opposite direction. Sparse common-time differences equal its
primary differences. No universal structural ranking follows. Full curvature
retains the lowest dense primary curvature, and lower sparse gradient and
curvature than either physics objective. Better physics-regularized field means
do not imply Pareto domination of spatial smoothing.

Both physics procedures improve the condition-specific field means over none
by approximately 3.18% under dense labels and 9.72% under sparse labels. On the
common interior times, the interval gains are 2.4976% and 9.7245%, with closely
matching pointwise values. These observations are compatible with a benefit
shared by the tested physical constraints under this training procedure, not a
unique integral contribution or proof of a general causal mechanism. They do
not turn three model seeds into independent data replications.

## Time-local counterexamples remain

At the initial time, dense field error rises from `.03678793` for none to
`.03761778`/`.03761797` for interval/pointwise. At `.05`, it rises from
`.02224878` to `.02251639`/`.02251611`. Under sparse labels, initial-time error
rises from `.02578026` to `.02752370`/`.02752130`. Thus even the shared field
benefits over none are not uniform across times. Full-curvature tradeoffs and
all five time-wise field/gradient/curvature scores remain in the audit/report;
no early time is removed to improve an aggregate.

The dense primary seed standard deviations are `.00223584` and `.00223621`
for interval and pointwise, versus `.00159704` for none. Sparse values are
`.01946208`/`.01946777`, versus `.02671284`. Opposite changes across conditions
do not identify a training-stability mechanism, and cross-seed dispersion is
not a measurement of within-run optimization fluctuation.

## Known algebra, not a new method

The six fixed warm models' first 16 observed-training probes verify

`sum_j alpha_j*E(r_j)=E(sum_j alpha_j*r_j)+sum_j alpha_j*E(r_j-r_bar)`.

Here `r_bar=sum_j alpha_j*r_j`. This is the known weighted mean/variance identity.
Its observed within-window
variance spans approximately `6.18e-5`–`2.70e-4` on those probes; it is a diagnostic,
not a newly fitted penalty, a success threshold or a causal allocation of field
error. Moreover, an endpoint increment rate need not equal the Simpson average
of the model's AD time derivative. Their normalized gap RMS is approximately
`3.36e-6`–`3.97e-6` on these probes. The gap is reported, not selected on. It
may combine Simpson quadrature error with float32 forward/JVP and endpoint
subtraction roundoff; it is not an isolated quadrature-truncation estimate. It
prevents using Jensen's inequality to assert that the old endpoint interval
loss is always bounded above by this pointwise loss. The small observed probe
gaps also do not establish an identity for all trajectories, widths or models.

## Reproducibility and measured costs

All 24 prediction/stream/checkpoint records, six calibration replays, allowed
warm lineages and registered training-probe algebra checks pass. All 18
none/full/interval controls reproduce their preceding final tensors exactly.
These repeated controls are reproducibility evidence, not 18 new independent
observations. The full regression suite passes 130 tests in 130.826 seconds.
The source/protocol/plan seal preceded training. Shared equation and metric
replays are not an independently derived numerical solver or an independent
scientific verification.
The completed runner replay skips all 24 fits, preserves the hashes of 122
artifacts and validates 54 parent-lineage hashes. Training and audit workers
have exited with empty stderr logs; no background fit remains pending.

Persisted optimizer-step time totals 1,147.324 seconds for the 24 new
continuations. The preceding reused warm cost was 93.245 seconds and is reported
separately. Mean continuation times in seconds are:

| Condition | None | Full curvature | Interval | Pointwise |
|---|---:|---:|---:|---:|
| Dense | 44.207 | 47.559 | 47.810 | 52.144 |
| Sparse | 43.239 | 47.572 | 47.723 | 52.187 |

These timings exclude calibration, evaluation, I/O and audit. Equal optimizer
steps and five nodes are not equal FLOPs or end-to-end work: all arms execute
all three penalty forwards, with differing backward/JVP costs. The data-only
cost is therefore not a minimum-cost production data-only baseline. Last-segment
peak allocated CUDA memory spans approximately 860.2–910.9 MiB, not cumulative
resumed-run peaks or whole-device usage. Median direct inference batches of
16 trajectories at five times range approximately 1.81–2.77 ms, excluding
transfers/postprocessing; all models have the same architecture. These timing
fluctuations do not establish an inference speedup or require time JVPs at
deployment.

## Revised claim and next decision

The matched comparator weakens an interval-specific new-method claim. The
registered narrow windows and fixed `.3` calibration ratio yield tiny,
condition-dependent interval-versus-pointwise differences, without a stable
predictive superiority claim. It would be inappropriate to widen windows,
change strengths or remove unfavorable times retrospectively to obtain a win.
Neither arbitrary pointwise PINNs nor fully tuned physics regularizers have
been ruled out by this restricted projected-time-pointwise comparison.

The appropriate next action is to freeze the general physics controls and
organize the diagnostic, negative and tradeoff evidence. The old failed screens
and all-one reliability contrast remain preserved. Any genuinely unused
confirmation requires separately locked comparisons and further authorization;
none was opened here, and no winner was exported. This study provides no new
noise, cross-resolution, temporal-extrapolation or second-PDE validation, and
does not justify adding network modules or retuning this experiment's windows.

[Complete results and limitations](../docs/ALLEN_POINTWISE_CONTROL_V1_RESULTS.md).
[Raw audit and all paired evidence](../results/allen_pointwise_control_v1/audit.json).
