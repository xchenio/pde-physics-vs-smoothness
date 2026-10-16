# Fixed uniform interval consistency under a reshuffled development split and sparse time labels

Exploratory follow-up, 2026-10-16. All 24 fits and their audit are complete.
This is development evidence, not an independent confirmation or a new
architecture. Earlier failures and the degenerate reliability-weight experiment
are retained, not reinterpreted as successes.

## Design fixed before training

We retained the periodic-coordinate 34,209-real-parameter FNO and the previous
uniform joint interval objective, including its Simpson5 rule, modes0–16,
half-widths .005/.01 and initial penalty/data gradient ratio .3. No reliability
map, tolerance search, numerical-reference solves or new network modules were
introduced. Both the predicted endpoint rate and predicted Allen–Cahn RHS
remain trainable in the same joint constraint.

A deterministic permutation (seed2026101601) repartitioned the previously
consumed 300 Allen trajectories into 100 training and 200 development cases.
The new development subset contains 68 former training cases. Consequently,
all six warm models were trained from scratch, without old model, optimizer
or normalization state. Repartitioning previously consumed cases does not
produce an unseen confirmation dataset or remove adaptive-design dependence.

Dense supervision retained times `[0,.05,.125,.25,.5]`; sparse supervision used
only `[0,.5]`. Each condition and seed received its own 1,500-step warm, followed
by paired 750-step continuations for data only, full-frequency curvature and
uniform interval consistency. Three seeds, a fixed final checkpoint and the
same schedules were used throughout: no early stopping or parameter search.
All continuation arms computed the same penalty forwards; backward work differs.

The raw integrity-checked caches contain all five fields. Selection occurred
before fitting statistics: the sparse training engine received only its two
observed labels. Its target scale, energy-protected sample weights and probe
calibration never used omitted training frames or development targets. Observed
target standard deviations were .62932425 and .68579004 for dense and sparse,
respectively. Fixing a formula does not justify reusing a dense-data scale that
would leak omitted supervision into the sparse experiment.

## Field and structure results

Means over three model seeds are reported below. Dense primary scores use four
future times; sparse primary scores use the three omitted intermediate times.
Their different primary percentages must not be compared as a sparsity effect.

| Condition / continuation | Primary field | Common interior field | Primary gradient | Primary curvature |
|---|---:|---:|---:|---:|
| Dense / data only | .02980754 | .02554231 | .11059764 | .31122602 |
| Dense / full curvature | .03054326 | .02607461 | .11120644 | .29038237 |
| Dense / uniform interval | .02885937 | .02490436 | .10625808 | .29740449 |
| Sparse / data only | .12884243 | .12884243 | .27462137 | .67546866 |
| Sparse / full curvature | .12675533 | .12675533 | .24008005 | .53499381 |
| Sparse / uniform interval | .11631309 | .11631309 | .24856337 | .61127615 |

Uniform consistency improves the dense primary field mean by3.1810% and the
sparse primary mean by9.7245%. The registered descriptive marker (at least1%
mean improvement and all three seed directions positive) is met in both
conditions. This arbitrary marker is neither statistical significance nor
demonstrated application relevance.

For a valid descriptive cross-condition comparison, both conditions were also
evaluated at the same three interior times on the same200 development cases.
Relative field gains are2.4976% for dense and9.7245% for sparse, a sparse-minus-
dense interaction of+7.2269 percentage points. Per-seed interactions are+11.1999,
+4.3148 and+4.4164 points. Common-time gradient and curvature gains are2.3325%
and2.7605% in dense, versus9.4887% and9.5034% in sparse.

This supports the limited hypothesis that the fixed physical prior is more
useful under sparse time labels in this setup. It is an interaction between
complete training procedures, not causal isolation of an interval mechanism or
an independently replicated dataset effect. Sparse uniform error remains about
4.67 times dense uniform error on the same interior times; the constraint does
not replace the missing observations.

## Counterexamples and strong-control tradeoffs

Per-seed primary gains are2.4644%,5.9880%,1.3447% in dense, and13.2403%,8.7165%,
5.6574% in sparse. All directions agree, not their magnitudes. Dense cross-seed
standard deviation increases about40.00%; sparse decreases about27.14%.
Three-seed outcome dispersion is not within-training fluctuation or evidence
of a generally more stable optimizer.

Every condition/seed pair worsens initial-condition field error. Mean IC errors
increase2.2557% in dense and6.7627% in sparse. Dense error at time.05 also rises
1.2028%; later-time average gains do not erase these failures. Sparse interior
time means improve7.8675%,9.3174%,10.9302%, and its final-time mean improves
2.0427%; this is not an all-case or temporal-extrapolation claim.

Full curvature remains a necessary strong baseline. Dense full curvature has
lower four-future curvature error (.29038237 versus.29740449). Sparse full
curvature has lower gradient (.24008005 versus.24856337) and curvature
(.53499381 versus.61127615), although uniform has lower field error. Uniform
therefore does not Pareto-dominate spatial smoothing or establish optimality
over tuned regularizers. Common-time and four-future structure scores must
also remain distinguished.

## Audit, cost and claim boundary

The regression suite passed125 tests in113.049 seconds, including CPU/CUDA
exact interruption/resumption and end-to-end sparse omitted-label poisoning:
replacing omitted training fields with extreme values or NaNs leaves the
observed targets, calibration, model/optimizer and RNG states unchanged.
The audit verifies all24 records,24 stream replays,6 fresh warm lineages and
calibration replays,3 cross-condition initialization pairs, observation-only
normalization proofs and all200-case final field/structure predictions.
Shared Fourier/loss helpers are implementation replays, not a second PDE solver.

New warm optimizer-step segments total93.2449s, continuations438.4390s,
total531.6839s for22,500 updates. Warm cost is counted once per fit, not per
fork. Evaluation, calibration, test/audit and other untimed work are excluded.
Uniform continuations cost about22.02% more than the shared-forward data-only
control in dense and17.96% more in sparse; this is not a minimum-cost or
FLOP-matched training benchmark. Recorded final-model last-segment peak allocated
CUDA memory spans107.92–135.76MiB, not cumulative resumed-run/device peaks.
Direct batch16-by-five-time inference medians span approximately1.85–2.69ms
after three warmups and20 CUDA-synchronized repeats, excluding transfers and
postprocessing. Identical architectures do not demonstrate a speedup from
these timing differences.

The completed runner was repeated:24 fits skipped,122 artifact hashes unchanged.
Both workers exited without stderr. No reserved confirmation was read, no new
trajectory/reference was generated, and no winning predictor was exported.

Integral/weak-form physical constraints have prior art, including
[VPINNs](https://arxiv.org/abs/1912.00873). This replication neither establishes
methodological novelty nor validates heterogeneous reliability weights, which
were absent here and degenerate in the previous pilot. It does not establish
noise, spatial-resolution, cross-PDE, independent-data or sample-count gains.
There is still no matched periodic-FNO pointwise-PDE/AD-loss arm, so superiority
over pointwise physical constraints or PINN-style training is not supported.

The next proposed comparison is a separately registered, same-architecture,
same-observation-budget pointwise-PDE control, retaining uniform consistency
and full curvature before deciding whether to authorize genuine confirmation.
No such expansion is executed here. More network modules or manufactured
reliability contrast are not warranted by these results.

[Complete Chinese report and artifacts](../docs/ALLEN_INTERVAL_REPLICATION_V1_RESULTS.md).
