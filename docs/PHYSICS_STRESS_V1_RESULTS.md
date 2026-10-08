# Fixed-architecture stress/second nonlinear PDE results (2026-10-09)

48/48 fits COMPLETE:12 condition-specific observation-only warm starts plus36
paired continuations. Seal600273d preceded training. Same TimeFNO1d34209 real
parameters,100 trajectories, three seeds,2250 total steps; transferred penalty
gradient ratio0.3, no per-condition tuning, all final outcomes retained.
All48 checkpoints, observed-label/noise identities, predictions, regenerated
batch/query hashes, fork starts/calibrations and group summaries were replayed.
No training worker remains; no new confirmation was opened.

| Condition | No penalty | Resolved PDE | Resolved curvature | PDE gain vs none | Curvature gain vs none |
|---|---:|---:|---:|---:|---:|
| Burgers sparse times | 0.644463 | 0.603849 | 0.589233 | 6.302% | 8.570% |
| Burgers 10% noisy future labels | 0.273533 | 0.270520 | 0.267287 | 1.102% | 2.284% |
| Burgers sparse + noisy labels | 0.648667 | 0.611887 | 0.599397 | 5.670% | 7.596% |
| Nonlinear Allen-Cahn dense clean | 0.043007 | 0.042646 | 0.043966 | 0.839% | -2.230% |

Lower field error is better. Sparse primary averages the three intermediate
times OMITTED within this fitting run; noisy/dense primary averages four
noninitial times. These numbers are not directly comparable across conditions
as a noise-effect estimate because primary times differ. Noise affects future
TRAINING LABELS only, not inference inputs, and is one fixed corruption shared
by three fitted seeds. Omitted training-time values do not affect corruption
amplitude, normalization, weighting or calibration. No dense-clean pretrained
model is reused for sparse/noisy warm starts.

## Interpretation and limitations

On Burgers, physical constraints help compared with no regularization, but do
not exceed the strong curvature control under any of the three predeclared
stress conditions. Sparse-time improvement alone therefore does not isolate
equation-specific information. The prior clean-confirmation conclusion remains
unreversed; do not claim physics uniquely improves noise/sparse supervision.

Allen-Cahn is a genuinely second NONLINEAR equation, newly generated synthetic
trajectories with independent train/dev IDs and dedicated solver/residual QA.
The physical arm is better than both no-penalty and curvature in each of the
three paired seeds. Mean benefit versus no penalty is only0.839%, not yet a
confirmed substantial effect; a1% gate like the previous confirmation would
not pass on this descriptive development mean. Curvature strength0.3 is a
transferred Burgers setting, not an optimally tuned Allen-Cahn baseline. Do not
claim universal physics superiority or a second-PDE independent confirmation.

All Burgers development trajectories/times were used in earlier design. Fresh
Allen-Cahn dev trajectories are evaluation data, not the reserved confirmation;
all intermediate training progress was monitored on them. No confirmatory
interval or significance assertion is made for these development summaries.
Both prior500 Burgers sets remain consumed. Allen-Cahn reserved IDs200300..
200799 are NOT generated/evaluated here.

## Derivative diagnostics (exploratory secondary, not selection criteria)

Spectral derivatives are evaluated on the full native output grid. Denominators
are protected by max(true derivative norm,0.001 clean TRAIN INITIAL derivative
RMS*sqrtN,1e-12), independent of noisy/omitted future labels. Diagnostics were
added during the run, not predeclared primary tests; no model selection uses them.

| Condition/metric | No penalty | PDE | Curvature |
|---|---:|---:|---:|
| Burgers sparse gradient error | 1.104427 | 1.067364 | 1.043912 |
| Burgers sparse second-derivative error | 1.386107 | 1.361684 | 1.345179 |
| Allen-Cahn gradient error | 0.210859 | 0.210184 | 0.213937 |
| Allen-Cahn second-derivative error | 3.694110 | 3.704976 | 3.699850 |

Allen-Cahn field accuracy does NOT imply accurate curvature. PDE slightly
improves field/gradient while its second-derivative diagnostic is slightly worse;
the scalar primary cannot certify multiscale structure or physical fidelity.
These observations motivate checking high-frequency errors and denominator
sensitivity, not claiming a proved artifact mechanism from these aggregates.

## Next decision

Keep architecture frozen and retain no-penalty and curvature controls. Before
adding modules or spending a new holdout, diagnose time-derivative error and
frequency-resolved field/derivative error on existing development data, particularly
Allen-Cahn high-frequency curvature. Separate discretization consistency from
regularization bias; any proposed scale-aware physical/closure correction must
be tested against equally strong smoothing. After a concrete rule is fixed,
register truly unused-ID confirmation. No further weight search on prior500.

This round needs no public download. All solver-generated development chunks and
binary model/optimizer checkpoints persist locally (external_data/checkpoints
excluded from Git). Console logs and curated results are versioned. Training
times include a partial replay-audit competing for GPU, so are descriptive only;
same forward architecture is not a controlled training-efficiency result.
