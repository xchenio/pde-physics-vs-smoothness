# Strong FNO: physics versus matched smoothness (2026-10-08)

Development experiment complete:30 paired runs, three seeds, fixed2250 total
steps (1500 shared data-only warm start plus750 continuation). Same34209 real
parameters,100 training trajectories/five labels, initial weights/optimizer,
labeled batches and query-time streams. Zero residual head initialization is
preserved in the original warm start. All strengths and final checkpoints are
retained; no best checkpoint or seed selection.

| Selected family | Initial gradient ratio | Development error | Gain versus no penalty |
|---|---:|---:|---:|
| No penalty | 0 | 0.270949 | — |
| Resolved PDE | 0.3 | 0.267574 | 1.246% |
| Resolved gradient | 0.3 | 0.267647 | 1.219% |
| Resolved curvature | 0.3 | 0.265248 | 2.104% |

Three-strength grid0.03/0.1/0.3, training-only gradient calibration, fixed Fourier
band0..16 for all penalties. PDE is almost tied with gradient smoothing and worse
than curvature smoothing on reused development data. Thus physical regularization
has no demonstrated unique predictive advantage. These are matched-band controls,
not optimally tuned full-band smoothing; all selected strengths lie on the upper
search boundary, so do not describe the search as fully optimized.

All30 predictions, final hashes, batches, collocations and selection decisions
independently replayed. Do not compare2250-step results directly against the
historical1500-step FNO as if compute budgets were equal. Penalty forwards match,
but backward costs differ; training times are not controlled efficiency evidence.

## Confirmation registered, not yet evaluated

New500 trajectories are selected from1671 conservatively unused IDs after
excluding8329 historical IDs, including the previous consumed500. Twelve models
(all four selected families x three seeds) and five contrasts are locked before
opening labels: each penalty versus none, physics versus gradient, physics versus
curvature.10000 paired trajectory bootstrap, Bonferroni5 two-sided intervals,
minimum1% improvement and negative upper bound; conditional on three fitted seeds.
This is one-PDE, fixed-grid/seen-time confirmation, not second-PDE or extrapolation.
