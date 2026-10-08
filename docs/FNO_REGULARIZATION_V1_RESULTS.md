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

## Fresh500 confirmation complete (NOW CONSUMED)

New500 trajectories are selected from1671 conservatively unused IDs after
excluding8329 historical IDs, including the previous consumed500. Twelve models
(all four selected families x three seeds) and five contrasts are locked before
opening labels: each penalty versus none, physics versus gradient, physics versus
curvature.10000 paired trajectory bootstrap, Bonferroni5 two-sided intervals,
minimum1% improvement and negative upper bound; conditional on three fitted seeds.
This is one-PDE, fixed-grid/seen-time confirmation, not second-PDE or extrapolation.

Selection/model/rule seal4d60f18 preceded target extraction. All12 model
predictions and five bootstrap decisions were independently replayed.

| Frozen family | Confirmation mean relativeL2 | Seed SD | Gain versus none |
|---|---:|---:|---:|
| No penalty | 0.290253 | 0.008013 | — |
| Resolved PDE | 0.286260 | 0.007680 | 1.376% |
| Resolved gradient | 0.286076 | 0.006990 | 1.439% |
| Resolved curvature | 0.282820 | 0.006706 | 2.561% |

| Registered difference (left minus right) | Mean difference | Adjusted paired interval | >=1% gain and upper<0 |
|---|---:|---|---|
| PDE - none | -0.003993 | [-0.004726,-0.003376] | pass |
| Gradient - none | -0.004177 | [-0.004843,-0.003602] | pass |
| Curvature - none | -0.007433 | [-0.008552,-0.006432] | pass |
| PDE - gradient | +0.000184 | [-0.000086,+0.000440] | fail |
| PDE - curvature | +0.003440 | [+0.002879,+0.004032] | fail |

All three penalties improve each paired seed relative to no penalty. PDE versus
gradient has mixed seed signs and an interval containing zero: no established
predictive advantage. Curvature is better than PDE in every paired seed and the
registered PDE-minus-curvature interval is wholly positive. These conclusions
are conditional on the three fitted seeds and the searched constraints/strengths;
they do not establish that all physics methods are inferior to smoothing.

The evidence supports modest regularization gains in this100-shot regime, NOT
a unique physics or hybrid-architecture innovation. The most useful new result
is the strong smoothness control: enforcing the governing equation is not needed
to explain the observed field-error gains. No after-confirmation strength search
or model selection was performed. Both previous500 and this new500 are consumed.
Under the frozen conservative registry,1171 IDs remain outside8329 history+500
new confirmation; a later protocol must rebuild its registry, not assume these
remain unused forever.

Next priority: retain the selected curvature arm as a mandatory strong baseline,
and test whether physics offers extra information under sparse TIME supervision
or noise, with predeclared stress settings. Keep the architecture fixed, record
field/gradient/temporal-derivative tradeoffs, and add a second nonlinear PDE before
claiming generality. Existing clean/seen-time confirmation cannot support these
claims. Do not add Transformer/CNN complexity to hide the missing PDE advantage.
