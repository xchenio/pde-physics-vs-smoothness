# Fitting and architecture diagnosis (development evidence)

This follow-up supersedes any assumption that frozen-ridge nonlinear residuals are generally preferable to direct coefficient prediction. It does not supersede historical measurements, which remain archived.

## Protocol

Three nonlinear tasks: public Burgers original/independent partitions and locally generated nonlinear Allen-Cahn. A fixed rank-16 POD, frozen linear anchor and 600 correction training samples are shared. Stage 1 compares a one-hidden-layer width-32 MLP under residual penalties 0, 0.1, 1 and 1500/6000-update cosine schedules, plus direct coefficient prediction with penalty0 at both budgets. Three seeds, fixed final evaluation; 72 fits. Selection uses the same 200-row development partition, not unseen confirmation data.

Subsequent CNN/POD and full-field FNO transfer both selected residual and direct settings without additional architecture-specific tuning. CNN has8848 real parameters versus8752 for MLP; FNO has136481. This is not an equal-compute or fully tuned architecture ranking.

## Stage 1: fitting is not generalization

| Task | Frozen linear dev | Selected MLP residual dev | Selected MLP direct dev |
|---|---:|---:|---:|
| Burgers original | 0.373252 | 0.363701 ± 0.001892 | 0.369034 ± 0.002815 |
| Burgers independent | 0.409382 | 0.402240 ± 0.001267 | 0.412777 ± 0.001916 |
| Allen-Cahn | 0.458885 | 0.287272 ± 0.008826 | 0.094198 ± 0.000262 |

Both Burgers tasks select residual penalty1/1500 and direct1500. Longer training decreases training error but worsens final development error for every tested Burgers penalty. Allen-Cahn selects residual penalty0.1/6000 and direct6000. Direct coefficients outperform the entire tested residual family on Allen-Cahn, including penalty0; removing the correction penalty alone does not explain the difference. Mapping target and initialization differ together, so this is not a clean causal isolation of either one.

The previous Allen-Cahn residual gain over linear regression is therefore insufficient evidence for preferring residual learning: a simple direct nonlinear baseline is much stronger on this development partition. All choices are development-adaptive and need a locked, genuinely unused confirmation before claims of generalization. No newly opened confirmation is included here; all previously consumed sets remain consumed.

## Architecture stages

CNN stage completed (18 fits; archived efe95e4):

| Task | CNN residual dev | CNN direct dev |
|---|---:|---:|
| Burgers original | 0.367876 ± 0.000291 | 0.389238 ± 0.001264 |
| Burgers independent | 0.401304 ± 0.000833 | 0.423261 ± 0.001015 |
| Allen-Cahn | 0.063771 ± 0.001256 | 0.046134 ± 0.000290 |

The almost parameter-matched CNN provides no consistent Burgers improvement, but substantially improves Allen-Cahn under the same POD basis and transferred settings. Direct CNN remains better than residual CNN on Allen-Cahn. This is task-dependent evidence for spatial inductive structure, not a universal superiority claim. POD projection oracle development relative L2 is0.110922/0.120144/0.025842 respectively; it is a lower bound within each fixed-POD affine output space, not for full-field models.

FNO stage completed (18 fits):

| Task | FNO residual dev | FNO direct dev | Direct FNO projected to same POD |
|---|---:|---:|---:|
| Burgers original | 0.251721 ± 0.002890 | 0.195846 ± 0.007377 | 0.192454 |
| Burgers independent | 0.276705 ± 0.004599 | 0.220706 ± 0.005479 | 0.217435 |
| Allen-Cahn | 0.043046 ± 0.000211 | 0.003315 ± 0.000286 | 0.026098 |

Post-hoc projection is a diagnostic, not a trained FNO/POD candidate or an independent validation. Burgers gains largely remain when FNO predictions are projected back into exactly the same affine POD space, so decoder freedom alone cannot explain the difference. It does not isolate spectral structure from parameter count or optimization. Allen-Cahn projected FNO approaches the POD oracle0.025842, while the full-field prediction is far below it: rank16 limits attainable accuracy once mapping is strong enough. Projection can reduce Burgers error by removing prediction components outside the basis; its error is not required to exceed the unprojected predictor's error.

All108 new runs have been replay-audited and52 regression tests pass. Numerical source: `results/public_architecture_stages_v1`; source commit12d4c1e; selection seal8192e38. Full tables: `docs/ARCHITECTURE_STAGES_V1_RESULTS.md`. No new confirmation was opened; three development seeds and transferred settings do not establish a fully tuned architecture ranking. No method novelty follows merely from using CNN/POD, FNO, or frozen-ridge residual learning.

## Revised research priority

Retain old continuous/all-mode controls, but stop treating small modal-residual gains as the main contribution. First isolate encoder and decoder: spectral encoder with the same POD output, a parameter-matched non-spectral control, and rank16/32/64 selected using training/development only. Compare accuracy, measured compute and sample efficiency against direct full-field FNO. If an efficient low-rank model is competitive, lock its rule and all controls before genuinely unused confirmation and parameter/distribution shift tests. Efficient adaptive representation is a research hypothesis, not established novelty or success. No further architecture fitting is part of this completed108-run round.
