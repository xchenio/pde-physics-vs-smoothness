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

CNN and FNO results are pending. Numerical source: `results/public_architecture_stages_v1`; source commit12d4c1e; selection seal8192e38. See `docs/ARCHITECTURE_STAGES_V1.md` for resume commands and limitations. No method novelty follows merely from using CNN/POD, FNO, or frozen-ridge residual learning.
