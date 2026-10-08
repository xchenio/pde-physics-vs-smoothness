# Parameter-matched encoders and nested POD rank

Development-only follow-up to the108-run architecture study. Protocol seal de5cc93; 54 planned fits, three paired seeds, fixed final budgets, same correction600/representation400/development200 split. No new confirmation opened. Spectral and periodic-convolution encoders have the same nominal real parameter count at each rank:147696/160000/184608 for ranks16/32/64. Pooling/readout and other training settings are shared. Compute, initialization, receptive field and effective degrees of freedom are not matched.

## Completed Burgers controls

| Task | Rank | Spectral dev | Matched convolution dev | POD oracle |
|---|---:|---:|---:|---:|
| Original | 16 | 0.273799 | 0.373109 | 0.110922 |
| Original | 32 | 0.288870 | 0.393550 | 0.070985 |
| Original | 64 | 0.296425 | 0.398890 | 0.041327 |
| Independent | 16 | 0.302838 | 0.380088 | 0.120144 |
| Independent | 32 | 0.318006 | 0.391903 | 0.075410 |
| Independent | 64 | 0.322749 | 0.394451 | 0.042819 |

Means across three seeds; all models use direct coefficient prediction. Spectral encoding improves over the specified parameter-matched convolutional control at each rank. Increasing rank reduces training error and projection error but worsens final development error on both partitions. The selected rank is16 for both encoders. Spectral/POD remains worse than prior direct full-field FNO (0.195846/0.220706), despite more nominal parameters. This is evidence against simple parameter-count or insufficient-rank explanations of that gap, not proof of a single causal bottleneck. The POD readout differs from the full-field shared pointwise decoder; training objectives also differ by output projection.

Allen-Cahn is in progress. Its POD oracle falls from0.025842 at16 to0.001480 at32 and0.00000613 at64; this is attainable projection error with true coefficients, not a predictor. A rank increase can only help deployment if coefficients are predicted accurately enough.

No architecture novelty or independent generalization claim is established. All previous continuous-shrinkage/all-mode baselines remain archived. Numerical source: results/public_spectral_pod_v1; protocol: docs/SPECTRAL_POD_V1.md. Partial audit replayed36 completed runs;55 regression tests passed.
