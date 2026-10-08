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

## Completed Allen-Cahn controls

| Rank | Spectral dev mean ± seed SD | Matched convolution dev mean ± seed SD | POD oracle |
|---|---:|---:|---:|
| 16 | 0.027097 ± 0.000395 | 0.026244 ± 0.000040 | 0.025842 |
| 32 | 0.006965 ± 0.000837 | 0.004486 ± 0.000211 | 0.001480 |
| 64 | 0.006762 ± 0.000606 | 0.004215 ± 0.000146 | 0.00000613 |

Both encoders select rank64 by mean development error. Most rank benefit occurs between16 and32; the extra gain from32 to64 is small compared with its projection-oracle improvement. Convolution is stronger than the spectral encoder at all tested ranks on this task, unlike Burgers. None beats the old full-field FNO0.003315 ±0.000286 under these transferred settings. The rank64 models have184608 nominal real parameters versus136481 for full FNO, so no parameter-efficiency advantage has been demonstrated. Recorded optimization time is lower for the local model, but it is not an equal-compute benchmark or a latency measurement.

## Exact error decomposition and interpretation

Using the stored orthonormal basis, the audit verifies normalized squared Frobenius field error = projection error + coefficient error for every model. On rank16 spectral Burgers, coefficient mismatch contributes86.710%/86.332% of this squared error; on rank16 Allen-Cahn it contributes6.207% for spectral and2.076% for local. At rank32 Allen-Cahn it becomes94.415%/87.236%, and at64 it is approximately100%. These percentages refer to the squared Frobenius aggregate across seeds, not the main mean relative L2 metric and not causal attribution to an optimizer. They quantify the changing location of the error, not a certificate that a particular architecture can eliminate it.

## Next discriminating experiment (not executed)

Avoid another unconstrained architecture search. Keep the same FNO backbone and shared pointwise output head, comparing full-field training with the same output projected into a fixed POD space during training. This preserves trainable parameter count between those two arms and removes the present flattened positional coefficient head as a confound. Include the current pooled-head model as context, report train/development gaps, and retain direct full FNO and historical shrinkage controls. If a competitive accuracy/compute tradeoff emerges, lock choices before genuinely unused confirmation and parameter-shift tests. The role of shared spatial readout is a hypothesis, not an established cause or novel method.

No architecture novelty or independent generalization claim is established. All previous continuous-shrinkage/all-mode baselines remain archived. Numerical source: results/public_spectral_pod_v1; protocol: docs/SPECTRAL_POD_V1.md; complete tables: docs/SPECTRAL_POD_V1_RESULTS.md. Final audit replayed all54 runs, parameter counts, nested bases, projection identities, aggregates and rank selections;55 regression tests passed. Re-running the driver skips all54 completed runs. No new confirmation data were opened.
