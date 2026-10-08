# Spectral/POD development results

2026-10-08. Three seeds; mean per-sample relative L2, lower is better. Development only. Fixed final checkpoints; no new confirmation.

| Task | Encoder / rank | Train | Dev mean ± seed SD | POD oracle | Nominal real parameters | Optimization seconds |
|---|---|---:|---:|---:|---:|---:|
| burgers_original | spectral_pod_r16 | 0.156608 | 0.273799 ± 0.008761 | 0.110922 | 147696 | 16.9 |
| burgers_original | local_pod_r16 | 0.167074 | 0.373109 ± 0.006276 | 0.110922 | 147696 | 12.8 |
| burgers_original | spectral_pod_r32 | 0.129824 | 0.288870 ± 0.009547 | 0.070985 | 160000 | 16.2 |
| burgers_original | local_pod_r32 | 0.140194 | 0.393550 ± 0.005244 | 0.070985 | 160000 | 12.4 |
| burgers_original | spectral_pod_r64 | 0.110772 | 0.296425 ± 0.003222 | 0.041327 | 184608 | 16.5 |
| burgers_original | local_pod_r64 | 0.121883 | 0.398890 ± 0.001471 | 0.041327 | 184608 | 12.2 |
| burgers_independent | spectral_pod_r16 | 0.154339 | 0.302838 ± 0.005612 | 0.120144 | 147696 | 16.4 |
| burgers_independent | local_pod_r16 | 0.162628 | 0.380088 ± 0.005585 | 0.120144 | 147696 | 12.1 |
| burgers_independent | spectral_pod_r32 | 0.126719 | 0.318006 ± 0.006458 | 0.075410 | 160000 | 16.1 |
| burgers_independent | local_pod_r32 | 0.136293 | 0.391903 ± 0.008212 | 0.075410 | 160000 | 12.3 |
| burgers_independent | spectral_pod_r64 | 0.107727 | 0.322749 ± 0.003234 | 0.042819 | 184608 | 16.3 |
| burgers_independent | local_pod_r64 | 0.116871 | 0.394451 ± 0.002568 | 0.042819 | 184608 | 12.0 |
| allen_cahn | spectral_pod_r16 | 0.026178 | 0.027097 ± 0.000395 | 0.025842 | 147696 | 63.6 |
| allen_cahn | local_pod_r16 | 0.026216 | 0.026244 ± 0.000040 | 0.025842 | 147696 | 49.1 |
| allen_cahn | spectral_pod_r32 | 0.002499 | 0.006965 ± 0.000837 | 0.001480 | 160000 | 62.0 |
| allen_cahn | local_pod_r32 | 0.002882 | 0.004486 ± 0.000211 | 0.001480 | 160000 | 47.8 |
| allen_cahn | spectral_pod_r64 | 0.001922 | 0.006762 ± 0.000606 | 0.000006 | 184608 | 65.3 |
| allen_cahn | local_pod_r64 | 0.002374 | 0.004215 ± 0.000146 | 0.000006 | 184608 | 45.0 |

## Historical controls (same development rows)

Previous three-seed means; full FNO has different output space and parameters; earlier small CNN/MLP are not parameter-matched to the new encoders.

| Task | Best prior MLP residual | Prior direct CNN/POD16 | Prior direct full FNO |
|---|---:|---:|---:|
| burgers_original | 0.363701 | 0.389238 | 0.195846 |
| burgers_independent | 0.402240 | 0.423261 | 0.220706 |
| allen_cahn | 0.287272 | 0.046134 | 0.003315 |

Times exclude evaluation/checkpoints and vary with load. Equal nominal parameter counts do not imply equal compute, initialization, effective degrees of freedom or receptive field. Ranks change readout parameter count. All development outcomes reported; no claim of architecture-specific optimum or method novelty.
