# FNO shared-head projection development results

2026-10-08. Three seeds; fixed final checkpoints; mean per-sample relative L2. Same136481-parameter FNO and head in every current arm. Reused development data, no new confirmation.

| Task | Arm | Train | Dev mean ± seed SD | Optimization seconds |
|---|---|---:|---:|---:|
| burgers_original | full | 0.120503 | 0.195846 ± 0.007377 | 15.6 |
| burgers_original | pod16 | 0.158025 | 0.191759 ± 0.008412 | 15.7 |
| burgers_original | pod32 | 0.137961 | 0.188328 ± 0.008705 | 16.2 |
| burgers_original | pod64 | 0.126137 | 0.190009 ± 0.007556 | 15.7 |
| burgers_independent | full | 0.122561 | 0.220706 ± 0.005479 | 15.3 |
| burgers_independent | pod16 | 0.157206 | 0.218613 ± 0.006967 | 15.8 |
| burgers_independent | pod32 | 0.136878 | 0.216032 ± 0.008146 | 15.6 |
| burgers_independent | pod64 | 0.126244 | 0.215583 ± 0.005049 | 15.5 |
| allen_cahn | full | 0.002139 | 0.003315 ± 0.000286 | 59.1 |
| allen_cahn | pod16 | 0.026184 | 0.026170 ± 0.000049 | 62.5 |
| allen_cahn | pod32 | 0.002482 | 0.003653 ± 0.000202 | 62.2 |
| allen_cahn | pod64 | 0.001913 | 0.003207 ± 0.000226 | 61.7 |

## Same POD output space

| Task | Rank | Training projection | Full training then projection | Prior pooled coefficient head |
|---|---:|---:|---:|---:|
| burgers_original | 16 | 0.191759 | 0.192454 | 0.273799 |
| burgers_original | 32 | 0.188328 | 0.192368 | 0.288870 |
| burgers_original | 64 | 0.190009 | 0.194286 | 0.296425 |
| burgers_independent | 16 | 0.218613 | 0.217435 | 0.302838 |
| burgers_independent | 32 | 0.216032 | 0.218254 | 0.318006 |
| burgers_independent | 64 | 0.215583 | 0.218886 | 0.322749 |
| allen_cahn | 16 | 0.026170 | 0.026098 | 0.027097 |
| allen_cahn | 32 | 0.003653 | 0.003552 | 0.006965 |
| allen_cahn | 64 | 0.003207 | 0.003182 | 0.006762 |

The projected-training raw field is outside the deployed arm and its unsupervised complement is unconstrained. Gradients/objective change with projection. Previous pooled-head models differ in parameter count. Times exclude evaluation/checkpoints; they are load-dependent. All arms reported, no arm selected for confirmation, no novelty/independent validation claim.
