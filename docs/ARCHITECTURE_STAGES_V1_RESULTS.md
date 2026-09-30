# Architecture development results

Mean per-sample relative L2; lower is better. New runs: three seeds, fixed final checkpoints. Development selection only; no confirmatory claims.

| Task | Stage / setting | Train | Development mean ± seed SD | Real parameters | Optimization seconds |
|---|---|---:|---:|---:|---:|
| burgers_original | frozen linear | 0.397497 | 0.373252 | — | — |
| burgers_original | fit: residual_p0_b1500 | 0.349670 | 0.370425 ± 0.001720 | 8752 | 2.4 |
| burgers_original | fit: residual_p0_b6000 | 0.310076 | 0.386662 ± 0.007081 | 8752 | 9.9 |
| burgers_original | fit: residual_p0.1_b1500 | 0.350238 | 0.367432 ± 0.002983 | 8752 | 2.5 |
| burgers_original | fit: residual_p0.1_b6000 | 0.309885 | 0.378459 ± 0.006632 | 8752 | 10.8 |
| burgers_original | fit: residual_p1_b1500 | 0.359897 | 0.363701 ± 0.001892 | 8752 | 2.4 |
| burgers_original | fit: residual_p1_b6000 | 0.329798 | 0.365952 ± 0.004380 | 8752 | 11.9 |
| burgers_original | fit: direct_p0_b1500 | 0.350661 | 0.369034 ± 0.002815 | 8752 | 4.0 |
| burgers_original | fit: direct_p0_b6000 | 0.308595 | 0.386344 ± 0.000940 | 8752 | 15.9 |
| burgers_original | spatial: residual_p1_b1500 | 0.356884 | 0.367876 ± 0.000291 | 8848 | 9.1 |
| burgers_original | spatial: direct_p0_b1500 | 0.341414 | 0.389238 ± 0.001264 | 8848 | 8.8 |
| burgers_original | fno: residual_p1_b1500 | 0.236473 | 0.251721 ± 0.002890 | 136481 | 14.9 |
| burgers_original | fno: direct_p0_b1500 | 0.120503 | 0.195846 ± 0.007377 | 136481 | 15.6 |
| burgers_independent | frozen linear | 0.388369 | 0.409382 | — | — |
| burgers_independent | fit: residual_p0_b1500 | 0.343913 | 0.411857 ± 0.002433 | 8752 | 4.1 |
| burgers_independent | fit: residual_p0_b6000 | 0.303998 | 0.439150 ± 0.010807 | 8752 | 16.3 |
| burgers_independent | fit: residual_p0.1_b1500 | 0.344240 | 0.408517 ± 0.001600 | 8752 | 4.2 |
| burgers_independent | fit: residual_p0.1_b6000 | 0.305386 | 0.430581 ± 0.007836 | 8752 | 16.3 |
| burgers_independent | fit: residual_p1_b1500 | 0.352391 | 0.402240 ± 0.001267 | 8752 | 3.3 |
| burgers_independent | fit: residual_p1_b6000 | 0.325689 | 0.404522 ± 0.005437 | 8752 | 15.0 |
| burgers_independent | fit: direct_p0_b1500 | 0.343543 | 0.412777 ± 0.001916 | 8752 | 4.1 |
| burgers_independent | fit: direct_p0_b6000 | 0.304874 | 0.434613 ± 0.004063 | 8752 | 15.4 |
| burgers_independent | spatial: residual_p1_b1500 | 0.350865 | 0.401304 ± 0.000833 | 8848 | 8.8 |
| burgers_independent | spatial: direct_p0_b1500 | 0.337818 | 0.423261 ± 0.001015 | 8848 | 8.8 |
| burgers_independent | fno: residual_p1_b1500 | 0.232868 | 0.276705 ± 0.004599 | 136481 | 15.4 |
| burgers_independent | fno: direct_p0_b1500 | 0.122561 | 0.220706 ± 0.005479 | 136481 | 15.4 |
| allen_cahn | frozen linear | 0.457111 | 0.458885 | — | — |
| allen_cahn | fit: residual_p0_b1500 | 0.318316 | 0.321721 ± 0.003894 | 8752 | 3.8 |
| allen_cahn | fit: residual_p0_b6000 | 0.268295 | 0.290741 ± 0.012629 | 8752 | 15.0 |
| allen_cahn | fit: residual_p0.1_b1500 | 0.321457 | 0.325304 ± 0.003717 | 8752 | 3.7 |
| allen_cahn | fit: residual_p0.1_b6000 | 0.266326 | 0.287272 ± 0.008826 | 8752 | 15.5 |
| allen_cahn | fit: residual_p1_b1500 | 0.355445 | 0.360583 ± 0.004744 | 8752 | 3.5 |
| allen_cahn | fit: residual_p1_b6000 | 0.302827 | 0.317214 ± 0.006292 | 8752 | 16.0 |
| allen_cahn | fit: direct_p0_b1500 | 0.201075 | 0.205536 ± 0.004539 | 8752 | 3.9 |
| allen_cahn | fit: direct_p0_b6000 | 0.089687 | 0.094198 ± 0.000262 | 8752 | 15.9 |
| allen_cahn | spatial: residual_p0.1_b6000 | 0.058191 | 0.063771 ± 0.001256 | 8848 | 35.4 |
| allen_cahn | spatial: direct_p0_b6000 | 0.041433 | 0.046134 ± 0.000290 | 8848 | 35.3 |
| allen_cahn | fno: residual_p0.1_b6000 | 0.041661 | 0.043046 ± 0.000211 | 136481 | 61.3 |
| allen_cahn | fno: direct_p0_b6000 | 0.002139 | 0.003315 ± 0.000286 | 136481 | 60.8 |

## Historical strong controls

Previous five-seed development means, retained as context (not paired three-seed comparisons; different checkpoint selection).

| Task | All modes | Old continuous shrinkage |
|---|---:|---:|
| burgers_original | 0.363277 | 0.362297 |
| burgers_independent | 0.403154 | 0.402073 |
| allen_cahn | 0.359048 | 0.359048 |

Source: public_architecture_stages_v1/*/*_summary.json; historical controls from public_uncertainty_modal_v1/*/development.json.
Budgets also change cosine schedules. CNN uses the same POD; FNO predicts the full field and has more parameters. Transferred MLP-selected settings are not architecture-specific tuning. Optimization times exclude evaluation/checkpoint overhead and are hardware/load dependent.
