# Spatial CNN / smooth-trunk DeepONet baseline, 2026-10-08

9 fits, rank64, three seeds; historical correction600 fitting and development200.
Same guarded field objective, minibatches and step budgets as retained FNO.
CNN/trunk capacity differs from FNO, so comparison is contextual, not encoder-isolated.
No physics/attention, future-label branch input, fresh confirmation, or temporal claims.

| Task | Model | Params | Train relative L2 | Dev relative L2 | Dev seed SD |
|---|---|---:|---:|---:|---:|
| burgers_original | CNN DeepONet64 | 24356 | 0.329657 | 0.383864 | 0.000997 |
| burgers_original | Retained full FNO | 136481 | 0.120503 | 0.195846 | 0.007377 |
| burgers_independent | CNN DeepONet64 | 24356 | 0.318988 | 0.405222 | 0.005340 |
| burgers_independent | Retained full FNO | 136481 | 0.122561 | 0.220706 | 0.005479 |
| allen_cahn | CNN DeepONet64 | 24356 | 0.021316 | 0.023343 | 0.001351 |
| allen_cahn | Retained full FNO | 136481 | 0.002139 | 0.003315 | 0.000286 |

Full per-seed and per-sample replay: results/spatial_deeponet_v1/audit.json.
Coordinates are native retained cell centers; physical x/time inputs have tested
first/second derivative chain factors. Constant training time does not constrain
u_t or certify extrapolation. The next physics stage needs trajectory data first.
