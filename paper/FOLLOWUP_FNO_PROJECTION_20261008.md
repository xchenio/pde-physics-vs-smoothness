# Shared pointwise FNO head with training-time POD projection

Protocol seal3cac70a. 36 paired development fits: full output and training-time POD16/32/64 on three tasks, three seeds. The same FNO encoder and shared32→32→1 pointwise head has136481 real parameters in all arms. Initial states, minibatch sequences, optimizer and schedules are paired. Projection changes gradients and the output constraint together. Coefficient loss equals projected-field squared loss up to the fixed target projection error. Basis and400/600/200 representation/correction/development split match prior experiments. No new confirmation is opened or candidate selected for confirmation.

## Original Burgers completed

| Rank | Training-time projection dev | Full-training post-hoc projection dev | Prior pooled coefficient head dev |
|---|---:|---:|---:|
| 16 | 0.191759 | 0.192454 | 0.273799 |
| 32 | 0.188328 | 0.192368 | 0.288870 |
| 64 | 0.190009 | 0.194286 | 0.296425 |

Full-field dev0.195846 is reproduced bitwise by all three new full-arm checkpoints. The same pointwise model can learn projected coefficients without the prior large pooled-head gap. Rank32/64 training projection improves the paired post-hoc projection in each seed here; rank16 has mixed seed differences. These are development observations, without unseen-data inference. Previous pooled models differ in head structure, parameter count and initialization ordering, so their comparison supports investigating readout structure but does not uniquely establish head geometry as the cause.

Independent Burgers and Allen-Cahn are in progress. All intermediate/final metrics include raw and all three projected outputs. For POD-trained models only the projected field is the deployed output; the untrained complement of the raw field is diagnostic. Full numeric source: results/public_fno_projection_v1; protocol: docs/FNO_PROJECTION_V1.md. Current audit12 completed checkpoints;57 regression tests passed.
