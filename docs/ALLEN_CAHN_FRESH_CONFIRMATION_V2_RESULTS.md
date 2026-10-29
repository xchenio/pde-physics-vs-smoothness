# Allen--Cahn fresh-trajectory confirmation V2 results

## Locked protocol and artifacts

- Plan SHA-256: `5929fb27b95a881f9d6f13be14109995c9534d48a91aa9c626c046d4facc47a4`.
- Confirmation trajectories: 200 new Fourier-random initial conditions, IDs `320000..320199`; no overlap with the old development pool `200000..200299`, solver-QA IDs `300000..300015`, or retired unscored V1 IDs `310000..310199`.
- Frozen checkpoints: 24 audited models, four arms, dense and sparse conditions, three fixed seeds. New training steps: zero.
- Confirmation dataset SHA-256: `50a7375941f2577f10abff9bd70996fd576ebbbcca0c9cb33fd29d5ec15abfed`.
- Solver QA passed: maximum time-refinement relative error `8.50e-8`; maximum spatial-refinement discrepancy `1.24e-13`; minimum coarse/fine refinement ratio `5.00`; independent DOP853 discrepancy `7.49e-8`; maximum resolved residual `2.66e-7`; constant-state error `2.51e-14`.
- Evaluation used the audited normalization and target scales. Metric floors were derived from the exact 100 frozen training IDs. Inference wrote one hash-bound record per model and completed all 24 models before the registered bootstrap summary.

## Primary field results

The field metric is the mean per-trajectory relative $L^2$ error over future times: four times for dense supervision and the three unobserved interior times for sparse supervision. `Difference` is interval minus pointwise; negative values favor the interval condition.

| Supervision | Data only | Full curvature | Uniform interval | Pointwise PDE | Interval − pointwise | 97.5% paired trajectory-bootstrap interval | Difference as % of pointwise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.028565504 | 0.029290873 | 0.027689008 | 0.027688556 | +4.5158e-7 | [3.4321e-7, 5.5879e-7] | +0.00163% |
| Sparse | 0.126244478 | 0.124731618 | 0.114505896 | 0.114514050 | -8.1544e-6 | [-8.7682e-6, -7.5516e-6] | -0.00712% |

The bootstrap resamples the 200 trajectories after averaging each error over the same three fitted seeds. Its intervals are conditional on those frozen fits and do not measure training-seed variability. Both intervals exclude zero, but the absolute and relative field differences are very small. No equivalence margin was specified, so this is not a formal equivalence result. The results show no practically material interval-versus-pointwise difference in this experiment.

Relative to the data-only mean, the interval field error is lower by 3.07% in dense supervision and 9.30% in sparse supervision; the pointwise model is lower by 3.07% and 9.29%. These are descriptive fixed-fit comparisons, not a new training replication or evidence that a physical constraint generally outperforms smoothing.

## Structural trade-offs

| Supervision and arm | Field relative $L^2$ | Gradient relative error | Curvature relative error | Initial-time relative error |
|---|---:|---:|---:|---:|
| Dense, data only | 0.0285655 | 0.108517 | 0.300265 | 0.034094 |
| Dense, full curvature | 0.0292909 | 0.109170 | **0.279846** | 0.034856 |
| Dense, interval | 0.0276890 | 0.104470 | 0.287756 | 0.034841 |
| Dense, pointwise | 0.0276886 | 0.104469 | 0.287753 | 0.034841 |
| Sparse, data only | 0.1262445 | 0.265855 | 0.662483 | 0.024156 |
| Sparse, full curvature | 0.1247316 | **0.232804** | **0.523595** | 0.024957 |
| Sparse, interval | 0.1145059 | 0.239772 | 0.597934 | 0.025680 |
| Sparse, pointwise | 0.1145141 | 0.239792 | 0.597987 | 0.025678 |

The physical-loss fits retain lower sparse field error than the two controls, while full curvature retains lower sparse gradient and curvature error. Both physical fits have higher initial-time error than data-only fits. Dense full curvature has the lowest curvature error; no objective dominates all metrics.

## Interpretation and limits

This is a new-trajectory confirmation for the declared Allen--Cahn initial-condition distribution, with all model weights frozen before the trajectories were generated. It supports the observed similarity of interval and pointwise predictions on new cases. It does not repeat training, provide a second independent model-seed study, test distribution shift, test a second PDE, or establish superiority over a fully tuned smoothness baseline. The training-pool mistake detected in V1 is documented in `docs/ALLEN_CAHN_FRESH_CONFIRMATION_V1_ERRATUM.md`; V1 IDs were retired before any scoring and do not contribute to these results.

Full machine-readable results, per-model predictions' hashes, bootstrap inputs, event records, solver QA, and data receipts are under `results/allen_cahn_fresh_confirmation_v2/`.
