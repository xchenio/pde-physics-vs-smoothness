# Compression reconstruction results, 2026-10-08

81 completed neural fits, independently replayed; POD train-only controls at each rank.
Reused development200. Means over three seeds; no unseen confirmation or operator accuracy claim.
Identical scalar bottlenecks. Local/scrambled exactly matched capacity; global capacity differs.
New POD uses train-only scalar centering, not the historical input-mean-offset POD.
Derivative metrics are finite-grid spectral diagnostics with train-only denominator guards.

| Task | Rank | Model | Parameters | Field | Gradient | Curvature |
|---|---:|---|---:|---:|---:|---:|
| burgers_original | 16 | POD | basis | 0.115408 | 0.946463 | 1.233028 |
| burgers_original | 16 | global | 43536 | 0.279136 | 1.769377 | 5.786739 |
| burgers_original | 16 | local | 19202 | 0.115602 | 0.944581 | 1.320254 |
| burgers_original | 16 | scrambled | 19202 | 0.364586 | 2.300118 | 6.511494 |
| burgers_original | 32 | POD | basis | 0.072913 | 0.880836 | 1.342715 |
| burgers_original | 32 | global | 45600 | 0.283668 | 1.824361 | 6.093776 |
| burgers_original | 32 | local | 19267 | 0.086863 | 0.879363 | 1.308145 |
| burgers_original | 32 | scrambled | 19267 | 0.360066 | 2.606393 | 7.446630 |
| burgers_original | 64 | POD | basis | 0.042112 | 0.769676 | 1.478754 |
| burgers_original | 64 | global | 49728 | 0.273703 | 1.827003 | 6.137008 |
| burgers_original | 64 | local | 19397 | 0.088439 | 0.870982 | 1.211940 |
| burgers_original | 64 | scrambled | 19397 | 0.362100 | 2.389334 | 6.730347 |
| burgers_independent | 16 | POD | basis | 0.124900 | 0.949694 | 1.238161 |
| burgers_independent | 16 | global | 43536 | 0.277258 | 1.692691 | 4.668795 |
| burgers_independent | 16 | local | 19202 | 0.116420 | 0.947628 | 1.351376 |
| burgers_independent | 16 | scrambled | 19202 | 0.397418 | 2.521432 | 6.620654 |
| burgers_independent | 32 | POD | basis | 0.078195 | 0.877422 | 1.312071 |
| burgers_independent | 32 | global | 45600 | 0.272016 | 1.634165 | 4.436959 |
| burgers_independent | 32 | local | 19267 | 0.079238 | 0.862322 | 1.331856 |
| burgers_independent | 32 | scrambled | 19267 | 0.396238 | 2.486603 | 6.490874 |
| burgers_independent | 64 | POD | basis | 0.043446 | 0.752621 | 1.375499 |
| burgers_independent | 64 | global | 49728 | 0.260557 | 1.659243 | 4.492742 |
| burgers_independent | 64 | local | 19397 | 0.083473 | 0.872963 | 1.320474 |
| burgers_independent | 64 | scrambled | 19397 | 0.386991 | 3.031603 | 8.133726 |
| allen_cahn | 16 | POD | basis | 0.025837 | 0.190273 | 0.506683 |
| allen_cahn | 16 | global | 43536 | 0.135748 | 3.207217 | 85.554315 |
| allen_cahn | 16 | local | 19202 | 0.017695 | 0.361935 | 8.101366 |
| allen_cahn | 16 | scrambled | 19202 | 0.676119 | 17.274454 | 467.076968 |
| allen_cahn | 32 | POD | basis | 0.001479 | 0.019455 | 0.091974 |
| allen_cahn | 32 | global | 45600 | 0.123602 | 2.997239 | 80.041983 |
| allen_cahn | 32 | local | 19267 | 0.012192 | 0.313640 | 8.207899 |
| allen_cahn | 32 | scrambled | 19267 | 0.539436 | 20.518089 | 562.188397 |
| allen_cahn | 64 | POD | basis | 0.000006 | 0.000152 | 0.001347 |
| allen_cahn | 64 | global | 49728 | 0.118738 | 2.730453 | 73.051354 |
| allen_cahn | 64 | local | 19397 | 0.010969 | 0.263379 | 7.017581 |
| allen_cahn | 64 | scrambled | 19397 | 0.445701 | 19.255227 | 524.846375 |

## Frequency-band errors

Band error / total field norm (not relative band energy); mean across samples/seeds.
Per sample, sum of squared band errors equals squared guarded relative field error.

| Task | Rank | Model | 0–4 | 5–16 | 17–64 | 65–128 |
|---|---:|---|---:|---:|---:|---:|
| burgers_original | 16 | POD | 0.005382 | 0.092428 | 0.066426 | 0.013781 |
| burgers_original | 16 | global | 0.216469 | 0.149213 | 0.071158 | 0.031975 |
| burgers_original | 16 | local | 0.014939 | 0.091281 | 0.065176 | 0.014052 |
| burgers_original | 16 | scrambled | 0.314581 | 0.141786 | 0.078233 | 0.054243 |
| burgers_original | 32 | POD | 0.001564 | 0.025746 | 0.065689 | 0.014588 |
| burgers_original | 32 | global | 0.221770 | 0.148950 | 0.071892 | 0.034026 |
| burgers_original | 32 | local | 0.013701 | 0.052807 | 0.063489 | 0.014032 |
| burgers_original | 32 | scrambled | 0.307763 | 0.140230 | 0.082145 | 0.061530 |
| burgers_original | 64 | POD | 0.000526 | 0.004193 | 0.038261 | 0.016220 |
| burgers_original | 64 | global | 0.209467 | 0.148888 | 0.071450 | 0.033682 |
| burgers_original | 64 | local | 0.009522 | 0.056627 | 0.064552 | 0.013513 |
| burgers_original | 64 | scrambled | 0.311839 | 0.141144 | 0.079050 | 0.055055 |
| burgers_independent | 16 | POD | 0.005073 | 0.100285 | 0.070929 | 0.016106 |
| burgers_independent | 16 | global | 0.202315 | 0.159131 | 0.075311 | 0.036246 |
| burgers_independent | 16 | local | 0.018346 | 0.087499 | 0.069287 | 0.016203 |
| burgers_independent | 16 | scrambled | 0.339068 | 0.152051 | 0.087239 | 0.066889 |
| burgers_independent | 32 | POD | 0.001864 | 0.028914 | 0.069281 | 0.016688 |
| burgers_independent | 32 | global | 0.194708 | 0.160038 | 0.075176 | 0.034529 |
| burgers_independent | 32 | local | 0.013038 | 0.040150 | 0.061903 | 0.016373 |
| burgers_independent | 32 | scrambled | 0.338416 | 0.151863 | 0.086805 | 0.064219 |
| burgers_independent | 64 | POD | 0.000458 | 0.004147 | 0.039083 | 0.017626 |
| burgers_independent | 64 | global | 0.180415 | 0.159912 | 0.075067 | 0.035663 |
| burgers_independent | 64 | local | 0.009653 | 0.044168 | 0.066452 | 0.016112 |
| burgers_independent | 64 | scrambled | 0.324624 | 0.148128 | 0.096144 | 0.079164 |
| allen_cahn | 16 | POD | 0.000423 | 0.025803 | 0.001179 | 0.000000 |
| allen_cahn | 16 | global | 0.082707 | 0.094344 | 0.024026 | 0.027668 |
| allen_cahn | 16 | local | 0.007466 | 0.014010 | 0.005538 | 0.003904 |
| allen_cahn | 16 | scrambled | 0.585385 | 0.141662 | 0.194033 | 0.224667 |
| allen_cahn | 32 | POD | 0.000001 | 0.000741 | 0.001202 | 0.000000 |
| allen_cahn | 32 | global | 0.067094 | 0.091911 | 0.022393 | 0.026118 |
| allen_cahn | 32 | local | 0.005383 | 0.009279 | 0.003613 | 0.003324 |
| allen_cahn | 32 | scrambled | 0.394844 | 0.131432 | 0.221833 | 0.254783 |
| allen_cahn | 64 | POD | 0.000000 | 0.000000 | 0.000006 | 0.000000 |
| allen_cahn | 64 | global | 0.059046 | 0.093162 | 0.020728 | 0.023785 |
| allen_cahn | 64 | local | 0.004664 | 0.008748 | 0.003065 | 0.002738 |
| allen_cahn | 64 | scrambled | 0.292780 | 0.119208 | 0.201312 | 0.236296 |

Machine-readable per-seed pairing differences: results/manifold_reconstruction_v1/audit.json.
Full per-sample errors, training curves, time, peak last-segment GPU allocation and hashes
remain in result.json per run. Fixed1500-step MSE training; derivative errors not optimized.
All reconstruction encoders see the true output snapshot; this is a representation
diagnostic and does not show initial-field-to-future-field prediction performance.
