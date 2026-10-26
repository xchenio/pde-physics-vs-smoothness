# JCS manuscript restructuring notes (2026-10-26)

Active manuscript: `jcs_submission/main.tex`. Supplement: `jcs_submission/supplement.tex`.

## Changes to the publication claim

- Replaced the September DeepONet-focused draft with a coherent study of physics penalties versus matched smoothness controls. The prior draft is preserved under `archive/pre_jcs_20261026/` and is not the active submission.
- Removed the old draft's basis/projection, optimizer, and generalization claims from the new main text because they concern a separate supervised DeepONet study and do not support the latest physics-versus-smoothness question.
- Put the locked PDEBench Burgers holdout, matched gradient/curvature controls, the Allen--Cahn pointwise-versus-interval comparison, and observed negative results in the main text.
- Explicitly distinguish the 500-trajectory Burgers holdout, which was locked before evaluation and is now consumed, from reused Burgers stress data and the previously consumed Allen--Cahn development pool.
- Report Allen--Cahn pointwise/interval scores descriptively. No equivalence margin, significance test, or independent confirmation exists for that comparison.
- Include the sparse/resolution study's per-family means only. Its archived significance intervals resample three seed aggregates and select a smoothness family on test outcomes; these intervals are not used as evidence in the manuscript.
- Avoid claims of a new architecture, new integral loss, universal PINN failure, universal PDE/smoothing equivalence, or cross-resolution transfer.

## Remaining release gates

Author metadata and declarations, public code/data archive URL/DOI, current JCS author-guide checks, AI-assisted writing disclosure if required, and final rebuild after those edits. See `jcs_submission/submission_checklist.md`.
