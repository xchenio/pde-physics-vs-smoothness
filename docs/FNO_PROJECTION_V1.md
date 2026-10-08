# Shared FNO output projection experiment

Completed36/36; source seal3cac70a. Tables:FNO_PROJECTION_V1_RESULTS.md.
Audit: `D:/miniconda3/envs/pytorch/python.exe scripts/audit_fno_projection.py --require-complete`.
All36 predictions/identities/projection closures/parameter counts/pairing fingerprints pass;
all9 full final checkpoints match historical tensors bitwise;57 tests passed. Repeat driver skips36.
Paper figure:paper/figures/fno_projection_20261008.png and.svg, visually inspected.
No new confirmation consumed. Training projection improves some Burgers arms; Allen-Cahn
post-hoc projection is slightly better at every rank/seed. Shared readout closes much of the
prior coefficient-head gap; prior-model differences prevent a uniquely causal pooling claim.

2026-10-08. 36 paired development fits:3 tasks ×4 output arms(full, POD16, POD32, POD64) ×3 seeds. Historical136481-real-parameter FNO encoder and shared32→32→1 pointwise head are unchanged in every arm. Initial full states, minibatch sequences and optimizer/schedule settings are paired; their fingerprints are saved.

Full arm optimizes normalized squared field error. POD arm optimizes squared coefficient error `(z Qr − target_coeff)^2` using a fixed basis. This is equivalent to squared projected-field error up to the target's fixed projection-error constant. Both use the same spatial divisor256, target scale and guarded training-energy weights. Float32 projection is differentiable; final metrics use exact stored float64 bases. In all arms the input spatial mean is added after output reconstruction.

Basis/normalization and representation400/correction600/development200 rows are exactly reused from SPECTRAL_POD_V1. Three seeds. Final budgets1500 for Burgers and6000 for Allen-Cahn, batch32, AdamW0.001→0.00001 cosine, weight decay0.0001. Evaluate250/checkpoint50; fixed final step with no early selection. Report every arm/rank; no confirmation selection in this round.

Every trained model is also evaluated after projection to all three ranks. Primary diagnostic: POD-training rankr versus the paired full-training model projected afterward to rankr. Both then have the same output space. Compare the previous pooled coefficient head at rankr as contextual evidence; it has a different parameter count. The study changes gradients/loss and output constraint together, so it cannot identify decoder geometry separately from optimization. A projected-training model's raw field outside its supervised POD subspace is unconstrained and is diagnostic only.

All data are reused development data; no new confirmation labels or IDs are opened. No method novelty or independent success follows from the comparison. Full-arm reproduction against historical FNO is audited.

```powershell
D:/miniconda3/envs/pytorch/python.exe scripts/run_fno_projection.py run
```

Rerunning resumes full optimizer/RNG states from the latest atomic checkpoint and skips completed matching runs. At most49 unsaved updates may be lost after abrupt interruption. Checkpoints remain local/Git-ignored; code/plans/JSON histories are versioned. Preserve checkpoint directories for recovery.
