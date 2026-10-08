# Spectral/POD controlled development study

Completed54/54; source seal de5cc93. Full tables: SPECTRAL_POD_V1_RESULTS.md.
Audit command: `D:/miniconda3/envs/pytorch/python.exe scripts/audit_spectral_pod.py --require-complete`.
All54 predictions/identities/parameter counts/rank selections replayed;55 regression tests passed; repeat driver skips all54.
Both Burgers tasks select16 and favor spectral; Allen-Cahn selects64 and favors local convolution.
Prior full-field FNO remains stronger on all3 tasks. No new confirmation consumed.

Protocol date:2026-10-08 (user timezone Asia/Shanghai). Previous108-run architecture study remains unchanged.

54 new fits:3 tasks ×3 nested POD ranks(16,32,64) ×2 encoders ×3 seeds. Direct coefficient prediction only, zero head initialization, guarded normalized squared objective, AdamW0.001→0.00001, weight decay0.0001,batch32. Fixed final steps1500 for both Burgers partitions,6000 for synthetic Allen-Cahn, transferred from previous direct MLP selections. No early stopping or extra hyperparameter search. All ranks reported; best rank by mean development relative L2 per encoder, lower-rank tie break. No new confirmation data.

Encoders share input lift(value+coordinate),4 width32 GELU blocks,24-bin positional average pooling and a linear POD head. Spectral block:16 complex Fourier modes plus local1x1. Non-spectral block:periodic kernel33 convolution. Their nominal real parameter counts match exactly at each rank. This does not match compute, receptive field, initialization, or effective degrees of freedom (the imaginary DC spectral weight is stored but ineffective). The comparison tests these specified architectures, not Fourier transforms in isolation.

Representation basis uses only the existing400 representation rows; preserve historical16 vectors exactly and extend by the orthogonal complement of their representation-only POD. All ranks are nested. Historical input normalization retained. Model fitting uses the same600 correction rows; reused200-row development sets. No labels outside those sets loaded. Zero dummy linear coefficients in direct-only anchors are placeholders, not new ridge baselines.

Code copies frozen training semantics into spectral_pod_core.py to avoid changing historical source hashes. Checkpoint model,optimizer,step,RNG every50 steps; evaluate every250. After interruption rerun:

```powershell
D:/miniconda3/envs/pytorch/python.exe scripts/run_spectral_pod.py run
```

Completed runs skip only with matching identities. Binary checkpoints are local, not in Git; source,protocol,logs and JSON evidence are committed. Existing FNO/direct-CNN/MLP and old continuous/all-mode results remain strong contextual controls. Full-field FNO has a different output space and parameter count. This study is exploratory development work, not a novelty claim or new independent confirmation.
