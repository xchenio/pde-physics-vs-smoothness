# Fixed-architecture physics stress protocol (2026-10-08)

UPDATE2026-10-09:48/48 fits and independent replays COMPLETE, no worker remains.
Results/limits: PHYSICS_STRESS_V1_RESULTS.md. No fresh confirmation opened.

Hypothesis, not a conclusion: governing-equation information may exceed strong
curvature smoothing when temporal supervision is sparse or labels are noisy.
The previous clean100-shot confirmation found no unique PDE advantage. Both
previous500 confirmation sets are consumed and will not tune this protocol.

Same TimeFNO1d throughout: width16/modes16/layers4,34209 real parameters; zero
final head initialization;100 initial trajectories, clean initial-field inputs.
Three model seeds and previously selected initial penalty-gradient ratio0.3;
no per-condition weight search or seed/checkpoint selection. Both PDE and
curvature use resolved modes0..16. Retain no-penalty control.

| Task/condition | Fitting time labels | Future-label noise | Primary clean development times |
|---|---|---|---|
| Burgers sparse | 0,1 | none | 0.1,0.25,0.5 (omitted in this fit) |
| Burgers noisy dense | 0,0.1,0.25,0.5,1 | 10% | 0.1,0.25,0.5,1 |
| Burgers noisy sparse | 0,1 | 10% | 0.1,0.25,0.5 (omitted in this fit) |
| Allen-Cahn dense clean | 0,0.05,0.125,0.25,0.5 | none | 0.05,0.125,0.25,0.5 |

Noise is a fixed Gaussian draw per condition,10% of global RMS of SELECTED
observed clean future training labels. Omitted time labels do not affect noise
amplitude, loss weights, normalization or calibration; isolation is unit-tested.
IC labels and branch inputs are clean. This is noisy-label fitting, NOT noisy-
input inference. Model-seed variability is conditional on that one corruption.

Each condition starts from scratch on its own observation budget:1500 data-only
warm steps, then identical model/optimizer/RNG forks for750 continuation steps
under none/PDE/curvature penalties. Thus no dense-clean warm-start leakage into
sparse/noisy conditions. All forks2250 total steps, cosine LR restart0.0003 to
0.00001, paired cases/query times and fixedfinal evaluation. Training-only first
16 observations calibrate penalty weights; do not use clean hidden targets.

48 fits total:36 Burgers then12 Allen-Cahn, including12 warm starts. Every50
steps persist atomic model/optimizer/stage/history/calibration and both RNGs.
All continuation arms compute both penalty forwards; backward costs differ.
Primary=mean trajectory relativeL2 at registered clean times, average3 seeds.
Observed final-time error is secondary in sparse conditions. Existing Burgers
development200 and its intermediate times have been used in previous design;
"omitted" is within-current-fitting, not globally unseen validation.

Allen-Cahn is newly GENERATED synthetic data, not an open public benchmark:
u_t=0.01u_xx+5u(1-u^2),period2,256 nodes,T0.5,12 initial Fourier modes. Fresh
deterministic IDs200000..200299 for train/dev,200300..200799 reserved and NOT
generated here; solver QA IDs300000..300015. Do not reuse earlier final-only
Allen-Cahn outcomes as physics validation. Cubic reaction uses4x FFT dealiasing;
mean is not conserved. The known initial mean offset is just a network offset.

Before fitting, new trajectory solver QA passes: time refinement8.50e-8,
space refinement1.24e-13, independent DOP853 discrepancy7.49e-8, nonlinear
departure1.371, maximum resolved PDE discrepancy2.65e-7 on dedicated QA IDs.
This validates synthetic numerical data/constraints, not learned prediction.

No new confirmation is opened by this driver. After development, seal decisions
and independent-ID stress verification; later registry must include the latest
consumed500, not reuse a historical frozen registry. No architectural innovation,
time extrapolation or second-PDE success claim before evidence is produced.

```powershell
Set-Location 'F:\Desktop\temp\pinode_align'
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/run_physics_stress.py run
& 'D:\miniconda3\envs\pytorch\python.exe' scripts/audit_physics_stress.py --require-complete
```

Run only one writer. If a process survives sleep, let it continue. After shutdown,
rerun same command without --resume; completed files are checked then skipped.
At most<50 unsaved steps per active model are lost. Git excludes checkpoint
binaries: back them up locally. No additional download is needed for this pilot.
