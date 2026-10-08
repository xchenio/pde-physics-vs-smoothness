# Strong FNO: physics feasibility prerequisite (2026-10-08)

The previous500-trajectory confirmation is consumed. Its labels will not be used
to tune this follow-up. No result from that study is overwritten.

The historical FNO has five discrete output heads; differentiating those heads
does not define a time derivative. A new continuous-time FNO conditions its
encoder on query time. This is an architectural change, so it must have its own
data-only control before physics can be credited with any benefit.

`src/time_fno.py` implements time JVP at fixed input trajectory, spectral spatial
derivatives, and conservative Burgers residual coefficients on fixed modes0..16.
Nonlinear flux uses2x oversampling and explicit Nyquist splitting. This is a
resolved weak constraint, not a claim of enforcing the full PDE at all scales.

Before training, audit100 existing training trajectories at1024/512/256 sensors,
using central time differences0.01/0.02 around0.1/0.25/0.5/1.0. Adjacent frames
are diagnostic-only extra training-label access, NOT five-label-budget fitting.
No development or confirmation labels are read by the audit. All coordinates
are the native retained cell centers; viscosity is0.001/pi.

Feasibility gate, fixed before the audit: the median normalized resolved PDE
residual at raw256 sensors and dt0.01 must be<=0.1 at each registered time. The
denominator is the root sum of squared time/flux/diffusion term RMS. Passing
only establishes plausibility at the retained scales; it does not prove target
consistency or generalization. Failure requires diagnosing discretization before
optimizing this residual, not tuning the frequency threshold for gains.

Next, if the gate passes: compare data-only, resolved PDE, gradient and curvature
regularization within the SAME continuous-time FNO and fixed data budget. Common
warm starts, paired batches and gradient-calibrated penalty-strength grids are
needed. Keep historical discrete-head FNO as a contextual strong control, report
continuous-time baseline quality, and seal selection before opening new IDs.
No claim of novelty from PINO or adding physics regularization alone is made.

## Audit result and next prerequisite

The registered feasibility gate passed. At raw256 and dt0.01, median resolved
relative residual at0.1/0.25/0.5/1.0 is0.042138/0.036492/0.031825/0.018282
(see audit.json for exact values). The corresponding almost-full-band medians
are0.507245/0.511209/0.421037/0.254306. Full-grid residual minimization is thus
not justified by these target-discretization diagnostics. Passing low-mode
consistency still does not establish that physics improves prediction.

`time_fno_baseline_v1` registers three seeds of the new time-conditioned FNO,
same100 trajectories and five labels, width16/modes16/layers4,1500 fixed steps,
all five labels per batch, checkpoints every50 steps. Temporal JVP PDE values
are training-probe diagnostics only, NOT loss terms in this prerequisite.
The continuation gate is the three-seed development mean<=1.10 times the
historical discrete-head FNO16 mean0.2871897. It is a predeclared engineering
gate, NOT an independent scientific test. A failed gate requires reporting
the architecture/optimization penalty before attempting physics gains.

The three data-only runs completed: development0.269737/0.283157/0.278617,
mean0.277170 and seedSD0.006826;34209 real parameters. This passes the gate and
is descriptively3.49% below the previous five-head FNO16 mean0.287190, but is
NOT a physics effect or fresh-test gain. All checkpoint predictions and rolling
minibatch streams were independently replayed; batch streams equal the previous
matched100-shot FNO. Training-probe resolved PDE RMS remains2.439/2.673/2.530.

## Locked physics versus smoothness continuation

`fno_regularization_v1` forks each final1500-step checkpoint, including optimizer
and minibatch RNG. Thirty runs:3 seeds times [none;3 strengths for each of PDE,
gradient, curvature]. All continue750 steps with common LR restart0.0003 to
0.00001; compare these2250-total-step arms to each other, not to1500-step models
as though the training budgets were equal. Parameter counts and initial states
are identical. Common random query times0.1..1 are used for all penalties.

At the frozen warm start, the first16 TRAINING IDs and fixed query times calibrate
each penalty weight to initial gradient ratios0.03/0.1/0.3 relative to the data
loss. Weights remain fixed; later ratios are recorded, not retuned. Calibration
uses real-component norms for complex parameters. There are no target labels
at the sampled collocation times. All arms compute all three penalty forwards;
backward work differs, so training times are not exact compute-matched benchmarks.

All constraints use the same FIXED0..16 band. The controls are matched resolved
gradient/curvature smoothing, not optimally tuned full-band regularization.
All strengths/seeds are retained; per-family selection uses minimum three-seed
development mean, ties favor weaker regularization. No best checkpoint or best
seed selection. No confirmation IDs are selected or opened in this pilot.
The next family must snapshot CURRENT historical IDs, seal rules/checkpoints,
and only then extract truly unused trajectories. This is not second-PDE validation.

Resume command (automatically skips completed models):

```powershell
D:\miniconda3\envs\pytorch\python.exe scripts/run_fno_regularization.py run
```

Checkpoints persist model, optimizer, both RNG streams, stage, normalization,
calibration, histories and accumulated elapsed time every50 steps. After an
abrupt shutdown, rerun the command; at most the unsaved<50-step segment is lost.
Binary checkpoints stay local and must be backed up separately from Git.

Commands (pytorch interpreter):

```powershell
D:\miniconda3\envs\pytorch\python.exe -m unittest tests.test_time_fno -q
D:\miniconda3\envs\pytorch\python.exe scripts/audit_fno_physics_resolution.py plan
D:\miniconda3\envs\pytorch\python.exe scripts/audit_fno_physics_resolution.py run
```
