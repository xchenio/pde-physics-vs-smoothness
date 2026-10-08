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

Commands (pytorch interpreter):

```powershell
D:\miniconda3\envs\pytorch\python.exe -m unittest tests.test_time_fno -q
D:\miniconda3\envs\pytorch\python.exe scripts/audit_fno_physics_resolution.py plan
D:\miniconda3\envs\pytorch\python.exe scripts/audit_fno_physics_resolution.py run
```
