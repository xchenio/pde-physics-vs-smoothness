# Spatial latent operator roadmap, 2026-10-08

Research hypothesis, not an established result: under the same scalar compression
budget, spatially organized latent tokens may preserve derivatives and scales better
than a global latent vector; physics-informed training may then improve small-data
or cross-resolution prediction. Each clause requires a separate control.

## Stage 1: representation diagnostic (locked before training)

Run `D:/miniconda3/envs/pytorch/python.exe scripts/run_manifold_reconstruction.py run`.
Family `results/manifold_reconstruction_v1`; immutable source/data identities.
Three existing tasks: public Burgers original and independent split, and synthetic
nonlinear Allen-Cahn. Only representation400 output snapshots train the compressor;
existing development200 evaluate it. Mapping correction600 remain excluded here.
POD and each AE use the same train-only scalar mean/std, no sample mean side channel.
POD is newly fitted to these snapshots, not the historical input-mean-offset POD.

Scalar bottlenecks16/32/64, seeds0/1/2, 1500 optimizer steps:81 neural fits.
Global:256->64->64->r MLP and mirrored decoder. Local: periodic stride4/4 CNN,
16 spatial positions with r/16 channels and local upsample decoder, no skip.
Scrambled: same local model/initial trainable weights, fixed seed2026100801 spatial
permutation before encoding and its inverse after decoding. This retains the full
function class under relabeling but removes alignment with physical neighborhoods.
Local/scrambled exactly match parameters/batches; global capacity differs and must
not be presented as a uniquely isolated locality effect. Scalar budget excludes
trained decoder parameters, which are separately reported for every model.

AdamW1e-3 to1e-5 cosine, wd1e-4, batch32, normalized field MSE only; fixed final
step, no development checkpoint picking. Derivative/scale losses are not optimized.
Evaluate per-sample field, first/second periodic spectral derivative L2, guarded by
max(1e-12,0.001 train RMS for that quantity). Nyquist first derivative is zero.
Bands0-4/5-16/17-64/65-128 report band ERROR/full field norm, not division by tiny
band energy. Band squared errors sum to squared field relative error per sample.
These are finite-grid structure diagnostics, not continuum/PDE residual certificates.
All ranks and seeds reported. Repeated development; no new confirmation opened.
Full optimizer/model/random/step/history checkpoints every50; eval every250.

## Stage 2: data-only operator baseline (registered, training after stage1)

Retain historical full-field FNO and its post-hoc projection. New registered family
`results/spatial_deeponet_v1`: CNN branch plus smooth coordinate trunk DeepONet;
rank64 predetermined before stage1 finished,9 fits, three tasks x three seeds.
Command `D:/miniconda3/envs/pytorch/python.exe scripts/run_spatial_deeponet.py run`.
Periodic stride4/4 CNN,16positions x4channels, linear64->64 branch; periodic Fourier
features1..16 with physical x and normalized t, tanh33->64->64->64 trunk. Branch
head zero initialized, fixed final step1500 Burgers/6000 Allen-Cahn. Same correction600
and development200, normalized weighted field objective and paired batches as FNO.
Capacity differs; no attribution to encoder alone. No transferred output AE at inference.
Coordinates use actual Burgers retained centers(4i+0.5)/1024 and Allen-Cahn2i/256.
Physical coordinate derivative finite-difference and periodicity tests passed; CPU/CUDA
bitwise resume passed. This first baseline is fixed-time only; distinguish reconstruction
from the initial-field-to-solution mapping. Check coordinate derivative correctness
before adding physics. Select architecture/trunk rank by a declared development
protocol, not by reusing stage1 reconstruction as evidence of operator accuracy.
For evolving PDEs, expand training to u(a;x,t); fixed-time training cannot identify u_t.
Existing Burgers HDF5 contains time slices; inspect coordinate/time conventions and
subsampling before choosing a trajectory protocol. Synthetic solver can provide
time slices; register generator versions and discretization QA. No new data required
for these initial studies. Cross-resolution labels must be native higher resolution
or independently verified solver data, not upsampled256 predictions.

## Stage 3: attention-only ablation (pending)

Keep multiple spatial tokens; compare attention with no attention using matched
trainable parameter count and steps, report actual time/VRAM. Include spatial
permutation/global-vector controls if stage1 supports their diagnostic value.
Coordinate and position information explicit. No physics loss in this stage.

## Stage 4: physics-only ablation (pending)

Same chosen data-only architecture, same labeled trajectories/time coverage, matched
collocation sets and optimization budgets, physics weight0 versus registered nonzero
weights. Smooth coordinate path, physical-unit chain factors, independent derivative
tests, periodic IC/BC controls. Training-only collocation; never feed true future
fields into inference branch. Choose one primary stress condition before outcomes
(small data OR noise OR time generalization), not post-hoc best of all three.
Report data and PDE metrics separately and gradient norms of composite objectives.

## Stage 5: sealed combination confirmation (pending)

Use a factorial attention on/off x physics on/off design, not only best combination
versus a weak baseline. Keep FNO and strongest data-only control. Register full
model hashes, rules, training budget, primary endpoint, multiplicity and minimum
effect BEFORE extracting genuinely unused trajectory IDs/targets. All earlier
confirmation sets remain consumed; audit complete historical ID union at that time.
Report paired per-trajectory differences and uncertainty, seed variability, training
time, peak GPU allocated memory, and synchronized warmed-up inference timings at
fixed batch/resolution. No superiority, new-manifold or novelty claim before evidence.

Sources informing the roadmap: DeepONet https://www.nature.com/articles/s42256-021-00302-5 ;
physics-informed DeepONet https://arxiv.org/abs/2103.10974 ;
latent DeepONet https://www.nature.com/articles/s41467-024-49411-w .
