# Matched token attention ablation, 2026-10-08

18 fits, same CNN/trunk,64-scalar bottleneck, fixed-time600 fitting /200 development.
All gate/attention parameters, initial tensors and minibatch sequences match exactly.
Pointwise QKV gate versus four-head attention; final branch head mixes globally in both.
No physics or unused confirmation in this study. Reused development, three seeds.

| Task | Arm | Params | Train L2 | Dev L2 | Seed SD |
|---|---|---:|---:|---:|---:|
| burgers_original | gate | 33256 | 0.317622 | 0.376506 | 0.004294 |
| burgers_original | attention | 33256 | 0.263010 | 0.288996 | 0.005388 |
| burgers_independent | gate | 33256 | 0.311043 | 0.396253 | 0.003215 |
| burgers_independent | attention | 33256 | 0.268345 | 0.332460 | 0.013545 |
| allen_cahn | gate | 33256 | 0.021586 | 0.022733 | 0.010142 |
| allen_cahn | attention | 33256 | 0.025891 | 0.027256 | 0.004406 |

Per-seed paired attention-minus-gate differences and per-sample errors in audit.json.
Equal parameters/steps do not imply equal time/FLOPs. This isolates the registered
attention-vs-gate interaction, not all possible nonattention/global mixers.
