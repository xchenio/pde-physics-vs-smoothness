| target | C1 shape | C2 contamination | C3 pairing | verdict | failing leaves | max |diff| |
|---|---|---|---|---|---|---|
| `burgers/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `burgers/evaluator:fixed` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `burgers/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: ic/target | 0.000e+00 |
| `burgers/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `burgers2d/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `burgers2d/evaluator:fixed` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `burgers2d/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: ic/target | 0.000e+00 |
| `burgers2d/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `dr/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `dr/evaluator:fixed` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `dr/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: col/forcing | 0.000e+00 |
| `dr/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `darcy/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `darcy/evaluator:fixed` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `darcy/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: col/a, col/ax, col/ay | 0.000e+00 |
| `darcy/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
