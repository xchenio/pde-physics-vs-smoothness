| target | C1 shape | C2 contamination | C3 pairing | verdict | failing leaves | max |diff| |
|---|---|---|---|---|---|---|
| `deepxde/cartesian_prod(shared grid, correct)` | PASS | PASS | PASS | **PASS** | -- | 6.661e-16 |
| `deepxde/cartesian_prod(per-sample coords)` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `deepxde/per-sample einsum(correct pairing)` | PASS | PASS | PASS | **PASS** | -- | 8.882e-16 |
