| target | C1 shape | C2 contamination | C3 pairing | verdict | failing leaves | max |diff| |
|---|---|---|---|---|---|---|
| `pi_deeponet/ic_target(official tile semantics)` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `pi_deeponet/branch_tile(official np.tile)` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `pi_deeponet/res_target(official zeros)` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `pi_deeponet/counterfactual: shared-grid reshape` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `pi_deeponet/counterfactual: broadcast tile` | PASS | PASS | PASS | **PASS** | -- | 4.163e-17 |
