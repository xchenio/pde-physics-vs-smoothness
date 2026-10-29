# Allen--Cahn fresh confirmation V1 erratum

V1 protocol and data preparation were committed and pushed before evaluation. The first `evaluate` command stopped at the training-cache ID consistency check, before initializing model inference, writing the consumption marker, or producing a model result. No V1 confirmation metric was computed and its trajectories were not used to choose or modify a model.

The cause was a mistaken assumption in the training-reference loader: the frozen model's 100 training IDs are selected across the complete old pool `200000..200299`, while the loader initially read only the two shards corresponding to the old contiguous training split. The source plan and audit already pin all six shards. V2 corrects the loader to verify all six shards and extract only the exact 100 IDs in the frozen training split for scale and denominator-floor calculations.

To keep the scored set unambiguous, V1 IDs `310000..310199` are retired without scoring. V2 uses new confirmation IDs `320000..320199`, repeats the separate solver QA, and locks a new plan before generation. The original V1 plan, files, and commits remain preserved as an audit trail.
