# JCS submission package manifest

Prepared: 2026-10-26. This is a submission draft; author metadata and release identifiers listed below still require the corresponding author's input.

## Package contents

| File | Role |
|---|---|
| `main.tex` | Main manuscript in the supplied Elsevier CAS single-column format |
| `supplement.tex` | Supplementary methods, full secondary results, provenance, and limits |
| `references.bib` | Bibliography database used by both documents |
| `highlights.txt` | Separate highlights upload |
| `cover_letter.md` | Editable cover-letter draft |
| `submission_checklist.md` | Author and journal-policy actions before upload |
| `plot_data.csv`, `scripts/plot_summary.py`, `figures/relative_gains.pdf` | Figure source values, generator, and vector figure |
| `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Support files copied from the supplied `els-cas-templates` folder |
| `../../output/pdf/jcs_manuscript.pdf`, `../../output/pdf/jcs_supplement.pdf` | Compiled release PDFs after final layout inspection |

SHA-256 checksums: `jcs_manuscript.pdf` = `7de4f3e5e228b4d86665110c1ae9ce6c700a139dccd7f455edd05ee08442483c`; `jcs_supplement.pdf` = `517aa218ec4c09dbf5f999443dea42227afb277d75a0b306020eb6e8e372911a`.

## Evidence source map

| Manuscript evidence | Project records |
|---|---|
| Locked Burgers primary comparison | `docs/FNO_REGULARIZATION_V1.md`, `docs/FNO_REGULARIZATION_V1_RESULTS.md`; `results/fno_regularization_v1/`; `results/fno_regularization_confirmation_v1/` |
| Burgers sparse/noisy stress tests | `docs/PHYSICS_STRESS_V1.md`, `docs/PHYSICS_STRESS_V1_RESULTS.md`; `results/physics_stress_v1/` |
| Burgers sample-count/resolution secondary study | `docs/PHYSICS_AS_SMOOTHNESS_V1.md`, `docs/PHYSICS_AS_SMOOTHNESS_V1_RESULTS.md`; `results/fno_physics_resolution_v1/` |
| Allen--Cahn pointwise versus interval and curvature controls | `docs/ALLEN_POINTWISE_CONTROL_V1.md`, `docs/ALLEN_POINTWISE_CONTROL_V1_RESULTS.md`; `results/allen_pointwise_control_v1/`, `results/allen_interval_replication_v1/`, and `results/allen_interval_pilot_v1/` |
| Public PDEBench source | DaRUS DOI `10.18419/darus-2986`; 1D Burgers HDF5 shape `(10000, 201, 1024)`; SHA-256 `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d` |

## Data-use and claim boundaries

- The 500-trajectory Burgers confirmation holdout was opened after protocol lock and is now consumed. The manuscript reports it only as the registered, one-time confirmation comparison.
- Burgers stress and sample-count/resolution studies reuse project splits and are secondary descriptive evidence. Their old seed-level bootstrap significance labels are not used as confirmatory inference.
- Allen--Cahn uses synthetic IDs 200000--200299; 68 of the 200 development IDs were in an earlier training partition. Pointwise/interval outcomes are development evidence, not independent confirmation or an untouched-trajectory result.
- The two Allen--Cahn constraints are close in the reported means, but no equivalence margin or formal equivalence test was specified. The paper makes no equivalence claim.
- The package does not include raw trajectory caches, model/optimizer checkpoints, or a future untouched holdout. A code and derived-artifact archive must be created and assigned a stable URL and DOI before submission.

## Required before journal upload

Replace the author, affiliation, corresponding email, funding, competing-interest, AI-use, and code/archive DOI placeholders with verified statements. Confirm the current journal instructions and every author’s approval. Then regenerate and inspect both PDFs; the delivered source package does not imply these fields have been completed.
