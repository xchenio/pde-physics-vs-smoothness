# JCS submission package manifest

Prepared: 2026-10-28. Author identity and selected declarations have been added from the author's instructions. The archival URL and persistent identifier for this study's code and derived research materials, a full postal-address check, and final author verification remain outstanding.

## Package contents

| File | Role |
|---|---|
| `main.tex` | Main manuscript in the supplied Elsevier CAS single-column format |
| `supplement.tex` | Supplementary methods, full secondary results, provenance, and limits |
| `references.bib` | Bibliography database used by both documents |
| `highlights.txt` | Separate highlights upload |
| `cover_letter.md` | Editable cover-letter draft |
| `submission_checklist.md` | Author and journal-policy actions before upload |
| `plot_data.csv`, `scripts/plot_summary.py`, `figures/relative_gains.pdf` | Figure 1 source values, generator, and vector figure |
| `paired_burgers_differences.csv`, `scripts/plot_paired_burgers_differences.py`, `figures/paired_burgers_differences.pdf` | Figure 2 paired per-trajectory differences from the consumed locked holdout |
| `cas-sc.cls`, `cas-common.sty` | Support files copied from the supplied `els-cas-templates` folder |
| `elsarticle-num.bst` | Elsevier numeric reference style used by the main text and supplement |
| `cas-model2-names.bst` | Original author-year CAS sample style retained but not used |
| `../../output/pdf/jcs_manuscript.pdf`, `../../output/pdf/jcs_supplement.pdf` | Compiled release PDFs after final layout inspection |

## Checked release PDFs

The manuscript PDF is 9 pages total (CAS highlights page plus 8 article pages); the supplement is 4 pages. Both were rebuilt and every page was visually reviewed.

| File | SHA-256 |
|---|---|
| `../../output/pdf/jcs_manuscript.pdf` | `5019F4270977BA93A36A74C18310C8A88BDBD17AA9B510003FAA50540F631D99` |
| `../../output/pdf/jcs_supplement.pdf` | `A952A19ED27E51A969F8570D23F4669EE384C5D23EB38C25518D4D2977C453BB` |

## Evidence source map

| Manuscript evidence | Project records |
|---|---|
| Locked Burgers primary comparison | `docs/FNO_REGULARIZATION_V1.md`, `docs/FNO_REGULARIZATION_V1_RESULTS.md`; `results/fno_regularization_v1/`; `results/fno_regularization_confirmation_v1/` |
| Burgers sparse/noisy stress tests | `docs/PHYSICS_STRESS_V1.md`, `docs/PHYSICS_STRESS_V1_RESULTS.md`; `results/physics_stress_v1/` |
| Burgers sample-count/resolution secondary study | `docs/PHYSICS_AS_SMOOTHNESS_V1.md`, `docs/PHYSICS_AS_SMOOTHNESS_V1_RESULTS.md`; `results/fno_physics_resolution_v1/` |
| Allen--Cahn pointwise versus interval and curvature controls | `docs/ALLEN_POINTWISE_CONTROL_V1.md`, `docs/ALLEN_POINTWISE_CONTROL_V1_RESULTS.md`; `results/allen_pointwise_control_v1/`, `results/allen_interval_replication_v1/`, and `results/allen_interval_pilot_v1/` |
| Public PDEBench source | DaRUS `PDEBench Datasets`, version 8.0, DOI `10.18419/darus-2986`; 1D Burgers HDF5 shape `(10000, 201, 1024)`; SHA-256 `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d` |

## Data-use and claim boundaries

- The 500-trajectory Burgers confirmation holdout was opened after protocol lock and is now consumed. The manuscript reports it only as the registered, one-time confirmation comparison.
- Burgers stress and sample-count/resolution studies reuse project splits and are secondary descriptive evidence. Their old seed-level bootstrap significance labels are not used as confirmatory inference.
- Allen--Cahn uses synthetic IDs 200000--200299; 68 of the 200 development IDs were in an earlier training partition. Pointwise/interval outcomes are development evidence, not independent confirmation or an untouched-trajectory result.
- The two Allen--Cahn constraints are close in the reported means, but no equivalence margin or formal equivalence test was specified. The paper makes no equivalence claim.
- The package does not include raw trajectory caches, model/optimizer checkpoints, or a future untouched holdout. A public code and derived-artifact archive with a stable URL and persistent identifier remains to be created before submission.

## Required before journal upload

Before submission, verify the author name, ORCID, corresponding email, affiliation and postal address, CRediT roles, no-funding and no-competing-interest statements, and generative-AI disclosure. Add the code/data repository URL and persistent identifier, verify whether the 2025 thesis has overlapping material that needs citation, confirm the no-concurrent-submission declaration, and regenerate and inspect both PDFs.
