# JCS submission package manifest

Prepared: 2026-10-29. Author identity and selected declarations have been added from the author's instructions. The code repository is currently private; a public archival URL and persistent identifier for this study's code and derived research materials, a full postal-address check, and final author verification remain outstanding.

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

The manuscript PDF is 9 pages total (CAS highlights page plus 8 article pages); the supplement is 5 pages. Both were rebuilt and every page was visually reviewed after adding the fresh Allen--Cahn trajectory confirmation. The main PDF retains the existing CAS title overfull-box warning; visual review found no clipping. The design-table cells have underfull-box warnings but are readable. The supplement SHA line was made breakable; no overfull paragraph remains there.

| File | SHA-256 |
|---|---|
| `../../output/pdf/jcs_manuscript.pdf` | `9FA977E791413CD5D0A19407B5D1753055CCC9450A15F54256DA5D36BB27C83D` |
| `../../output/pdf/jcs_supplement.pdf` | `9534AAA6FBAA21750E4E9E49395C55BE8449AC0C5190667E1408671173D40A36` |

## Evidence source map

| Manuscript evidence | Project records |
|---|---|
| Locked Burgers primary comparison | `docs/FNO_REGULARIZATION_V1.md`, `docs/FNO_REGULARIZATION_V1_RESULTS.md`; `results/fno_regularization_v1/`; `results/fno_regularization_confirmation_v1/` |
| Burgers sparse/noisy stress tests | `docs/PHYSICS_STRESS_V1.md`, `docs/PHYSICS_STRESS_V1_RESULTS.md`; `results/physics_stress_v1/` |
| Burgers sample-count/resolution secondary study | `docs/PHYSICS_AS_SMOOTHNESS_V1.md`, `docs/PHYSICS_AS_SMOOTHNESS_V1_RESULTS.md`; `results/fno_physics_resolution_v1/` |
| Allen--Cahn development controls and fresh-trajectory confirmation | `docs/ALLEN_POINTWISE_CONTROL_V1.md`, `docs/ALLEN_POINTWISE_CONTROL_V1_RESULTS.md`, `docs/ALLEN_CAHN_FRESH_CONFIRMATION_V2.md`, `docs/ALLEN_CAHN_FRESH_CONFIRMATION_V2_RESULTS.md`; `results/allen_pointwise_control_v1/`, `results/allen_interval_replication_v1/`, `results/allen_interval_pilot_v1/`, and `results/allen_cahn_fresh_confirmation_v2/` |
| Public PDEBench source | DaRUS `PDEBench Datasets`, version 8.0, DOI `10.18419/darus-2986`; 1D Burgers HDF5 shape `(10000, 201, 1024)`; SHA-256 `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d` |

## Data-use and claim boundaries

- The 500-trajectory Burgers confirmation holdout was opened after protocol lock and is now consumed. The manuscript reports it only as the registered, one-time confirmation comparison.
- Burgers stress and sample-count/resolution studies reuse project splits and are secondary descriptive evidence. Their old seed-level bootstrap significance labels are not used as confirmatory inference.
- The Allen--Cahn development split uses the reused pool 200000--200299; 68 of its 200 development IDs were in an earlier training partition. V2 additionally evaluates the 24 frozen model fits on 200 new IDs 320000--320199 from the same declared initial-condition distribution. This is new-trajectory generalization evidence, not independent retraining, a distribution-shift test, or second-PDE validation. V1 IDs 310000--310199 were generated but retired without scoring after a cache-ID preflight failure; the erratum preserves the audit trail.
- Fresh V2 interval-minus-pointwise differences are small (+0.00163% dense and -0.00712% sparse relative to pointwise). The trajectory-bootstrap intervals are conditional on the three frozen seeds. No equivalence margin was specified; the manuscript makes no equivalence claim.
- The two Allen--Cahn constraints are close in the reported means, but no equivalence margin or formal equivalence test was specified. The paper makes no equivalence claim.
- The package does not include raw PDEBench trajectory caches, model/optimizer checkpoints, or a future unused holdout. The GitHub repository is currently private; a public code and derived-artifact archive with a stable URL and persistent identifier remains to be created before submission.

## Required before journal upload

Before submission, verify the author name, ORCID, corresponding email, affiliation and postal address, CRediT roles, no-funding and no-competing-interest statements, and generative-AI disclosure. Add the code/data repository URL and persistent identifier, verify whether the 2025 thesis has overlapping material that needs citation, confirm the no-concurrent-submission declaration, and regenerate and inspect both PDFs.
