# Journal of Computational Science submission draft

This folder contains an English manuscript draft prepared with the provided Elsevier CAS single-column template. It is a substantially restructured paper on physics penalties versus matched smoothness controls. The previous DeepONet manuscript is preserved under `../archive/pre_jcs_20261026/` and is not part of this submission.

## Files

- `main.tex`, `references.bib`: main manuscript source.
- `supplement.tex`: supplementary methods, full results, provenance, and limitations.
- `highlights.txt`: separate highlights upload.
- `cover_letter.md`: editable cover letter draft.
- `submission_checklist.md`: items requiring author confirmation before upload.
- `plot_data.csv`, `scripts/plot_summary.py`: source values and generator for the main summary figure.
- `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst`: template support files copied from the provided `els-cas-templates` directory.
- `figures/relative_gains.pdf`: vector figure used in the manuscript.

The release PDFs are built under the repository's `output/pdf/` directory. The draft intentionally retains explicit placeholders for author identity, affiliation, funding, competing interests, code archive URL/DOI, and any required AI-use disclosure. These must be completed and checked against the journal's current instructions before submission.

## Build

From this directory in PowerShell:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts\plot_summary.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build supplement.tex
```

The checked PDF copies are saved as `../../output/pdf/jcs_manuscript.pdf` and `../../output/pdf/jcs_supplement.pdf` relative to this folder.

## Claim boundary

The primary independent evidence is the one-time locked Burgers holdout; it is now consumed. The Allen--Cahn pointwise/interval comparison and the Burgers stress studies reuse development data. The paper does not claim a new architecture, formal equivalence between pointwise and interval losses, universal physics-versus-smoothing rankings, cross-resolution transfer, or an untouched Allen--Cahn confirmation.
