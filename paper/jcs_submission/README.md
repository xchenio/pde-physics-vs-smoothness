# Journal of Computational Science submission draft

This folder contains an English manuscript draft prepared with the provided Elsevier CAS single-column template. It is a substantially restructured paper on physics penalties versus matched smoothness controls. The previous DeepONet manuscript is preserved under `../archive/pre_jcs_20261026/` and is not part of this submission.

## Files

- `main.tex`, `references.bib`: main manuscript source.
- `supplement.tex`: supplementary methods, full results, provenance, and limitations.
- `highlights.txt`: separate highlights upload.
- `cover_letter.md`: editable cover letter draft.
- `submission_checklist.md`: items requiring author confirmation before upload.
- `plot_data.csv`, `scripts/plot_summary.py`: source values and generator for Figure 1.
- `paired_burgers_differences.csv`, `scripts/plot_paired_burgers_differences.py`: per-trajectory paired differences and generator for Figure 2, derived from the consumed locked holdout.
- `cas-sc.cls`, `cas-common.sty`: CAS template support files. `elsarticle-num.bst` supplies the journal's numeric reference style; the original CAS sample `cas-model2-names.bst` is retained but not used.
- `figures/relative_gains.pdf`, `figures/paired_burgers_differences.pdf`: vector figures used in the manuscript.

The release PDFs are built under the repository's `output/pdf/` directory. The manuscript now contains Xiaochen Yan's ORCID, corresponding email, independent-researcher affiliation in Shenzhen, no-funding and no-competing-interest statements, and a draft generative-AI disclosure. The study's public archive URL and persistent identifier remain pending. The author must verify the CRediT roles, final AI disclosure, postal-address details, thesis-overlap statement, and all declarations before submission.

## Build

From this directory in PowerShell:

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' scripts\plot_summary.py
& 'D:\miniconda3\envs\pytorch\python.exe' scripts\plot_paired_burgers_differences.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build supplement.tex
```

The checked PDF copies are saved as `../../output/pdf/jcs_manuscript.pdf` and `../../output/pdf/jcs_supplement.pdf` relative to this folder. The manuscript PDF contains the CAS-generated highlights page; upload `highlights.txt` separately as the journal requires.

## Claim boundary

The primary independent evidence is the one-time locked Burgers holdout; it is now consumed. The Allen--Cahn pointwise/interval comparison and the Burgers stress studies reuse development data. The paper does not claim a new architecture, formal equivalence between pointwise and interval losses, universal physics-versus-smoothing rankings, cross-resolution transfer, or an untouched Allen--Cahn confirmation.
