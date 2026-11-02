# JCS submission checklist

Preflight status checked 2026-11-02. The JCS Guide for Authors linked by the official journal page returned HTTP 403 during this check, so verify journal-specific requirements in the live author guide/submission portal before final approval.

## Ready in the current package

- [x] Main manuscript, supplement, cover-letter draft, editable LaTeX sources, bibliography, and the two cited vector figures are present.
- [x] Main manuscript includes a 215-word abstract and six keywords; a separate five-item highlights file is present. Confirm the journal's current word/character limits in its live guide.
- [x] PDEBench dataset record, source-file version, and source-file hash are identified; the raw public dataset is not repackaged.
- [x] Manuscript reports the new Allen--Cahn trajectories as inference-only evaluation of frozen fits, not as retraining, a second-PDE replication, or a future untouched holdout.
- [x] Corresponding-author email in the manuscript and cover letter is `xiaochen-yan@qq.com`.

## Author confirmations before final submission

- [ ] Confirm the publication name Xiaochen Yan, ORCID `0009-0002-9280-4323`, new email, and the affiliation wording. Enter any postal address details required by the submission system; none have been invented in the manuscript.
- [ ] Confirm each single-author CRediT role reflects work personally performed.
- [ ] Reconfirm the no-funding and no-competing-interest statements, exclusive submission, and approval of the exact final files.
- [ ] Review and correct the generative-AI disclosure to cover the actual extent of assistance with research code, data analysis, plotting, and manuscript drafting. Elsevier's current policy says AI use in the research process should also be described in Methods. Personally verify the full text, methods, results, references, and every AI-assisted output.
- [ ] Check the 2025 Harbin Institute of Technology graduation thesis for overlapping text, code, data, or results; cite and disclose overlap where relevant. Former supervisors are not automatically manuscript coauthors; include them only for qualifying contributions to this paper.
- [ ] Recheck all reference entries and DOI links, equations and PDE parameters, split IDs, table/figure values, and data hashes against the source records.
- [ ] The bibliography currently has 12 entries. Review whether key work on matched smoothness controls, physics-informed losses, neural operators, PDEBench, and evaluation/derivative metrics is adequately covered; add relevant sources rather than padding the count.

## Repository and data statement

- [ ] The GitHub repository `https://github.com/xchenio/pde-physics-vs-smoothness` is private, and there is no public archive URL or DOI yet. The manuscript now states this accurately rather than promising an unpublished DOI. Check the journal-specific research-data policy in the live guide. A public, versioned code/protocol/derived-data release with a persistent identifier is strongly recommended for reproducibility; do not upload raw PDEBench data or checkpoints without checking permissions and size.
- [ ] If a public archive is created, update the manuscript, supplement, cover letter, references, and manifest with the exact release URL/DOI, then rebuild both PDFs. Otherwise retain an accurate access statement that matches the journal's policy and the author's willingness to provide materials.

## Upload and final proof

- [ ] Open the live JCS Guide for Authors and verify article type, abstract/keyword limits, reference style, review model, figure rules, required declarations, data policy, and file-type requirements.
- [ ] Prepare several qualified, independent reviewer suggestions if requested by the submission system; avoid recent collaborators and editorial-board members.
- [ ] Upload the main manuscript, supplement, `highlights.txt`, figure files, cover letter, and LaTeX source/support files in the roles requested by the system. Figures are embedded in the manuscript and also available separately as vector PDFs.
- [ ] A graphical abstract is optional and encouraged according to the previously reviewed guide; make one only if useful and reconfirm the current rule. An author biography is not currently known to be required; provide one only if the journal asks.
- [ ] Rebuild and inspect the main and supplementary PDFs after all author metadata and repository decisions are final; inspect the system-generated review PDF before approving the final submission.

## Evidence boundary already addressed

- [x] Allen--Cahn is a secondary, synthetic second-PDE study. The fresh set `320000--320199` evaluates the same 24 frozen fits; it is not independent retraining or a distribution-shift test.
- [x] The 500-trajectory Burgers holdout and all reported Allen--Cahn cases are consumed; do not call them an unused future holdout.
