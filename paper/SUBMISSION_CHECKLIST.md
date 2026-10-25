# Submission checklist

Work through this before submitting. Items marked **(you)** need the student or advisor; the rest are already done, and how to re-check each is noted.

## 1. Venue (you)
- [ ] Choose the conference. Read its call for papers: page limit (main text and references), template version, deadline.
- [ ] **Anonymisation:** if the review is double-blind, remove the author block and acknowledgment, and avoid "our previous work" and links to the repository. Upload code as anonymous supplementary material, or omit it.
- [ ] Check whether supplementary material is allowed and in which format. If not, move the essentials of `supplementary.tex` into the paper or drop them.
- [ ] **AI-use policy:** read the venue's and your course's rules, and edit the acknowledgment placeholder in `main.tex` to match (keep it truthful).

## 2. Content (you + advisor)
- [ ] Read the whole paper and `docs/EXPLAINER.md`. Be ready to explain every number, figure and design choice.
- [ ] Advisor review of the framing, related work and limitations.
- [ ] Every claim points to a table or figure. No claim says our model is "state of the art" or "more robust than SAM2-UNet" (it is not).
- [ ] Confirm the COD10K-C numbers quoted in the text against its PDF (see `docs/NOVELTY_REPORT.md`, items marked [verify]).
- [ ] Re-run the novelty search shortly before submitting (`docs/NOVELTY_REPORT.md` §1), in case new work appeared.

## 3. Numbers and references (done; re-check after any change)
- [x] Every number in `main.tex` and `supplementary.tex` comes from macros in `tables/numbers.tex` or from the generated tables. Regenerate them all with:
  `python scripts/summarize_benchmark.py; python scripts/significance.py; python scripts/make_tables.py; python scripts/make_figures.py`.
- [x] `refs.bib`: every entry has a `% verified:` URL comment. Unverifiable references are not cited (see the agent report in PROGRESS.md).
- [ ] Many entries were verified from GitHub READMEs, publisher pages or search-index snippets. Before submitting, check each one on dblp or Crossref (authors, venue, year, pages). `ravi2024sam2` and `xiong2024sam2unet` keep their old keys, but the entries carry the published years.
- [ ] COD10K-C: check its exact 8 corruption types and severity parameters (its §3.2) against `configs/corruptions.yaml` and §III of `main.tex`. The search confirmed Gaussian blur and the Dice retention numbers; the full parameter list was not visible.
- [ ] PySODMetrics asks users to cite a newer report (arXiv:2607.00886). It is not verified and not cited; add it if you can confirm it.
- [ ] After any edit, rebuild and check the log for `undefined` citations or references and for overfull boxes.

## 4. Formatting (done; re-check after edits)
- [x] IEEEtran `conference` class, two columns, 10 pt.
- [x] Main text is 6 pages; references start on page 6 and end on page 7 (35 cited references). Check against the venue's limit.
- [x] Figures are vector PDFs (plots) or 300 dpi (image grids), readable at column width, with a colour-blind-safe palette.
- [ ] Re-run `python scripts/make_qualitative.py` on the laptop so that `fig_qualitative` (5 rows) and `fig_qualitative_full` use the final layout.
- [ ] Check fonts are embedded (`pdffonts main.pdf`); IEEE PDF eXpress if the venue requires it.
- [ ] Fill in the author names, affiliations and emails (or anonymise, see §1).

## 5. Release (Phase 9)
- [ ] The README reproduces every table with one command (§3 above).
- [ ] MIT licence for our code; third-party code keeps its licences (`src/models/third_party/*/LICENSE`); datasets are not redistributed.
- [ ] Publish the trained Ours-Full / Ours-Full-512 checkpoints (trainable parameters only, about 15 MB each) and the corruption seeds (`configs/corruptions.yaml`).
