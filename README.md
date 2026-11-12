# ROMEO — Rapid Optical Mode Evaluation and Observation

Project page for the Royal Society Industry Fellowship ROMEO (Aston University × Arden Photonics, 2026–2029),
served at https://egor-manu.github.io/romeo/.

A single static page with no build step:

- `index.html`, `assets/style.css`, `assets/site.js`: the page, its styling and scroll behaviour.
- `assets/romeo.js`: the hero's live reconstruction and the interactive lab (forward model, direct PMP inversion
  with a stored pseudo-inverse, and rank-one projection, all in plain JavaScript).
- `assets/results.js`: the interactive result charts and the explorer of held-out experimental fields.
- `data/lab/`: the lab's simulated fibres (6, 10, 17 and 42 modes) in the paper's optical system with the optimised
  mask of its Figure 1. `index.json` lists them; each `<name>.json` + `<name>.bin` (loaded on demand, 28 MB for 42 modes)
  holds the detector-plane mode fields for the near field, a flat SLM, a random mask and the optimised mask, with the
  PMP pseudo-inverse (float16; float32 for the ill-conditioned near field and flat SLM), singular spectrum,
  condition number and light capture of each.
- `data/paper.json` + `data/paper.bin`: published results of the paper "Optical symmetry breaking for single-image field
  reconstruction": held-out modal coefficients and scores for the SLM and diffuser experiments, the released LP basis,
  the simulated noise statistics and the matched mask controls.
- `assets/figures/field_recovery.png`: an alternative concept figure, not one of the figures in the submitted paper.
  The figures of the submission are deliberately not reproduced here; the charts are redrawn from the data.
- `tools/export_demo_data.py` regenerates the lab data with `pmp_metal` (LP-mode solver, the paper's optical system
  and its Figure 1 mask); `tools/export_paper_data.py` regenerates the results data from the paper repository.

Data and code of the paper will become publicly available after journal publication: https://doi.org/10.5281/zenodo.23785123
