# doc-verify

Vision-led multimodal verification of study-abroad application documents (COMP6001 final project).

## Setup

```bash
uv sync                      # create .venv and install everything
uv run jupyter lab           # notebooks
uv run python -m ipykernel install --user --name doc-verify   # optional: register kernel
```

Requires Tesseract 5 on the system (`brew install tesseract`).

## Layout

- `src/doc_verify/` – package code (data, models, training, evaluation); `tobacco.py` loads the Tobacco3482 reference scans
- `sad_syn/` – synthetic offer letters:
  - `generate.py` – clean/inconsistent offers and rule checks
  - `render.py` – offer dict → page image
  - `degrade.py` – scan-like degradation (label-blind, per-document seed)
  - `ocr.py` – OCR, field checks and confidence
  - `pipeline.py` – render → degrade → OCR for many letters in parallel
  - `analysis.py` – accuracy summaries, clean-vs-altered tests, ablations, confidence comparison
- `fonts/` – font files bundled for rendering
- `data/raw` – Tobacco3482; `data/sad_syn` – generated SAD-Syn set; `data/processed` – splits + OCR
- `notebooks/` – exploration and experiments
- `models/` – checkpoints (git-ignored)
- `reports/figures/` – confusion matrices, plots
- `scripts/` – CLI entry points (dataset generation, OCR, training)
