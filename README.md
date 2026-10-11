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
- `sad_syn/` – synthetic study-abroad documents (offer letters, transcripts, bank statements; 3 templates each):
  - `generate.py` – offer letters: clean/inconsistent offers and rule checks
  - `render.py` – offer letters: templates, renderer, OCR values
  - `transcript.py`, `bank_statement.py` – the same four pieces for the other two types
  - `doc_types.py` – registry so the pipeline treats every type the same way
  - `layout.py` – shared page drawing (wrapping, tables, overflow check) and formatting
  - `degrade.py` – scan-like degradation (label-blind, per-document seed, ink safety check)
  - `ocr.py` – OCR, field checks and confidence
  - `pipeline.py` – render → degrade → OCR for many documents in parallel
  - `analysis.py` – accuracy summaries, clean-vs-altered tests, ablations, confidence comparison
  - `dataset.py` – paired clean/altered dataset build: images, OCR text and `manifest.csv`
- `fonts/` – font files bundled for rendering
- `data/raw` – Tobacco3482; `data/sad_syn` – generated SAD-Syn set; `data/processed` – splits + OCR
- `notebooks/` – exploration and experiments
- `models/` – checkpoints (git-ignored)
- `reports/figures/` – confusion matrices, plots
- `scripts/` – CLI entry points (dataset generation, OCR, training)
