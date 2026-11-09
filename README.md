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

- `src/doc_verify/` – package code (data, models, training, evaluation):
  - `config.py` – paths, split, seed and the shared `MANIFEST_COLUMNS`
  - `tobacco.py` – Tobacco3482: 200 DPI pages, OCR, stratified 70/15/15 split, manifest
  - `text_length.py` – OCR text length in CLIP tokens and where the evidence for a planted mistake falls
- `sad_syn/` – synthetic study-abroad documents (offer letters, transcripts, bank statements; 20 generated templates each, split 14/3/3 by template):
  - `generate.py` – offer letters: clean/inconsistent offers and rule checks
  - `render.py` – offer letters: templates, renderer, OCR values
  - `transcript.py`, `bank_statement.py` – the same four pieces for the other two types
  - `doc_types.py` – registry so the pipeline treats every type the same way
  - `templates.py` – template generator (fixed seed), tested font list, train/val/test split by template
  - `layout.py` – shared page drawing (wrapping, tables, overflow check) and formatting
  - `degrade.py` – scan-like degradation (label-blind, per-document seed, ink safety check)
  - `ocr.py` – OCR, field checks and confidence
  - `pipeline.py` – render → degrade → OCR for many documents in parallel
  - `analysis.py` – accuracy summaries, clean-vs-altered tests, ablations, confidence comparison
  - `dataset.py` – paired clean/altered dataset build: images, OCR text and `manifest.csv`
- `fonts/` – bundled fonts and their licences; only fonts that passed the OCR font test are used by templates
- `data/raw` – Tobacco3482 download; `data/sad_syn` – generated SAD-Syn set; `data/processed/tobacco3482` – Tobacco pages, OCR text and manifest (all git-ignored)
- `notebooks/` – exploration and experiments
- `models/` – checkpoints (git-ignored)
- `reports/figures/` – confusion matrices, plots
- `scripts/` – CLI entry points:
  - `uv run python scripts/build_sad_syn.py` – build SAD-Syn (3,000 images)
  - `uv run python scripts/build_tobacco.py` – OCR and split Tobacco3482 (3,482 pages)
