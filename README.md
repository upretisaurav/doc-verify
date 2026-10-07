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

- `src/doc_verify/` – package code (data, models, training, evaluation)
- `sad_syn/` – synthetic offer-letter generator (`generate.py`) and renderer (`render.py`)
- `fonts/` – font files bundled for rendering
- `data/raw` – Tobacco3482; `data/sad_syn` – generated SAD-Syn set; `data/processed` – splits + OCR
- `notebooks/` – exploration and experiments
- `models/` – checkpoints (git-ignored)
- `reports/figures/` – confusion matrices, plots
- `scripts/` – CLI entry points (dataset generation, OCR, training)
