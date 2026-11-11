"""Pack everything Colab needs into data/colab/doc_verify_bundle.zip: 224x224 pages, OCR text, manifests and code.

    uv run python scripts/prepare_colab.py

Upload the zip to Google Drive as MyDrive/doc_verify/doc_verify_bundle.zip and open notebooks/colab_train.ipynb.
"""

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from doc_verify.config import DATA_DIR, ROOT
from doc_verify.data import SOURCES, load_image, source_dir

BUNDLE = DATA_DIR / "colab" / "doc_verify_bundle"

# only the training-side modules: they need torch/timm/transformers, not Faker or Tesseract
CODE = ["__init__.py", "checks.py", "config.py", "data.py", "evaluate.py", "models.py", "train.py", "windows.py", "text_length.py"]


def shrink(args):
    source, image_path, text_path = args
    out = BUNDLE / source / image_path
    out.parent.mkdir(parents=True, exist_ok=True)
    load_image(source_dir(source) / image_path).save(out)
    shutil.copy(source_dir(source) / text_path, BUNDLE / source / text_path)


def main():
    import pandas as pd

    shutil.rmtree(BUNDLE, ignore_errors=True)
    jobs = []
    for source in SOURCES:
        manifest = pd.read_csv(source_dir(source) / "manifest.csv")
        (BUNDLE / source).mkdir(parents=True)
        manifest.to_csv(BUNDLE / source / "manifest.csv", index=False)
        jobs += [(source, row.image_path, row.text_path) for row in manifest.itertuples()]

    with ProcessPoolExecutor() as pool:
        list(pool.map(shrink, jobs, chunksize=64))

    code_dir = BUNDLE / "code" / "doc_verify"
    code_dir.mkdir(parents=True)
    for name in CODE:
        shutil.copy(ROOT / "src" / "doc_verify" / name, code_dir / name)

    archive = shutil.make_archive(str(BUNDLE), "zip", BUNDLE)
    size = Path(archive).stat().st_size / 2**20
    print(f"{len(jobs)} pages -> {archive} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
