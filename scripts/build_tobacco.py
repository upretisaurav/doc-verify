"""OCR and split Tobacco3482 into data/processed/tobacco3482: images at 200 DPI, OCR text and manifest.csv.

    uv run python scripts/build_tobacco.py
"""

import argparse
import shutil
import time

from doc_verify.config import SEED
from doc_verify.tobacco import TOBACCO_OUT_DIR, build_tobacco


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    shutil.rmtree(TOBACCO_OUT_DIR / "images", ignore_errors=True)

    start = time.time()
    manifest = build_tobacco(seed=args.seed)
    print(f"Processed {len(manifest)} Tobacco3482 pages in {(time.time() - start) / 60:.1f} min -> {TOBACCO_OUT_DIR}")
    print(manifest.groupby(["doc_type", "split"]).size().unstack()[["train", "val", "test"]].to_string())


if __name__ == "__main__":
    main()
