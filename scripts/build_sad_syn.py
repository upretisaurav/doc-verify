"""Build the SAD-Syn dataset into data/sad_syn: images, OCR text and manifest.csv.

    uv run python scripts/build_sad_syn.py            # 1,500 pairs = 3,000 images
"""

import argparse
import shutil
import time

from doc_verify.config import SAD_SYN_DIR, SEED
from sad_syn.dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # images from an earlier build would otherwise linger next to the new ones
    shutil.rmtree(SAD_SYN_DIR / "images", ignore_errors=True)

    start = time.time()
    manifest = build_dataset(args.pairs, SAD_SYN_DIR, seed=args.seed)
    print(f"Built {len(manifest)} SAD-Syn images in {(time.time() - start) / 60:.1f} min -> {SAD_SYN_DIR}")
    print(manifest.groupby(["split", "doc_type", "label"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
