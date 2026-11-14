"""
Large files kept out of the repository and attached to its GitHub release instead: the 224x224 data bundle (pages,
OCR text, manifests) and the best checkpoint of each model evaluated on the test split.
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

from doc_verify.config import DATA_DIR, MODELS_DIR, RESULTS_DIR

RELEASE_URL = "https://github.com/upretisaurav/doc-verify/releases/download/v1.0-submission"
BUNDLE_DIR = DATA_DIR / "colab" / "doc_verify_bundle"
CHECKPOINTS = ("fused", "fused_no_kind", "image_vit", "resnet18")


def download(name, dest):
    """Download one release asset to `dest`, unless it is already there."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"downloading {name} ...", flush=True)
    urllib.request.urlretrieve(f"{RELEASE_URL}/{name}", partial)
    partial.rename(dest)
    return dest


def fetch_bundle():
    """The data bundle unpacked at BUNDLE_DIR (pass it as `data_root`). 66 MB."""
    if not (BUNDLE_DIR / "sad_syn" / "manifest.csv").exists():
        archive = download("doc_verify_bundle.zip", DATA_DIR / "colab" / "doc_verify_bundle.zip")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(BUNDLE_DIR)
    return BUNDLE_DIR


def fetch_checkpoint(run):
    """
    A run directory that `doc_verify.train.predict_run` can load: the run's config.json (committed under
    results/runs) next to its best.pt (downloaded from the release).
    """
    run_dir = MODELS_DIR / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(RESULTS_DIR / "runs" / run / "config.json", run_dir / "config.json")
    download(f"{run}.pt", run_dir / "best.pt")
    return run_dir
