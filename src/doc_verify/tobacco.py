"""Tobacco3482 real scans: loading, the stratified split, OCR and the manifest."""

import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image

from doc_verify.config import MANIFEST_COLUMNS, PROCESSED_DIR, RAW_DIR, SEED
from sad_syn.ocr import mean_confidence, ocr_page, ocr_pool
from sad_syn.templates import SPLITS, split_sizes

TOBACCO_DIR = RAW_DIR / "tobacco3482"
TOBACCO_OUT_DIR = PROCESSED_DIR / "tobacco3482"

DOC_TYPE_NAMES = {
    "ADVE": "advertisement",
    "Email": "email",
    "Form": "form",
    "Letter": "letter",
    "Memo": "memo",
    "News": "news",
    "Note": "note",
    "Report": "report",
    "Resume": "resume",
    "Scientific": "scientific",
}


def load_tobacco(class_name=None):
    dataset = load_dataset(str(TOBACCO_DIR), split='train')
    if class_name is None:
        return dataset
    label = dataset.features['label'].names.index(class_name)
    return dataset.filter(lambda value: value == label, input_columns='label')


def to_200_dpi(img):
    # most pages are ~200 DPI fax scans; the wider ones are 300 DPI and get scaled to match SAD-Syn
    if img.width > 2000:
        img = img.resize((round(img.width * 2 / 3), round(img.height * 2 / 3)), Image.Resampling.LANCZOS)
    return img


def assign_splits(labels, seed=SEED):
    """
    Return the split of every page: within each class the pages are shuffled and cut 70/15/15.
    """
    splits = [None] * len(labels)
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        random.Random(f"tobacco-split:{seed}:{label}").shuffle(indices)

        start = 0
        for split, size in zip(SPLITS, split_sizes(len(indices))):
            for i in indices[start:start + size]:
                splits[i] = split
            start += size
    return splits


def save_and_ocr(img, image_path):
    """
    Scale the page to 200 DPI, SAVE it, OCR that same file and save its text. Return the mean word confidence.
    """
    img = to_200_dpi(img.convert('L'))

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)

    text, confidences = ocr_page(Image.open(image_path))
    image_path.with_suffix(".txt").write_text(text, encoding="utf-8")
    return mean_confidence(confidences)


def build_tobacco(out_dir=TOBACCO_OUT_DIR, seed=SEED, batch_size=50, max_workers=None):
    """
    OCR every Tobacco3482 page at 200 DPI, split it stratified by class, and write a manifest with the same
    columns as SAD-Syn's. Tobacco pages have no authenticity label, so `label` is left empty.
    Return the manifest DataFrame.
    """
    out_dir = Path(out_dir)
    dataset = load_tobacco()
    class_names = dataset.features['label'].names
    labels = dataset['label']
    splits = assign_splits(labels, seed)

    rows = []
    with ocr_pool(max_workers) as pool:
        for start in range(0, len(dataset), batch_size):
            indices = range(start, min(start + batch_size, len(dataset)))
            image_paths = [
                Path("images") / splits[i] / f"{DOC_TYPE_NAMES[class_names[labels[i]]]}_{i:04d}.png" for i in indices
            ]
            images = [example['image'] for example in dataset.select(indices)]
            confidences = pool.map(save_and_ocr, images, [out_dir / path for path in image_paths])

            for i, image_path, confidence in zip(indices, image_paths, confidences):
                rows.append({
                    "source": "tobacco3482",
                    "doc_type": DOC_TYPE_NAMES[class_names[labels[i]]],
                    "split": splits[i],
                    "mean_confidence": confidence,
                    "image_path": str(image_path),
                    "text_path": str(image_path.with_suffix(".txt")),
                })

    manifest = pd.DataFrame(rows).reindex(columns=MANIFEST_COLUMNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest
