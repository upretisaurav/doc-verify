"""Tobacco3482 real scans, used as the reference for how degraded SAD-Syn pages should look to OCR."""

from datasets import load_dataset
from PIL import Image

from doc_verify.config import RAW_DIR

TOBACCO_DIR = RAW_DIR / "tobacco3482"


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
