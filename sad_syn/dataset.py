"""Build the SAD-Syn dataset: paired clean/altered letters, degraded, OCR'd and recorded in a manifest."""

import random
from pathlib import Path

import pandas as pd
from PIL import Image

from doc_verify.config import SPLIT
from sad_syn.degrade import degrade, draw_strength
from sad_syn.generate import (
    INCONSISTENCY_KINDS,
    make_clean_offer,
    make_inconsistent_offer,
    set_seed,
)
from sad_syn.ocr import check_ocr, count_ocr_reads, mean_confidence, ocr_page
from sad_syn.pipeline import DEGRADE_SEED
from sad_syn.render import render_offer


def make_pairs(n_pairs):
    """
    Return a list of (clean_offer, altered_offer) tuples. Cycle the altered kind through INCONSISTENCY_KINDS.
    """
    pairs = []

    for i in range(n_pairs):
        clean_offer = make_clean_offer()
        altered_kind = INCONSISTENCY_KINDS[i % len(INCONSISTENCY_KINDS)]
        altered_offer = make_inconsistent_offer(altered_kind)
        pairs.append((clean_offer, altered_offer))

    return pairs


def assign_split(pair_id, seed=42):
    """
    Return 'train', 'val' or 'test' for this pair, roughly 70/15/15. Same input -> same answer every time.
    """
    train_p, val_p, _ = SPLIT
    rng = random.Random(f"split:{seed}:{pair_id}")
    u = rng.random()

    if u < train_p:
        return 'train'
    elif u < train_p + val_p:
        return 'val'
    else:
        return 'test'


def build_document(offer, template, seed, image_path):
    """
    Draw the strength from the seed, render, degrade, SAVE the image, OCR that same image.
    Return one manifest row (a dict).
    """
    strength = draw_strength(seed)

    img = degrade(render_offer(offer, template), strength, seed)

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)

    saved_img = Image.open(image_path)
    text, confidences = ocr_page(saved_img)
    read, total = count_ocr_reads(offer, text)
    altered_field = offer.get("altered_field")

    return {
        "template": template["template_id"],
        "strength": strength,
        "seed": seed,
        "label": "clean" if altered_field is None else "altered",
        "inconsistency_type": offer.get("inconsistency_type", "clean"),
        "altered_field": altered_field,
        "altered_value_ocr_ok": None if altered_field is None else altered_field not in check_ocr(offer, text),
        "copies_read": read,
        "copies_total": total,
        "mean_confidence": mean_confidence(confidences),
        "image_path": str(image_path),
    }


def build_dataset(n_pairs, template, out_dir, seed=42):
    """
    Make pairs, give pair k the seed DEGRADE_SEED + k, assign a split, build the documents, and save the manifest.
    Return the manifest DataFrame.
    """
    set_seed(seed)

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    rows = []

    pairs = make_pairs(n_pairs)

    for k, (clean_offer, altered_offer) in enumerate(pairs):
        pair_seed = DEGRADE_SEED + k
        split = assign_split(k, seed)

        clean_path = images_dir / split / f"pair_{k:05d}_clean.png"
        altered_path = images_dir / split / f"pair_{k:05d}_altered.png"

        clean_row = build_document(clean_offer, template, pair_seed, clean_path)
        altered_row = build_document(altered_offer, template, pair_seed, altered_path)

        clean_row.update({"pair_id": k, "split": split})
        altered_row.update({"pair_id": k, "split": split})

        rows.append(clean_row)
        rows.append(altered_row)

    manifest = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest
