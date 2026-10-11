"""Build the SAD-Syn dataset: paired clean/altered documents, degraded, OCR'd and recorded in a manifest."""

from pathlib import Path

import pandas as pd
from PIL import Image

from doc_verify.config import SPLIT
from sad_syn.degrade import degrade, draw_strength
from sad_syn.doc_types import DOC_TYPES, render_document
from sad_syn.generate import INCONSISTENCY_KINDS, set_seed
from sad_syn.ocr import check_ocr, count_ocr_reads, mean_confidence, ocr_page, ocr_pool
from sad_syn.pipeline import DEGRADE_SEED

SPLITS = ('train', 'val', 'test')


def make_pairs(doc_type, n_pairs):
    """
    Return a list of (clean_doc, altered_doc) tuples. Cycle the altered kind through INCONSISTENCY_KINDS.
    """
    kind = DOC_TYPES[doc_type]
    pairs = []

    for i in range(n_pairs):
        clean_doc = kind.make_clean()
        altered_kind = INCONSISTENCY_KINDS[i % len(INCONSISTENCY_KINDS)]
        altered_doc = kind.make_inconsistent(altered_kind)
        pairs.append((clean_doc, altered_doc))

    return pairs


def split_sizes(n_pairs):
    """
    Return how many of n_pairs go to train, val and test (70/15/15).
    """
    train_p, val_p, _ = SPLIT
    n_train = round(n_pairs * train_p)
    n_val = round(n_pairs * val_p)
    return n_train, n_val, n_pairs - n_train - n_val


def build_document(doc, template, seed, image_path):
    """
    Draw the strength from the seed, render, degrade, SAVE the image, OCR that same image and save its text.
    Return one manifest row (a dict).
    """
    strength = draw_strength(seed)

    img = degrade(render_document(doc, template), strength, seed)

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)

    saved_img = Image.open(image_path)
    text, confidences = ocr_page(saved_img)
    image_path.with_suffix(".txt").write_text(text)

    read, total = count_ocr_reads(doc, text, template)
    altered_field = doc.get("altered_field")

    return {
        "doc_type": doc["doc_type"],
        "template": template["template_id"],
        "strength": strength,
        "seed": seed,
        "label": "clean" if altered_field is None else "altered",
        "inconsistency_type": doc.get("inconsistency_type", "clean"),
        "altered_field": altered_field,
        "altered_value_ocr_ok": None if altered_field is None else altered_field not in check_ocr(doc, text, template),
        "copies_read": read,
        "copies_total": total,
        "mean_confidence": mean_confidence(confidences),
    }


def build_dataset(n_pairs, out_dir, seed=42, max_workers=None):
    """
    Make the same number of pairs for every document type. Each type's first template is its train split,
    the second its val split and the third its test split, so no template is shared across splits.
    Pair k gets the seed DEGRADE_SEED + k. Save the images, the OCR text and the manifest.
    Return the manifest DataFrame.
    """
    if n_pairs % len(DOC_TYPES):
        raise ValueError(f"n_pairs must be a multiple of {len(DOC_TYPES)} so every document type gets the same number")

    set_seed(seed)

    out_dir = Path(out_dir)
    pairs_per_type = n_pairs // len(DOC_TYPES)

    job_docs = []
    job_templates = []
    job_seeds = []
    job_image_paths = []
    job_meta = []

    k = 0
    for doc_type, kind in DOC_TYPES.items():
        pairs = make_pairs(doc_type, pairs_per_type)
        splits = [split for split, size in zip(SPLITS, split_sizes(pairs_per_type)) for _ in range(size)]

        for (clean_doc, altered_doc), split in zip(pairs, splits):
            template = kind.templates[SPLITS.index(split)]
            pair_seed = DEGRADE_SEED + k

            for label, doc in (("clean", clean_doc), ("altered", altered_doc)):
                image_path = Path("images") / split / f"{doc_type}_{k:05d}_{label}.png"

                job_docs.append(doc)
                job_templates.append(template)
                job_seeds.append(pair_seed)
                job_image_paths.append(out_dir / image_path)
                job_meta.append({
                    "pair_id": k,
                    "split": split,
                    "image_path": str(image_path),
                    "text_path": str(image_path.with_suffix(".txt")),
                })

            k += 1

    with ocr_pool(max_workers) as pool:
        rows = list(pool.map(build_document, job_docs, job_templates, job_seeds, job_image_paths))

    for row, meta in zip(rows, job_meta):
        row.update(meta)

    manifest = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest
