"""Build the SAD-Syn dataset: paired clean/altered documents, degraded, OCR'd and recorded in a manifest."""

from pathlib import Path

import pandas as pd
from PIL import Image

from doc_verify.config import MANIFEST_COLUMNS
from sad_syn.degrade import degrade, draw_strength
from sad_syn.doc_types import DOC_TYPES, render_document
from sad_syn.generate import INCONSISTENCY_KINDS, set_seed
from sad_syn.ocr import check_ocr, count_ocr_reads, mean_confidence, ocr_page, ocr_pool
from sad_syn.pipeline import DEGRADE_SEED
from sad_syn.templates import SPLITS, split_templates


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
    image_path.with_suffix(".txt").write_text(text, encoding="utf-8")

    read, total = count_ocr_reads(doc, text, template)
    altered_field = doc.get("altered_field")

    return {
        "source": "sad_syn",
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


def plan_documents(n_pairs, seed=42):
    """
    Generate every document of the dataset in build order, without rendering anything.
    Return a list of jobs: dicts with doc, template, seed, pair_id, split and image_path (relative).
    The same n_pairs and seed always give the same plan, so analyses can recover the exact documents.
    """
    n_templates = sum(len(kind.templates) for kind in DOC_TYPES.values())
    if n_pairs % n_templates:
        raise ValueError(f"n_pairs must be a multiple of {n_templates} so every template gets the same number of pairs")

    set_seed(seed)
    pairs_per_template = n_pairs // n_templates

    jobs = []
    k = 0
    for doc_type, kind in DOC_TYPES.items():
        by_split = split_templates(doc_type, kind.templates)

        for split in SPLITS:
            for template in by_split[split]:
                for clean_doc, altered_doc in make_pairs(doc_type, pairs_per_template):
                    for label, doc in (("clean", clean_doc), ("altered", altered_doc)):
                        jobs.append({
                            "doc": doc,
                            "template": template,
                            "seed": DEGRADE_SEED + k,
                            "pair_id": k,
                            "split": split,
                            "image_path": Path("images") / split / f"{template['template_id']}_{k:05d}_{label}.png",
                        })
                    k += 1

    return jobs


def build_dataset(n_pairs, out_dir, seed=42, max_workers=None):
    """
    Make the same number of pairs for every template of every document type. Templates are split 70/15/15 within
    each type, so no template is shared across splits, and a pair always stays inside one template.
    Pair k gets the seed DEGRADE_SEED + k. Save the images, the OCR text and the manifest.
    Return the manifest DataFrame.
    """
    out_dir = Path(out_dir)
    jobs = plan_documents(n_pairs, seed)

    with ocr_pool(max_workers) as pool:
        rows = list(pool.map(
            build_document,
            [job["doc"] for job in jobs],
            [job["template"] for job in jobs],
            [job["seed"] for job in jobs],
            [out_dir / job["image_path"] for job in jobs],
        ))

    for row, job in zip(rows, jobs):
        row.update({
            "pair_id": job["pair_id"],
            "split": job["split"],
            "image_path": str(job["image_path"]),
            "text_path": str(job["image_path"].with_suffix(".txt")),
        })

    manifest = pd.DataFrame(rows)[MANIFEST_COLUMNS]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest
