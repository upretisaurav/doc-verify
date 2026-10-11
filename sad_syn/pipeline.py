"""Render, degrade and OCR letters: the per-document step shared by experiments and dataset generation."""

from itertools import repeat

import pandas as pd

from sad_syn.degrade import EFFECTS, STRENGTH_MIXTURE, degrade, draw_strength
from sad_syn.doc_types import render_document
from sad_syn.ocr import check_ocr, count_ocr_reads, mean_confidence, ocr_page, ocr_pool

DEGRADE_SEED = 10_000


def default_seeds(n):
    return [DEGRADE_SEED + i for i in range(n)]


def swapped_seeds(n):
    # neighbours i and i ^ 1 are one clean and one altered letter, so this swaps their degradation
    return [DEGRADE_SEED + (i ^ 1) for i in range(n)]


def ocr_letter(offer, template, strength, seed, effects=EFFECTS):
    img = degrade(render_document(offer, template), strength, seed, effects)
    text, confidences = ocr_page(img)
    read, total = count_ocr_reads(offer, text, template)
    altered_field = offer.get('altered_field')
    return {
        'strength': strength,
        'seed': seed,
        'label': 'clean' if altered_field is None else 'altered',
        'inconsistency_type': offer.get('inconsistency_type', 'clean'),
        'copies_read': read,
        'copies_total': total,
        'mean_confidence': mean_confidence(confidences),
        'altered_value_ocr_ok': None if altered_field is None else altered_field not in check_ocr(offer, text, template),
    }


def _run_jobs(job_offers, template, job_strengths, job_seeds, effects, max_workers):
    with ocr_pool(max_workers) as pool:
        rows = list(pool.map(ocr_letter, job_offers, repeat(template), job_strengths, job_seeds, repeat(effects)))
    results = pd.DataFrame(rows)
    results['altered_value_ocr_ok'] = results['altered_value_ocr_ok'].astype('boolean')
    results['accuracy'] = results['copies_read'] / results['copies_total']
    return results


def run_ocr(offers, template, strengths, effects=EFFECTS, seeds=None, max_workers=None):
    seeds = default_seeds(len(offers)) if seeds is None else seeds
    jobs = [(offer, strength, seed) for strength in strengths for offer, seed in zip(offers, seeds)]
    job_offers, job_strengths, job_seeds = zip(*jobs)
    return _run_jobs(job_offers, template, job_strengths, job_seeds, effects, max_workers)


def run_ocr_sampled(offers, template, mixture=STRENGTH_MIXTURE, effects=EFFECTS, seeds=None, max_workers=None):
    seeds = default_seeds(len(offers)) if seeds is None else seeds
    strengths = [draw_strength(seed, mixture) for seed in seeds]
    return _run_jobs(offers, template, strengths, seeds, effects, max_workers)
