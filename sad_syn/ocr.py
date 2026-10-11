"""Check OCR output against the values the renderer printed on the page."""

import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

import pytesseract

from sad_syn.doc_types import DOC_TYPES


def expected_ocr_fields(doc, template):
    return DOC_TYPES[doc['doc_type']].ocr_fields(doc, template)


def printed_copies(doc, template):
    fields = expected_ocr_fields(doc, template)
    twice = [fields[field] for field in DOC_TYPES[doc['doc_type']].printed_twice]
    return Counter(list(fields.values()) + twice)


def squash(text):
    # OCR often splits one value into two words ("A$38, 500"), so values are matched with all whitespace removed
    return ''.join(text.split())


def check_ocr(doc, ocr_text, template):
    text = squash(ocr_text)
    copies = printed_copies(doc, template)
    return {field: value for field, value in expected_ocr_fields(doc, template).items()
            if text.count(squash(value)) < copies[value]}


def count_ocr_reads(doc, ocr_text, template):
    text = squash(ocr_text)
    copies = printed_copies(doc, template)
    read = sum(min(text.count(squash(value)), n) for value, n in copies.items())
    return read, sum(copies.values())


def ocr_page(img):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = [(word, float(conf)) for word, conf in zip(data['text'], data['conf'])
             if word.strip() and float(conf) >= 0]
    text = ' '.join(word for word, _ in words)
    return text, [conf for _, conf in words]


def mean_confidence(confidences):
    return sum(confidences) / len(confidences) if confidences else math.nan


def page_confidence(img):
    return mean_confidence(ocr_page(img)[1])


def _single_threaded_tesseract():
    os.environ['OMP_THREAD_LIMIT'] = '1'


def ocr_pool(max_workers=None):
    return ProcessPoolExecutor(max_workers=max_workers or os.cpu_count(), initializer=_single_threaded_tesseract)


def page_confidences(images, batch_size=100, max_workers=None):
    images = iter(images)
    confidences = []
    with ocr_pool(max_workers) as pool:
        while batch := list(islice(images, batch_size)):
            confidences += pool.map(page_confidence, batch)
    return confidences
