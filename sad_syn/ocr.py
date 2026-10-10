"""Check OCR output against the values the renderer printed on the page."""

import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

import pytesseract

from sad_syn.render import format_currency, format_date


def expected_ocr_fields(offer):
    currency = offer['currency']
    return {
        'university_header': offer['university_header'],
        'university_body': offer['university_body'],
        'acceptance_deposit_header': format_currency(offer['acceptance_deposit_header'], currency),
        'acceptance_deposit_body': format_currency(offer['acceptance_deposit_body'], currency),
        'first_year_tuition_fee': format_currency(offer['first_year_tuition_fee'], currency),
        'total_estimated_tuition': format_currency(offer['total_estimated_tuition'], currency),
        'date_of_issue': format_date(offer['date_of_issue']),
        'payment_deadline': format_date(offer['payment_deadline']),
        'orientation_date': format_date(offer['orientation_date']),
        'commencement_date': format_date(offer['commencement_date']),
        'completion_date': format_date(offer['completion_date']),
    }


def printed_copies(offer):
    fields = expected_ocr_fields(offer)
    return Counter(list(fields.values()) + [fields['payment_deadline']])


def check_ocr(offer, ocr_text):
    text = ' '.join(ocr_text.split())
    copies = printed_copies(offer)
    return {field: value for field, value in expected_ocr_fields(offer).items()
            if text.count(value) < copies[value]}


def count_ocr_reads(offer, ocr_text):
    text = ' '.join(ocr_text.split())
    copies = printed_copies(offer)
    read = sum(min(text.count(value), n) for value, n in copies.items())
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
