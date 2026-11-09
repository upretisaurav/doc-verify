"""Rule-based authenticity baseline: pull the printed values out of the OCR text and apply each type's check_rules.

A rule is only judged when every value it needs was found in the OCR text; otherwise it is skipped. The checker is
given the true document type (it does not classify).
"""

import datetime
import re
from functools import partial

import pandas as pd

from sad_syn.doc_types import DOC_TYPES

DATE = r"(\d{2}/\d{2}/\d{4}|\d{1,2} [A-Z][a-z]{2,8} \d{4}|[A-Z][a-z]{2,8} \d{1,2}, \d{4})"
DATE_FORMATS = ['%d/%m/%Y', '%d %B %Y', '%d %b %Y', '%B %d, %Y']
MONEY = r"(?:A\$|£)\s?(\d{1,3}(?:,\s?\d{3})*(?:\.\s?\d{2})?)"
# OCR often drops stray marks (dashes, quotes, semicolons) between a label and its value
JUNK = r"[^A-Za-z0-9£$]{0,15}?"

RULE_FIELDS = {
    'offer_letter': {
        'date_order': ['date_of_issue', 'payment_deadline', 'orientation_date', 'commencement_date', 'completion_date'],
        'completion_matches_duration': ['commencement_date', 'completion_date', 'duration_years'],
        'total_matches_fee': ['first_year_tuition_fee', 'duration_years', 'total_estimated_tuition'],
        'deposit_below_fee': ['acceptance_deposit_header', 'acceptance_deposit_body', 'first_year_tuition_fee'],
        'deposits_match': ['acceptance_deposit_header', 'acceptance_deposit_body'],
        'university_names_match': ['university_header', 'university_body'],
    },
    'transcript': {
        'date_order': ['commencement_date', 'completion_date', 'date_of_issue'],
        'wams_match': ['wam_summary', 'wam_statement'],
        'university_names_match': ['university_header', 'university_body'],
        'credits_match_courses': ['total_credits', 'courses'],
    },
    'bank_statement': {
        'date_order': ['period_start', 'period_end', 'date_of_issue'],
        'closing_balances_match': ['closing_balance_summary', 'closing_balance_body'],
        'bank_names_match': ['bank_header', 'bank_body'],
        'balance_reconciles': ['opening_balance', 'total_credits', 'total_debits', 'closing_balance_summary'],
    },
}

# rules beyond the four kinds still point at the kind of mistake that would break them
EXTRA_RULE_KINDS = {'completion_matches_duration': 'altered_date', 'deposit_below_fee': 'altered_amount'}

PLACEHOLDER = datetime.date(2000, 1, 1)


def parse_date(text):
    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt).replace(tzinfo=datetime.UTC).date()
        except ValueError:
            continue
    return None


def parse_money(text, cents=False):
    digits = re.sub(r"[\s,]", "", text)
    if cents:
        return int(digits.replace(".", "")) if "." in digits else int(digits) * 100
    return int(float(digits))


def find(pattern, text, parse=lambda value: value):
    match = re.search(pattern, text)
    if match is None:
        return None
    try:
        return parse(match.group(1))
    except ValueError:
        return None


def column_block(text, labels, value_patterns, stop=None):
    """
    Tesseract sometimes reads a two-column block column by column: all the labels, then all the values. Find the
    labels in order, then each value pattern in order after them (and before `stop`, if given).
    Return the values, or Nones if any is missing.
    """
    block = re.search(JUNK.join(re.escape(label) for label in labels), text)
    if block is None:
        return [None] * len(value_patterns)
    end = text.find(stop, block.end()) if stop else -1
    rest = text[block.end():end if end != -1 else len(text)]

    values, position = [], 0
    for pattern in value_patterns:
        match = re.compile(pattern).search(rest, position)
        if match is None:
            return [None] * len(value_patterns)
        values.append(match.group(1))
        position = match.end()
    return values


def fill_from_block(fields, text, layout, stop=None):
    """Fill fields that the label-value patterns missed from a column-by-column reading of the same block."""
    names = [name for name, _, _ in layout]
    if all(fields[name] is not None for name in names):
        return
    raw = column_block(text, [label for _, label, _ in layout], [pattern for _, _, pattern in layout], stop)
    for (name, _, pattern), value in zip(layout, raw):
        if fields[name] is None and value is not None:
            parse = parse_date if pattern == DATE else layout_parser(name)
            try:
                fields[name] = parse(value)
            except ValueError:
                pass


def layout_parser(name):
    return partial(parse_money, cents=True) if name in BANK_MONEY_FIELDS else parse_money


BANK_MONEY_FIELDS = {'opening_balance', 'total_credits', 'total_debits', 'closing_balance_summary'}


def header_name(text, body_name):
    """The header is the first line of the page: take as many words as the body's name has."""
    if body_name is None:
        return None
    words = text.split()[:len(body_name.split())]
    return ' '.join(words).rstrip('.,;:') if words else None


def extract_offer(text):
    fields = {
        'date_of_issue': find(r"Date:" + JUNK + DATE, text, parse_date),
        'payment_deadline': find(r"Payment Deadline:" + JUNK + DATE, text, parse_date)
        or find(r"\bby " + DATE, text, parse_date),
        'orientation_date': find(r"Orientation:" + JUNK + DATE, text, parse_date),
        'commencement_date': find(r"Commencement:" + JUNK + DATE, text, parse_date),
        'completion_date': find(r"Completion:" + JUNK + DATE, text, parse_date),
        'duration_years': find(r"Duration:" + JUNK + r"(\d+(?:\.\d+)?) years?", text, float),
        'first_year_tuition_fee': find(r"First-Year Tuition:" + JUNK + MONEY, text, parse_money),
        'total_estimated_tuition': find(r"Estimated Total:" + JUNK + MONEY, text, parse_money),
        'acceptance_deposit_header': find(r"Deposit Required:" + JUNK + MONEY, text, parse_money),
        'acceptance_deposit_body': find(r"deposit of " + MONEY, text, parse_money),
        'university_body': find(r"admission to (.+?) in the ", text),
    }
    fill_from_block(fields, text, [
        ('first_year_tuition_fee', 'First-Year Tuition:', MONEY),
        ('total_estimated_tuition', 'Estimated Total:', MONEY),
        ('acceptance_deposit_header', 'Deposit Required:', MONEY),
        ('payment_deadline', 'Payment Deadline:', DATE),
    ])
    fields['university_header'] = header_name(text, fields['university_body'])
    return fields


def extract_transcript(text):
    codes = re.findall(r"\b[A-Z]{4} \d{4}\b", text)
    rows = re.findall(r"\b[A-Z]{4} \d{4}\W*[A-Za-z][A-Za-z ]*? ([36]) (\d{2}) \S+", text)
    # the credits rule is only judged when every course code found has a fully read row
    courses = [{'credits': int(credits)} for credits, _ in rows] if rows and len(rows) == len(codes) else None
    fields = {
        'commencement_date': find(r"Commenced:" + JUNK + DATE, text, parse_date),
        'completion_date': find(r"Completed:" + JUNK + DATE, text, parse_date),
        'date_of_issue': find(r"Date of issue:" + JUNK + DATE, text, parse_date),
        'total_credits': find(r"Total credits:" + JUNK + r"(\d+) units", text, int),
        'wam_summary': find(r"Weighted average mark:" + JUNK + r"(\d{2}\.\d)", text, float),
        'wam_statement': find(r"weighted average mark of (\d{2}\.\d)", text, float),
        'university_body': find(r"requirements of the .+? at (.+?) with a weighted", text),
        'courses': courses,
    }
    fields['university_header'] = header_name(text, fields['university_body'])
    return fields


def extract_statement(text):
    period = re.search(r"Statement period:" + JUNK + DATE + r" to " + DATE, text)
    cents = partial(parse_money, cents=True)
    fields = {
        'period_start': parse_date(period.group(1)) if period else None,
        'period_end': parse_date(period.group(2)) if period else None,
        'date_of_issue': find(r"Date of issue:" + JUNK + DATE, text, parse_date),
        'opening_balance': find(r"Opening balance:" + JUNK + MONEY, text, cents),
        'total_credits': find(r"Total credits:" + JUNK + MONEY, text, cents),
        'total_debits': find(r"Total debits:" + JUNK + MONEY, text, cents),
        'closing_balance_summary': find(r"Closing balance:" + JUNK + MONEY, text, cents),
        'closing_balance_body': find(r"account was " + MONEY, text, cents),
        'bank_body': find(r"((?:[A-Z][\w.'-]*|of)(?: (?:[A-Z][\w.'-]*|of))*) confirms that", text),
    }
    fill_from_block(fields, text, [
        ('opening_balance', 'Opening balance:', MONEY),
        ('total_credits', 'Total credits:', MONEY),
        ('total_debits', 'Total debits:', MONEY),
        ('closing_balance_summary', 'Closing balance:', MONEY),
    ], stop='confirms that')
    fields['bank_header'] = header_name(text, fields['bank_body'])
    return fields


EXTRACTORS = {'offer_letter': extract_offer, 'transcript': extract_transcript, 'bank_statement': extract_statement}


DATE_FIELDS = {'date_of_issue', 'payment_deadline', 'orientation_date', 'commencement_date', 'completion_date',
               'period_start', 'period_end'}
NAME_FIELDS = {'university_header', 'university_body', 'bank_header', 'bank_body'}


def placeholder_for(field):
    if field == 'courses':
        return []
    if field in DATE_FIELDS:
        return PLACEHOLDER
    if field in NAME_FIELDS:
        return ''
    return 0


def check_page(doc_type, text):
    """
    Return (failed_rules, checked_rules) for one page. Unreadable values are filled with placeholders so the type's
    own check_rules can run, but only rules whose values were all read are reported.
    """
    fields = EXTRACTORS[doc_type](text)
    doc = {field: (value if value is not None else placeholder_for(field)) for field, value in fields.items()}
    results = DOC_TYPES[doc_type].check_rules(doc)

    checked = [rule for rule, needed in RULE_FIELDS[doc_type].items() if all(fields[f] is not None for f in needed)]
    failed = [rule for rule in checked if not results[rule]]
    return failed, checked


def predict_kind(doc_type, failed):
    if not failed:
        return 'clean'
    kinds = {rule: kind for kind, rule in DOC_TYPES[doc_type].expected_broken_rule.items()} | EXTRA_RULE_KINDS
    return kinds[failed[0]]


def rule_checker_predictions(pages, text_root):
    """
    One row per SAD-Syn page: the number of failed rules (the authenticity score), the prediction (any rule failed)
    and the predicted kind (the kind whose rule failed first).
    """
    rows = []
    for page in pages.itertuples():
        text = (text_root / page.text_path).read_text(encoding='utf-8')
        failed, checked = check_page(page.doc_type, text)
        rows.append({
            'page_index': page.page_index,
            'doc_type': page.doc_type,
            'auth_label': page.auth_label,
            'kind_label': page.kind_label,
            'rules_checked': len(checked),
            'rules_total': len(RULE_FIELDS[page.doc_type]),
            'auth_score': len(failed),
            'auth_pred': int(bool(failed)),
            'kind_name': predict_kind(page.doc_type, failed),
        })
    return pd.DataFrame(rows)
