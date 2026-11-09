"""How long OCR text is in CLIP tokens, and where in the token sequence the evidence for a planted mistake falls."""

import pandas as pd
from transformers import CLIPTokenizerFast

CLIP_MODEL = "openai/clip-vit-base-patch32"

# CLIP's window is 77 tokens including the start and end markers, leaving 75 for text
CLIP_TEXT_TOKENS = 75

# the printed values a reader must compare to spot each kind of planted mistake
EVIDENCE_FIELDS = {
    'offer_letter': {
        'altered_date': ['date_of_issue', 'payment_deadline', 'orientation_date'],
        'altered_amount': ['acceptance_deposit_header', 'acceptance_deposit_body'],
        'institution_mismatch': ['university_header', 'university_body'],
        'total_mismatch': ['first_year_tuition_fee', 'total_estimated_tuition'],
    },
    'transcript': {
        'altered_date': ['commencement_date', 'completion_date', 'date_of_issue'],
        'altered_amount': ['wam_summary', 'wam_statement'],
        'institution_mismatch': ['university_header', 'university_body'],
        'total_mismatch': ['total_credits'],
    },
    'bank_statement': {
        'altered_date': ['period_start', 'period_end', 'date_of_issue'],
        'altered_amount': ['closing_balance_summary', 'closing_balance_body'],
        'institution_mismatch': ['bank_header', 'bank_body'],
        'total_mismatch': ['opening_balance', 'total_credits', 'total_debits', 'closing_balance_summary'],
    },
}


def load_tokenizer():
    return CLIPTokenizerFast.from_pretrained(CLIP_MODEL)


def token_count(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


def find_occurrences(text, value):
    """
    (start, end) character spans in `text` where `value` appears, ignoring whitespace (OCR often splits a value in two).
    """
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    squashed = ''.join(text[i] for i in positions)
    target = ''.join(value.split())

    spans = []
    start = squashed.find(target)
    while start != -1:
        spans.append((positions[start], positions[start + len(target) - 1] + 1))
        start = squashed.find(target, start + 1)
    return spans


def evidence_end_token(text, values, tokenizer):
    """
    Token position by which every evidence value has appeared, or None if OCR lost one of them.
    A value printed for two fields (e.g. both deposit copies in a clean letter) must appear twice.
    """
    needed = pd.Series(values).value_counts()
    end_char = 0
    for value, copies in needed.items():
        spans = find_occurrences(text, value)
        if len(spans) < copies:
            return None
        end_char = max(end_char, spans[copies - 1][1])
    return token_count(text[:end_char], tokenizer)
