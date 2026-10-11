"""Registry of SAD-Syn document types, so the pipeline can treat every type the same way."""

from collections.abc import Callable
from dataclasses import dataclass

from sad_syn.bank_statement import (
    BANK_EXPECTED_BROKEN_RULE,
    BANK_PRINTED_TWICE,
    BANK_TEMPLATES,
    check_statement_rules,
    make_clean_statement,
    make_inconsistent_statement,
    render_statement,
    statement_ocr_fields,
)
from sad_syn.generate import (
    EXPECTED_BROKEN_RULE,
    check_rules,
    make_clean_offer,
    make_inconsistent_offer,
)
from sad_syn.render import OFFER_PRINTED_TWICE, OFFER_TEMPLATES, offer_ocr_fields, render_offer
from sad_syn.transcript import (
    TRANSCRIPT_EXPECTED_BROKEN_RULE,
    TRANSCRIPT_PRINTED_TWICE,
    TRANSCRIPT_TEMPLATES,
    check_transcript_rules,
    make_clean_transcript,
    make_inconsistent_transcript,
    render_transcript,
    transcript_ocr_fields,
)


@dataclass(frozen=True)
class DocType:
    make_clean: Callable
    make_inconsistent: Callable
    check_rules: Callable
    render: Callable
    ocr_fields: Callable
    printed_twice: tuple
    templates: list
    expected_broken_rule: dict


DOC_TYPES = {
    'offer_letter': DocType(
        make_clean_offer, make_inconsistent_offer, check_rules, render_offer,
        offer_ocr_fields, OFFER_PRINTED_TWICE, OFFER_TEMPLATES, EXPECTED_BROKEN_RULE,
    ),
    'transcript': DocType(
        make_clean_transcript, make_inconsistent_transcript, check_transcript_rules, render_transcript,
        transcript_ocr_fields, TRANSCRIPT_PRINTED_TWICE, TRANSCRIPT_TEMPLATES, TRANSCRIPT_EXPECTED_BROKEN_RULE,
    ),
    'bank_statement': DocType(
        make_clean_statement, make_inconsistent_statement, check_statement_rules, render_statement,
        statement_ocr_fields, BANK_PRINTED_TWICE, BANK_TEMPLATES, BANK_EXPECTED_BROKEN_RULE,
    ),
}


def render_document(doc, template, output_filename=None):
    return DOC_TYPES[doc['doc_type']].render(doc, template, output_filename)


def rule_test(doc_type, n=1000):
    kind = DOC_TYPES[doc_type]
    counts = {'clean': sum(all(kind.check_rules(kind.make_clean()).values()) for _ in range(n))}
    for inconsistency, expected_rule in kind.expected_broken_rule.items():
        counts[inconsistency] = sum(
            [name for name, ok in kind.check_rules(kind.make_inconsistent(inconsistency)).items() if not ok]
            == [expected_rule]
            for _ in range(n)
        )
    return counts
