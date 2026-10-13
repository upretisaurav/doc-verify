"""Summaries and statistical checks for OCR experiments on degraded letters."""

import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from sad_syn.degrade import EFFECTS
from sad_syn.doc_types import DOC_TYPES
from sad_syn.generate import INCONSISTENCY_KINDS, set_seed
from sad_syn.pipeline import run_ocr, run_ocr_sampled

CONFIDENCE_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def ocr_accuracy(results, by):
    sums = results.groupby(by)[['copies_read', 'copies_total']].sum()
    return sums['copies_read'] / sums['copies_total']


def strength_summary(results):
    summary = results.groupby('strength').agg(
        copies_read=('copies_read', 'sum'),
        copies_total=('copies_total', 'sum'),
        altered_value_ocr_ok=('altered_value_ocr_ok', 'mean'),
    )
    summary['ocr_accuracy'] = summary['copies_read'] / summary['copies_total']
    return summary


def label_gap_tests(results):
    tests = []
    for strength, group in results.groupby('strength'):
        clean = group[group['label'] == 'clean']
        altered = group[group['label'] == 'altered']
        _, p_copies = proportions_ztest(
            [clean['copies_read'].sum(), altered['copies_read'].sum()],
            [clean['copies_total'].sum(), altered['copies_total'].sum()],
        )
        _, p_letters = stats.ttest_ind(clean['accuracy'], altered['accuracy'], equal_var=False)
        tests.append({
            'strength': strength,
            'clean': clean['copies_read'].sum() / clean['copies_total'].sum(),
            'altered': altered['copies_read'].sum() / altered['copies_total'].sum(),
            'letters_per_group': len(clean),
            'z_test_p (copies as units)': p_copies,
            'welch_t_p (letters as units)': p_letters,
        })
    return pd.DataFrame(tests).set_index('strength')


def _accuracy_with(offers, template, strength, effects):
    return ocr_accuracy(run_ocr(offers, template, [strength], effects=effects), 'strength').iloc[0]


def ablate_one_at_a_time(offers, template, strength):
    results = {'all effects': _accuracy_with(offers, template, strength, EFFECTS)}
    for effect in EFFECTS:
        results[f'{effect} only'] = _accuracy_with(offers, template, strength, (effect,))
    return pd.Series(results, name='ocr_accuracy')


def ablate_leave_one_out(offers, template, strength):
    results = {'all effects': _accuracy_with(offers, template, strength, EFFECTS)}
    for effect in EFFECTS:
        rest = tuple(e for e in EFFECTS if e != effect)
        results[f'without {effect}'] = _accuracy_with(offers, template, strength, rest)
    return pd.Series(results, name='ocr_accuracy')


def confidence_profile(confidences, reference):
    # a page with no text found counts as confidence 0, so blank pages aren't silently dropped
    filled = confidences.fillna(0)
    return pd.Series({
        'no_text': confidences.isna().mean(),
        **{f'q{int(q * 100)}': filled.quantile(q) for q in CONFIDENCE_QUANTILES},
        'ks_vs_reference': stats.ks_2samp(filled, reference.fillna(0)).statistic,
    })


def compare_confidence(results, reference, reference_name='reference'):
    comparison = {reference_name: confidence_profile(reference, reference)}
    for strength, group in results.groupby('strength'):
        comparison[f'sad-syn {strength}'] = confidence_profile(group['mean_confidence'], reference)
    return pd.DataFrame(comparison).T


def compare_candidates(results_by_name, reference, reference_name='reference'):
    comparison = {reference_name: confidence_profile(reference, reference)}
    for name, results in results_by_name.items():
        profile = confidence_profile(results['mean_confidence'], reference)
        profile['ocr_accuracy'] = results['copies_read'].sum() / results['copies_total'].sum()
        profile['mean_strength'] = results['strength'].mean()
        comparison[name] = profile
    return pd.DataFrame(comparison).T


def font_test(fonts, n=40, seed=21):
    """
    For each font, render the same n documents of every type in the same layout and report, per document type,
    the share of clean pages read in full and the OCR accuracy under the frozen degradation.
    """
    rows = []
    for font in fonts:
        row = {'font': font}
        for doc_type, kind in DOC_TYPES.items():
            set_seed(seed)
            docs = [
                kind.make_clean() if i % 2 == 0 else kind.make_inconsistent(INCONSISTENCY_KINDS[(i // 2) % 4])
                for i in range(n)
            ]
            template = {**kind.templates[0], 'font': font, 'font_size_main': 32, 'font_size_title': 48,
                        'margin': 150, 'line_spacing_factor': 1.4}

            clean = run_ocr(docs, template, [0])
            degraded = run_ocr_sampled(docs, template)
            row[f'{doc_type} clean'] = (clean['copies_read'] == clean['copies_total']).mean()
            row[f'{doc_type} degraded'] = degraded['copies_read'].sum() / degraded['copies_total'].sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index('font')
