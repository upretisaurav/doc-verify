"""Summaries and statistical checks for OCR experiments on degraded letters."""

import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from sad_syn.degrade import EFFECTS
from sad_syn.pipeline import run_ocr

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
