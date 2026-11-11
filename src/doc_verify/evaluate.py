"""Validation/test metrics for the three heads, computed only over pages that carry each label."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from doc_verify.data import KIND_CLASSES, MISSING, TYPE_CLASSES


def head_metrics(predictions):
    """
    `predictions` has one row per page with the true labels and the model outputs:
    type_pred, auth_score (probability of altered), auth_pred, kind_pred. Missing labels are skipped.
    """
    metrics = {}

    typed = predictions[predictions.type_label != MISSING]
    if len(typed) and 'type_pred' in typed:
        metrics['type_accuracy'] = accuracy_score(typed.type_label, typed.type_pred)
        metrics['type_macro_f1'] = f1_score(typed.type_label, typed.type_pred, average='macro',
                                            labels=range(len(TYPE_CLASSES)), zero_division=0)

    authed = predictions[predictions.auth_label != MISSING]
    if len(authed) and 'auth_score' in authed:
        metrics['auth_auc'] = roc_auc_score(authed.auth_label, authed.auth_score)
        metrics['auth_accuracy'] = accuracy_score(authed.auth_label, authed.auth_pred)
        metrics['auth_macro_f1'] = f1_score(authed.auth_label, authed.auth_pred, average='macro', zero_division=0)

    kinded = predictions[predictions.kind_label != MISSING]
    if len(kinded) and 'kind_pred' in kinded:
        metrics['kind_macro_f1'] = f1_score(kinded.kind_label, kinded.kind_pred, average='macro',
                                            labels=range(len(KIND_CLASSES)), zero_division=0)

    return metrics


def selection_score(metrics):
    """
    The single number early stopping and model selection use on validation: the mean of type macro-F1 and
    authenticity AUC (whichever of the two the model produces).
    """
    parts = [metrics[name] for name in ('type_macro_f1', 'auth_auc') if name in metrics]
    return float(np.mean(parts))


# ---- planned analyses, declared before the test set was unlocked ----

def pair_table(predictions, pages):
    """
    SAD-Syn pages with their authenticity score, plus the attributes of their pair (taken from the altered partner):
    kind, document type and whether OCR read the altered value.
    """
    sad = pages[pages.source == 'sad_syn']
    altered = sad[sad.label == 'altered'].set_index('pair_id')
    table = predictions[['page_index', 'auth_score']].merge(
        sad[['page_index', 'pair_id', 'auth_label']], on='page_index')
    table['kind'] = table.pair_id.map(altered.inconsistency_type)
    table['doc_type'] = table.pair_id.map(altered.doc_type)
    table['evidence'] = table.pair_id.map(altered.altered_value_ocr_ok).map({True: 'readable', False: 'unreadable'})
    return table


def pair_bootstrap_auc(table, n_boot=2000, seed=0):
    """AUC with a 95% interval from resampling clean/altered pairs (partners share a template and scan noise)."""
    auc = roc_auc_score(table.auth_label, table.auth_score)
    groups = [g[['auth_label', 'auth_score']].to_numpy() for _, g in table.groupby('pair_id')]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        drawn = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        samples.append(roc_auc_score(drawn[:, 0], drawn[:, 1]))
    low, high = np.percentile(samples, [2.5, 97.5])
    return auc, low, high


def auth_breakdown(predictions, pages, n_boot=2000, seed=0):
    """
    Authenticity AUC overall and within pairs of each inconsistency kind, document type and evidence readability.
    Each subgroup keeps both partners of its pairs, so clean and altered pages come from the same templates.
    """
    table = pair_table(predictions, pages)
    rows = []
    for group, value, subset in [('all', 'all', table)] + [
        (column, value, subset) for column in ('kind', 'doc_type', 'evidence')
        for value, subset in table.groupby(column)
    ]:
        auc, low, high = pair_bootstrap_auc(subset, n_boot, seed)
        rows.append({'group': group, 'value': value, 'pairs': subset.pair_id.nunique(),
                     'auc': auc, 'ci_low': low, 'ci_high': high})
    return pd.DataFrame(rows)


def choose_threshold(labels, scores):
    """The cut-off (predict altered when score >= t) with the best balanced accuracy, chosen on validation only."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    candidates = np.unique(scores)
    accuracy = [balanced_accuracy_score(labels, scores >= t) for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])


def type_bootstrap_f1(predictions, n_boot=2000, seed=0):
    """Type macro-F1 with a 95% interval from resampling Tobacco pages."""
    typed = predictions[predictions.type_label != MISSING]
    truth, pred = typed.type_label.to_numpy(), typed.type_pred.to_numpy()

    def f1(index):
        return f1_score(truth[index], pred[index], average='macro', labels=range(len(TYPE_CLASSES)), zero_division=0)

    rng = np.random.default_rng(seed)
    samples = [f1(rng.integers(0, len(truth), len(truth))) for _ in range(n_boot)]
    low, high = np.percentile(samples, [2.5, 97.5])
    return f1(np.arange(len(truth))), low, high


def type_confusion(predictions):
    """Per-class confusion matrix for document type (rows true, columns predicted)."""
    typed = predictions[predictions.type_label != MISSING]
    matrix = confusion_matrix(typed.type_label, typed.type_pred, labels=range(len(TYPE_CLASSES)))
    return pd.DataFrame(matrix, index=TYPE_CLASSES, columns=TYPE_CLASSES)


def mcnemar(correct_a, correct_b):
    """Exact McNemar test on two models' right/wrong decisions for the same pages."""
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {'only_a_right': only_a, 'only_b_right': only_b, 'p_value': float(p)}
