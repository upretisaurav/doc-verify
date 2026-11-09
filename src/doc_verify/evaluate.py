"""Validation/test metrics for the three heads, computed only over pages that carry each label."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
