"""
Sanity checks before the test set is unlocked: every label belongs to its own document, the planted mistake is in the
text the model sees, and the cached text features belong to the page they are served with.
"""

import numpy as np
import pandas as pd

from doc_verify.data import PageDataset, source_dir
from doc_verify.windows import CLIP_CONTEXT, TEXT_WINDOWS, tokens_covered, window_ids


def read_text(page, data_root=None):
    return (source_dir(page.source, data_root) / page.text_path).read_text(encoding='utf-8')


def ground_truth_check(pages, n_pairs=1500, data_root=None):
    """
    One row per SAD-Syn page of `pages`, read through PageDataset (the training loader's dataset), next to the
    document it was generated from (rebuilt with plan_documents): the labels served, the rules the true values break,
    the evidence values a reader must compare, and the token position by which they all appear in the OCR text
    (None if OCR lost one). Clean pages are checked against the evidence fields of their partner's kind.
    `data_root` points at the Colab bundle; without it, the local build is read.
    """
    from doc_verify.text_length import EVIDENCE_FIELDS, evidence_end_token, load_tokenizer
    from sad_syn.dataset import plan_documents
    from sad_syn.doc_types import DOC_TYPES
    from sad_syn.ocr import expected_ocr_fields

    tokenizer = load_tokenizer()
    jobs = {str(job['image_path']): job for job in plan_documents(n_pairs)}
    sad = pages[pages.source == 'sad_syn'].reset_index(drop=True)
    partner_kind = sad[sad.label == 'altered'].set_index('pair_id').inconsistency_type
    dataset = PageDataset(sad, data_root)
    limit = tokens_covered(TEXT_WINDOWS)

    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        page = sad.iloc[i]
        doc, template = jobs[page.image_path]['doc'], jobs[page.image_path]['template']
        kind = partner_kind[page.pair_id]
        fields = expected_ocr_fields(doc, template)
        evidence = {field: fields[field] for field in EVIDENCE_FIELDS[page.doc_type][kind]}
        end = evidence_end_token(read_text(page, data_root), list(evidence.values()), tokenizer)
        broken = [rule for rule, ok in DOC_TYPES[page.doc_type].check_rules(doc).items() if not ok]
        rows.append({
            'page_index': item['page_index'],
            'served_auth': item['auth_label'],
            'served_kind': item['kind_label'],
            'manifest_label': page.label,
            'manifest_kind': page.inconsistency_type,
            'generated_kind': doc.get('inconsistency_type', 'clean'),
            'doc_type': page.doc_type,
            'rules_broken': broken,
            'expected_rule_broken': (not broken if page.label == 'clean'
                                     else DOC_TYPES[page.doc_type].expected_broken_rule[kind] in broken),
            'evidence': evidence,
            'evidence_end_token': end,
            'evidence_in_windows': end is not None and end <= limit,
        })
    return pd.DataFrame(rows)


def cache_check(pages, data_root, text_cache, sample=10, seed=0):
    """
    Compare the cached text features with a fresh computation. Every page's cached token mask must equal the mask of
    its own text's windows; for `sample` pages the CLIP features are recomputed and compared with the cached row and,
    as a control, with the next page's row. Returns a dict of the findings.
    """
    import torch
    from transformers import CLIPTextModel, CLIPTokenizerFast

    from doc_verify.text_length import CLIP_MODEL

    tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL)
    features = np.load(f"{text_cache}_features.npy", mmap_mode='r')
    masks = np.load(f"{text_cache}_mask.npy", mmap_mode='r')

    mismatched = []
    for page in pages.itertuples():
        _, mask = window_ids(read_text(page, data_root), tokenizer, TEXT_WINDOWS)
        if not np.array_equal(mask.astype(bool), masks[page.page_index]):
            mismatched.append(page.page_index)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device).eval()
    chosen = pages.sample(sample, random_state=seed)
    own, neighbour = [], []
    with torch.no_grad():
        for page in chosen.itertuples():
            ids, mask = window_ids(read_text(page, data_root), tokenizer, TEXT_WINDOWS)
            hidden = model(input_ids=torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(mask).to(device)).last_hidden_state.float().cpu().numpy()
            real = mask.astype(bool)
            own.append(np.abs(hidden[real] - features[page.page_index][real].astype(np.float32)).max())
            other = (page.page_index + 1) % len(features)
            neighbour.append(np.abs(hidden[real] - features[other][real].astype(np.float32)).max())

    return {
        'pages': len(pages),
        'cached_rows': len(features),
        'window_shape_ok': features.shape[1:3] == (TEXT_WINDOWS, CLIP_CONTEXT),
        'mask_mismatches': len(mismatched),
        'first_mismatches': mismatched[:10],
        'max_diff_own_row': float(max(own)),
        'min_diff_next_row': float(min(neighbour)),
    }


def predictions_check(predictions, pages):
    """Number of saved predictions whose labels disagree with the page table at their page_index (should be 0)."""
    merged = predictions.merge(pages, on='page_index', suffixes=('', '_page'))
    columns = [c for c in ('type_label', 'auth_label', 'kind_label') if c in predictions]
    wrong = sum(int((merged[c] != merged[f'{c}_page']).sum()) for c in columns)
    return {'rows': len(predictions), 'matched': len(merged), 'label_disagreements': wrong}
