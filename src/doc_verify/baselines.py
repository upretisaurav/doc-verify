"""Text and zero-shot baselines: TF-IDF + logistic regression, and CLIP zero-shot document type.

Hyperparameters are fixed in advance, not tuned on validation.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from doc_verify.data import MISSING, TYPE_CLASSES, load_image, source_dir

CLIP_IMAGE_MODEL = "openai/clip-vit-base-patch32"
CLIP_PROMPT = "a scanned {}"
TYPE_PROMPT_NAMES = {
    'advertisement': 'advertisement', 'email': 'email', 'form': 'form', 'letter': 'letter', 'memo': 'memo',
    'news': 'news article', 'note': 'handwritten note', 'report': 'report', 'resume': 'resume',
    'scientific': 'scientific paper',
}


def page_texts(pages, data_root=None):
    return [(source_dir(p.source, data_root) / p.text_path).read_text(encoding='utf-8') for p in pages.itertuples()]


def tfidf_baseline(train_pages, val_pages, data_root=None):
    """
    One TF-IDF + logistic regression per head, each fitted only on the training pages that carry that label.
    Returns validation predictions in the same format as doc_verify.train.predict.
    """
    predictions = val_pages[['page_index', 'type_label', 'auth_label', 'kind_label']].copy()
    train_texts = np.array(page_texts(train_pages, data_root), dtype=object)
    val_texts = np.array(page_texts(val_pages, data_root), dtype=object)

    for head in ('type', 'auth', 'kind'):
        label = f'{head}_label'
        fit_rows = (train_pages[label] != MISSING).to_numpy()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        model = LogisticRegression(max_iter=5000)
        model.fit(vectorizer.fit_transform(train_texts[fit_rows]), train_pages[label][fit_rows])

        features = vectorizer.transform(val_texts)
        predictions[f'{head}_pred'] = model.predict(features)
        if head == 'auth':
            predictions['auth_score'] = model.predict_proba(features)[:, 1]
    return predictions


def projected(features):
    """CLIP's projected embedding: a plain tensor in transformers v4, the pooler_output of a model output in v5."""
    return getattr(features, 'pooler_output', features)


def clip_zero_shot(pages, data_root=None, device='cpu', batch_size=32):
    """Document type by CLIP zero-shot: the class whose prompt is closest to the page image."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_IMAGE_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_IMAGE_MODEL)
    prompts = [CLIP_PROMPT.format(TYPE_PROMPT_NAMES[name]) for name in TYPE_CLASSES]

    with torch.no_grad():
        text = processor(text=prompts, return_tensors='pt', padding=True).to(device)
        text_embeddings = projected(model.get_text_features(**text))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        predicted = []
        for start in range(0, len(pages), batch_size):
            batch = pages.iloc[start:start + batch_size]
            images = [load_image(source_dir(p.source, data_root) / p.image_path).convert('RGB')
                      for p in batch.itertuples()]
            pixels = processor(images=images, return_tensors='pt').to(device)
            image_embeddings = projected(model.get_image_features(**pixels))
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            predicted += (image_embeddings @ text_embeddings.T).argmax(-1).cpu().tolist()

    predictions = pages[['page_index', 'type_label', 'auth_label', 'kind_label']].copy()
    predictions['type_pred'] = predicted
    return predictions

