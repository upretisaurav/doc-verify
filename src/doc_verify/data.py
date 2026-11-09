"""The combined page table (Tobacco3482 + SAD-Syn) with masked labels, and the torch dataset that serves it."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from doc_verify.config import IMAGE_SIZE, PROCESSED_DIR, SAD_SYN_DIR, TEST_SET_UNLOCKED

TYPE_CLASSES = ['advertisement', 'email', 'form', 'letter', 'memo', 'news', 'note', 'report', 'resume', 'scientific']
KIND_CLASSES = ['clean', 'altered_date', 'altered_amount', 'institution_mismatch', 'total_mismatch']

# a label that does not apply to a page is MISSING, and the masked loss skips it
MISSING = -1

SOURCES = ('tobacco3482', 'sad_syn')
LOCAL_DIRS = {'tobacco3482': PROCESSED_DIR / 'tobacco3482', 'sad_syn': SAD_SYN_DIR}


def source_dir(source, data_root=None):
    """
    Where a source's manifest lives: the local build, or `data_root/<source>` in the Colab bundle.
    """
    return LOCAL_DIRS[source] if data_root is None else Path(data_root) / source


def all_pages(data_root=None):
    """
    Every page of both sources, in a fixed order (Tobacco first, then SAD-Syn). `page_index` is the row of a page
    in this table, and the text-feature cache uses the same rows.
    """
    pages = pd.concat([pd.read_csv(source_dir(s, data_root) / 'manifest.csv') for s in SOURCES], ignore_index=True)
    pages['page_index'] = np.arange(len(pages))

    is_tobacco = pages.source == 'tobacco3482'
    pages['type_label'] = np.where(is_tobacco, pages.doc_type.map({c: i for i, c in enumerate(TYPE_CLASSES)}), MISSING)
    pages['auth_label'] = np.where(is_tobacco, MISSING, (pages.label == 'altered').astype(int))
    pages['kind_label'] = np.where(
        is_tobacco, MISSING, pages.inconsistency_type.map({c: i for i, c in enumerate(KIND_CLASSES)})
    )
    for column in ('type_label', 'auth_label', 'kind_label'):
        pages[column] = pages[column].fillna(MISSING).astype(int)
    return pages


def load_pages(splits=('train', 'val'), data_root=None):
    if 'test' in splits and not TEST_SET_UNLOCKED:
        raise PermissionError("The test split is locked until every modelling choice is frozen "
                              "(set TEST_SET_UNLOCKED in doc_verify/config.py).")
    pages = all_pages(data_root)
    return pages[pages.split.isin(splits)].reset_index(drop=True)


def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert('L')
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BOX)
    return img


class PageDataset:
    """
    Serves (image, text features, text mask, labels) for each page. Images are grey pages repeated to 3 channels
    in [0, 1]; the model applies its own normalisation. Text features come from the cache written by
    doc_verify.windows.cache_text_features (`text_cache` is its path prefix), read by page_index. The cache is opened
    lazily inside each worker, so the 4 GB file is memory-mapped rather than copied into every worker.
    """

    def __init__(self, pages, data_root=None, text_cache=None):
        self.pages = pages.reset_index(drop=True)
        self.data_root = data_root
        self.text_cache = text_cache
        self._features = None
        self._mask = None

    def __len__(self):
        return len(self.pages)

    def _open_cache(self):
        if self._features is None:
            self._features = np.load(f"{self.text_cache}_features.npy", mmap_mode='r')
            self._mask = np.load(f"{self.text_cache}_mask.npy", mmap_mode='r')

    def __getitem__(self, i):
        import torch

        page = self.pages.iloc[i]
        path = source_dir(page.source, self.data_root) / page.image_path
        pixels = torch.from_numpy(np.asarray(load_image(path), dtype=np.float32) / 255.0)
        item = {
            'image': pixels.unsqueeze(0).expand(3, -1, -1).contiguous(),
            'type_label': int(page.type_label),
            'auth_label': int(page.auth_label),
            'kind_label': int(page.kind_label),
            'page_index': int(page.page_index),
        }
        if self.text_cache is not None:
            self._open_cache()
            item['text'] = torch.from_numpy(np.array(self._features[page.page_index], dtype=np.float32))
            item['text_mask'] = torch.from_numpy(np.array(self._mask[page.page_index]))
        return item
