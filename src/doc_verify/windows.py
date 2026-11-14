"""Overlapping CLIP token windows over OCR text, and the frozen CLIP text features cached for fusion training."""

import math

import numpy as np

from doc_verify.text_length import CLIP_MODEL, CLIP_TEXT_TOKENS

CLIP_CONTEXT = 77
CLIP_TEXT_DIM = 512

# windows overlap by 25 tokens, so any value up to 26 tokens long appears whole in at least one window (tests/test_windows.py)
WINDOW_STRIDE = 50

# 8 overlapping windows (425 tokens) cover every SAD-Syn page; longer Tobacco pages are cut at 425 tokens
TEXT_WINDOWS = 8


def windows_needed(n_tokens, size=CLIP_TEXT_TOKENS, stride=WINDOW_STRIDE):
    if n_tokens <= size:
        return 1
    return 1 + math.ceil((n_tokens - size) / stride)


def tokens_covered(n_windows, size=CLIP_TEXT_TOKENS, stride=WINDOW_STRIDE):
    return size + (n_windows - 1) * stride


def window_ids(text, tokenizer, n_windows, stride=WINDOW_STRIDE):
    """
    Cut the page's tokens into n_windows overlapping windows, each wrapped in CLIP's start/end markers and padded
    to 77. Return (input_ids, attention_mask), both shaped (n_windows, 77). Windows past the end of the page are all
    padding (mask 0); text past the last window is dropped.
    """
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    input_ids = np.full((n_windows, CLIP_CONTEXT), tokenizer.pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((n_windows, CLIP_CONTEXT), dtype=np.int64)

    for w in range(min(n_windows, windows_needed(len(ids), stride=stride))):
        start = w * stride
        row = [tokenizer.bos_token_id] + ids[start:start + CLIP_TEXT_TOKENS] + [tokenizer.eos_token_id]
        input_ids[w, :len(row)] = row
        attention_mask[w, :len(row)] = 1

    return input_ids, attention_mask


def cache_text_features(texts, out_prefix, n_windows, batch_size=64, device=None):
    """
    Run the frozen CLIP text encoder over every window of every page and save its per-token outputs.
    Writes {out_prefix}_features.npy, float16 (n_pages, n_windows, 77, 512), and {out_prefix}_mask.npy, bool
    (n_pages, n_windows, 77). Rows are in the order of `texts`.
    """
    import torch
    from transformers import CLIPTextModel, CLIPTokenizerFast

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL)
    model = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device).eval()
    if device == "cuda":
        model = model.half()

    shape = (len(texts), n_windows, CLIP_CONTEXT)
    features = np.lib.format.open_memmap(f"{out_prefix}_features.npy", mode="w+", dtype=np.float16,
                                         shape=shape + (CLIP_TEXT_DIM,))
    masks = np.zeros(shape, dtype=bool)

    windows = [window_ids(text, tokenizer, n_windows) for text in texts]
    flat_ids = np.concatenate([ids for ids, _ in windows])
    flat_mask = np.concatenate([mask for _, mask in windows])

    with torch.no_grad():
        for start in range(0, len(flat_ids), batch_size):
            ids = torch.from_numpy(flat_ids[start:start + batch_size]).to(device)
            mask = torch.from_numpy(flat_mask[start:start + batch_size]).to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            for offset, row in enumerate(hidden.float().cpu().numpy().astype(np.float16)):
                page, window = divmod(start + offset, n_windows)
                features[page, window] = row

    for page, (_, mask) in enumerate(windows):
        masks[page] = mask.astype(bool)

    features.flush()
    np.save(f"{out_prefix}_mask.npy", masks)
