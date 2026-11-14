"""Overlapping CLIP token windows: what they cover, and that any short value appears whole in at least one window."""

import numpy as np
import pytest

from doc_verify.text_length import CLIP_TEXT_TOKENS
from doc_verify.windows import (
    CLIP_CONTEXT,
    TEXT_WINDOWS,
    WINDOW_STRIDE,
    tokens_covered,
    window_ids,
    windows_needed,
)

PAD, BOS, EOS = 0, 1, 2


class CountingTokenizer:
    """Stand-in for CLIP's tokenizer: one token per word, numbered from 3 so they never clash with the markers."""

    pad_token_id, bos_token_id, eos_token_id = PAD, BOS, EOS

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [3 + i for i, _ in enumerate(text.split())]}


def page(n_tokens):
    return ' '.join('w' for _ in range(n_tokens))


def test_eight_windows_cover_425_tokens():
    assert tokens_covered(TEXT_WINDOWS) == 425
    assert windows_needed(CLIP_TEXT_TOKENS) == 1
    assert windows_needed(425) == TEXT_WINDOWS
    assert windows_needed(426) == TEXT_WINDOWS + 1


def test_windows_hold_the_right_tokens_and_drop_the_rest():
    ids, mask = window_ids(page(500), CountingTokenizer(), TEXT_WINDOWS)
    assert ids.shape == mask.shape == (TEXT_WINDOWS, CLIP_CONTEXT)
    for w in range(TEXT_WINDOWS):
        start = w * WINDOW_STRIDE
        row = ids[w][mask[w].astype(bool)]
        assert row[0] == BOS and row[-1] == EOS
        assert list(row[1:-1]) == [3 + i for i in range(start, start + CLIP_TEXT_TOKENS)]
    assert ids.max() == 3 + 424  # token 425 onwards is dropped


def test_short_page_leaves_later_windows_empty():
    _, mask = window_ids(page(100), CountingTokenizer(), TEXT_WINDOWS)
    assert mask.sum(axis=1).tolist() == [77, 52, 0, 0, 0, 0, 0, 0]


def contains_whole(ids, mask, value):
    for row, valid in zip(ids, mask.astype(bool)):
        tokens = row[valid].tolist()
        if any(tokens[i:i + len(value)] == value for i in range(len(tokens) - len(value) + 1)):
            return True
    return False


@pytest.mark.parametrize('length', [1, 10, 25, 26])
def test_any_value_up_to_26_tokens_appears_whole_in_one_window(length):
    ids, mask = window_ids(page(425), CountingTokenizer(), TEXT_WINDOWS)
    for start in range(425 - length + 1):
        value = [3 + i for i in range(start, start + length)]
        assert contains_whole(ids, mask, value), (start, length)


def test_a_27_token_value_can_be_split():
    ids, mask = window_ids(page(425), CountingTokenizer(), TEXT_WINDOWS)
    split = [s for s in range(425 - 27 + 1) if not contains_whole(ids, mask, [3 + i for i in range(s, s + 27)])]
    assert split  # the guarantee stops at 26 tokens
    assert np.all(np.array(split) % WINDOW_STRIDE == WINDOW_STRIDE - 1)
