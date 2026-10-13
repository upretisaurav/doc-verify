"""Generate the layout templates for each document type from a fixed seed, and split them train/val/test."""

import random

from doc_verify.config import SPLIT

FONTS = [
    "LiberationSans-Regular.ttf",
    "LiberationSerif-Regular.ttf",
    "PTSans-Regular.ttf",
    "PTSerif-Regular.ttf",
    "Lato-Regular.ttf",
]

LAYOUT_OPTIONS = {
    "margin": [130, 140, 150, 160, 170],
    "font_size_main": [30, 31, 32, 33, 34],
    "line_spacing_factor": [1.3, 1.35, 1.4, 1.45],
    "header_style": ["centered", "left"],
    "date_format": ["%B %d, %Y", "%d %B %Y", "%d/%m/%Y", "%d %b %Y"],
}

TYPE_OPTIONS = {
    "offer_letter": {"fee_layout": ["table", "list"], "fees_first": [False, True]},
    "transcript": {"summary_first": [False, True], "title_align": ["left", "centered"]},
    "bank_statement": {"summary_first": [False, True], "title_align": ["left", "centered"]},
}

TEMPLATES_PER_TYPE = 20
TEMPLATE_SEED = 2026
SPLITS = ('train', 'val', 'test')


def split_sizes(n):
    """
    Return how many of n items go to train, val and test (70/15/15).
    """
    train_p, val_p, _ = SPLIT
    n_train = round(n * train_p)
    n_val = round(n * val_p)
    return n_train, n_val, n - n_train - n_val


def make_templates(doc_type, n=TEMPLATES_PER_TYPE, seed=TEMPLATE_SEED):
    rng = random.Random(f"templates:{doc_type}:{seed}")
    options = {**LAYOUT_OPTIONS, **TYPE_OPTIONS[doc_type]}

    templates = []
    seen = set()
    while len(templates) < n:
        settings = {"font": FONTS[len(templates) % len(FONTS)]}
        settings.update({name: rng.choice(choices) for name, choices in options.items()})

        layout = tuple(settings.items())
        if layout in seen:
            continue
        seen.add(layout)

        settings["font_size_title"] = round(settings["font_size_main"] * 1.5)
        templates.append({"template_id": f"{doc_type}_{len(templates) + 1:02d}", **settings})

    return templates


def split_templates(doc_type, templates, seed=TEMPLATE_SEED):
    """
    Return {'train': [...], 'val': [...], 'test': [...]} with every template in exactly one split.
    Within val and within test no font repeats, so neither split depends on a single font.
    """
    shuffled = list(templates)
    random.Random(f"template-split:{doc_type}:{seed}").shuffle(shuffled)

    _, n_val, n_test = split_sizes(len(shuffled))
    by_split = {'train': [], 'val': [], 'test': []}
    for template in shuffled:
        for split, size in (('test', n_test), ('val', n_val)):
            fonts_used = {held_out['font'] for held_out in by_split[split]}
            if len(by_split[split]) < size and template['font'] not in fonts_used:
                by_split[split].append(template)
                break
        else:
            by_split['train'].append(template)

    if (len(by_split['val']), len(by_split['test'])) != (n_val, n_test):
        raise ValueError(f"Could not fill val/test for {doc_type} without repeating a font")

    unseen = {template['font'] for template in templates} - {template['font'] for template in by_split['train']}
    if unseen:
        raise ValueError(f"Fonts {sorted(unseen)} never appear in a train template; val/test would test an unseen font")

    return by_split
