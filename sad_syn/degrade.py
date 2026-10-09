"""Scan-like degradation of rendered pages. Never looks at the offer, so it can't depend on the label."""

import random

import cv2
import numpy as np
from PIL import Image, ImageFilter

MAX_ROTATION_DEG = 1.5
MAX_BLUR_RADIUS = 1.2
MAX_RESOLUTION_LOSS = 0.5
MAX_NOISE_STD = 10.0
MAX_STROKE_CHANGE = 1.0
MAX_THRESHOLD_SHIFT = 0.3
MAX_SPECKS = 600

EFFECTS = ('rotation', 'blur', 'resolution', 'noise', 'stroke', 'threshold', 'specks')


def degrade(img, strength, seed, effects=EFFECTS):
    img = img.convert('L')
    if strength == 0:
        return img

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    def amount():
        return rng.uniform(0, strength)

    def sign():
        return rng.choice([-1, 1])

    angle = sign() * amount() * MAX_ROTATION_DEG
    blur_radius = amount() * MAX_BLUR_RADIUS
    scale = 1 - amount() * MAX_RESOLUTION_LOSS
    noise_std = amount() * MAX_NOISE_STD
    stroke_change = sign() * amount() * MAX_STROKE_CHANGE
    threshold_shift = sign() * amount() * MAX_THRESHOLD_SHIFT
    n_specks = round(amount() * MAX_SPECKS)

    page_size = img.size

    if 'rotation' in effects:
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)

    if 'blur' in effects:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    if 'resolution' in effects:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BOX)

    pixels = np.asarray(img, dtype=np.float32)

    if 'noise' in effects:
        pixels = pixels + np_rng.normal(0, noise_std, pixels.shape)

    if 'stroke' in effects:
        kernel = np.ones((3, 3), np.uint8)
        grown = cv2.erode(pixels, kernel) if stroke_change > 0 else cv2.dilate(pixels, kernel)
        pixels = pixels + abs(stroke_change) * (grown - pixels)

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    if 'threshold' in effects:
        ink, paper = np.percentile(pixels, [1, 50])
        cutoff = (ink + paper) / 2 + threshold_shift * (paper - ink) / 2
        pixels = np.where(pixels > cutoff, 255, 0).astype(np.uint8)

    if 'specks' in effects:
        ys = np_rng.integers(0, pixels.shape[0], n_specks)
        xs = np_rng.integers(0, pixels.shape[1], n_specks)
        sizes = np_rng.integers(1, 3, n_specks)
        for y, x, size in zip(ys, xs, sizes):
            pixels[y:y + size, x:x + size] = 0

    return Image.fromarray(pixels).resize(page_size, Image.Resampling.NEAREST)
