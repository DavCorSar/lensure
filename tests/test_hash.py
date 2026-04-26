"""
This module tests the implementation of the hash with some
example images.
"""

import os

import pytest
from PIL import Image
import numpy as np

from lensure.roles.authority import Authority

IMAGE_PATH = "tests/data/"


def test_perceptual_hash_is_same_with_small_changes():
    """
    For each of the images, modifies a random pixel.
    The perceptual hash of each image is computed before
    and after the modification. The change is so small that
    the value of the perceptual hash must be the same.
    """

    rng = np.random.default_rng(42)

    for image_name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + image_name)
        hash1 = Authority.compute_perceptual_hash(img)
        img2 = img.copy()

        width, height = img2.size
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        pixels = img2.load()
        r, g, b = pixels[x, y]
        pixels[x, y] = (min(255, r + 1), g, b)
        hash2 = Authority.compute_perceptual_hash(img2)

        assert np.sum(hash1 != hash2) == 0


def test_perceptual_hash_changes_with_local_region_modification():
    """
    Modifies a contiguous region of the image.
    The perceptual hash should reflect structural change.
    """

    rng = np.random.default_rng(42)

    for image_name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + image_name)
        hash1 = Authority.compute_perceptual_hash(img)
        img2 = img.copy()

        pixels = np.array(img2)

        height, width, _ = pixels.shape

        region_size_h = int(height * 0.5)
        region_size_w = int(width * 0.5)

        start_y = rng.integers(0, height - region_size_h)
        start_x = rng.integers(0, width - region_size_w)

        pixels[start_y : start_y + region_size_h, start_x : start_x + region_size_w] = (
            rng.integers(0, 255, size=(region_size_h, region_size_w, 3))
        )

        img2 = Image.fromarray(pixels.astype(np.uint8))
        hash2 = Authority.compute_perceptual_hash(img2)

        assert np.sum(hash1 != hash2) > 0
        assert np.mean(hash1 != hash2) < 0.5


if __name__ == "__main__":
    pytest.main()
