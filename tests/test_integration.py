"""
Integration tests that call external services.
"""

import os

import numpy as np
import pytest
from PIL import Image

from lensure.utils import social_media


IMAGE_PATH = "tests/data/"


@pytest.mark.integration
def test_bluesky_returns_modified_image():
    """
    Uploads an image to Bluesky and verifies that the image served by their
    CDN is actually different from the original. Bluesky re-encodes images as
    JPEG via the CDN, so at minimum there will be compression artifacts.
    The test also checks that the result is not completely unrelated
    (overall structure is preserved).
    """
    original = Image.open(IMAGE_PATH + os.listdir(IMAGE_PATH)[0]).convert("RGB")

    result = social_media.bluesky_attack(original)

    assert isinstance(result, Image.Image)

    original_resized = original.resize(result.size, Image.LANCZOS)
    original_arr = np.array(original_resized, dtype=np.int32)
    result_arr = np.array(result, dtype=np.int32)

    diff = np.abs(original_arr - result_arr)

    assert diff.max() > 0, "Expected Bluesky to modify at least one pixel"

    assert diff.mean() < 30, (
        f"Mean pixel difference {diff.mean():.1f} is too large — "
        "Bluesky may have returned an unrelated image"
    )
