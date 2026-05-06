"""
This module tests the implementation of the embedding process.
"""

import os

import pytest
from PIL import Image

from lensure.roles.authority import Authority


IMAGE_PATH = "tests/data/"


def test_lsb_watermark_roundtrip():
    """
    Embeds a watermark using LSB and verifies it can be extracted without errors.
    """
    for image_name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + image_name)
        authority = Authority(embed_og_hash=False, embed_method="LSB")

        watermarked = authority.embed_watermark(img)
        extracted = authority.extract_watermark(watermarked)

        assert "decode_error" not in extracted
        assert len(extracted["signature"]) > 0


def test_dwt_watermark_roundtrip():
    """
    Embeds a watermark using DWT and verifies it can be extracted without errors.
    """
    for image_name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + image_name)
        authority = Authority(embed_og_hash=False, embed_method="DWT")

        watermarked = authority.embed_watermark(img)
        extracted = authority.extract_watermark(watermarked)

        assert "decode_error" not in extracted
        assert len(extracted["signature"]) > 0


if __name__ == "__main__":
    pytest.main()
