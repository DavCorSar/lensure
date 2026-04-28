"""
This module tests the implementation of the embedding process.
"""

import os

import pytest
from PIL import Image

from lensure.roles.authority import Authority


IMAGE_PATH = "tests/data/"


def test_lsb_fixed_message_can_be_embedded_and_extracted():
    """
    For each of the sample images, embeds a fixed message using LSB.
    Then extracts the watermark and checks that the extracted message
    matches the original one.
    """

    message = b"fixed test message"

    for image_name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + image_name)

        authority = Authority(
            embed_og_hash=False,
            embed_method="LSB",
        )

        encoded = authority._encode_message(message.hex())

        watermarked_img = authority._embed_lsb(img, encoded)
        extracted = authority._extract_watermark_lsb(watermarked_img)

        assert extracted["signature"] == message


def test_dwt_fixed_message_can_be_embedded_and_extracted():
    """
    For each of the sample images, embeds a fixed message using DWT.
    Then extracts the watermark and checks that the extracted message
    matches the original one.
    """

    message = b"fixed test message"

    for image_name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + image_name)

        authority = Authority(
            embed_og_hash=False,
            embed_method="DWT",
        )

        encoded = authority._encode_message(message.hex())

        watermarked_img = authority._embed_dwt(img, encoded)
        extracted = authority._extract_watermark_dwt(watermarked_img)

        assert extracted["signature"] == message


if __name__ == "__main__":
    pytest.main()
