"""
Tests for the User verification flow.
"""

import os

import pytest
import numpy as np
from PIL import Image

from lensure.roles.authority import Authority
from lensure.roles.user import User


IMAGE_PATH = "tests/data/"


@pytest.fixture
def image():
    return Image.open(IMAGE_PATH + os.listdir(IMAGE_PATH)[0])


def test_verify_accepts_clean_lsb_watermark_without_og_hash(image):
    """
    With embed_og_hash=False the signature is over the original hash, and
    verification checks it against the current hash. LSB embedding is used
    because it does not modify the Y channel and therefore does not alter
    the perceptual hash. DWT embedding may shift the hash by a few bits,
    which would break verification in this mode.
    """
    authority = Authority(embed_og_hash=False, embed_method="LSB")
    watermarked = authority.embed_watermark(image)
    result = User(authority).verify(watermarked)
    assert result["accepted"]
    assert result["signature_valid"]
    assert result["distance"] == 0


def test_verify_accepts_clean_dwt_watermark_with_og_hash(image):
    """
    With embed_og_hash=True the original hash is embedded alongside the
    signature, so verification compares the extracted hash (original) against
    the current hash rather than signing the current hash directly.
    This makes the mode robust to small perceptual shifts introduced by DWT.
    """
    authority = Authority(embed_og_hash=True, embed_method="DWT")
    watermarked = authority.embed_watermark(image)
    result = User(authority).verify(watermarked)
    assert result["accepted"]
    assert result["signature_valid"]


def test_verify_distance_within_threshold_for_unmodified_dwt_image(image):
    """
    DWT watermarking may shift a few bits of the perceptual hash due to
    coefficient quantisation. The distance between the embedded original hash
    and the hash of the watermarked image should remain within the acceptance
    threshold (< 10).
    """
    authority = Authority(embed_og_hash=True, embed_method="DWT")
    watermarked = authority.embed_watermark(image)
    result = User(authority).verify(watermarked)
    assert result["distance"] < 10


def test_verify_rejects_signature_from_different_authority(image):
    """
    A user backed by a different authority (different key pair) must reject
    a watermark it did not issue.
    """
    auth1 = Authority(embed_og_hash=False, embed_method="LSB")
    auth2 = Authority(embed_og_hash=False, embed_method="LSB")
    watermarked = auth1.embed_watermark(image)
    result = User(auth2).verify(watermarked)
    assert not result["signature_valid"]
    assert not result["accepted"]


def test_verify_rejects_unwatermarked_image():
    """
    An image with no watermark should not be accepted.
    Reed-Solomon will likely fail to decode, or the extracted signature
    will not verify against the authority's public key.
    """
    rng = np.random.default_rng(0)
    random_img = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    authority = Authority(embed_og_hash=False, embed_method="DWT")
    result = User(authority).verify(random_img)
    assert not result["accepted"]


if __name__ == "__main__":
    pytest.main()
