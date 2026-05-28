"""
Tests for Authority public API: key operations and full watermark roundtrip.
"""

import os

import pytest
from PIL import Image

from lensure.roles.authority import Authority


IMAGE_PATH = "tests/data/"


@pytest.fixture
def image():
    return Image.open(IMAGE_PATH + os.listdir(IMAGE_PATH)[0])


@pytest.fixture
def authority():
    return Authority(embed_og_hash=False, embed_method="DWT")


def test_sign_and_verify_same_hash(authority, image):
    h = authority.compute_perceptual_hash(image)
    sig = authority.sign_hash(h)
    assert authority.verify_signature(h, sig)


def test_verify_signature_with_empty_signature_returns_false(authority, image):
    h = authority.compute_perceptual_hash(image)
    assert not authority.verify_signature(h, b"")


def test_verify_signature_from_different_authority_returns_false(image):
    auth1 = Authority(embed_og_hash=False)
    auth2 = Authority(embed_og_hash=False)
    h = auth1.compute_perceptual_hash(image)
    sig = auth1.sign_hash(h)
    assert not auth2.verify_signature(h, sig)


def test_invalid_embed_method_raises_value_error():
    with pytest.raises(ValueError, match="is not a valid method"):
        Authority(embed_og_hash=False, embed_method="INVALID")


def test_embed_watermark_with_og_hash_roundtrip():
    """
    When embed_og_hash=True, the extracted payload contains both
    the original hash and the signature.
    """
    for image_name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + image_name)
        authority = Authority(embed_og_hash=True, embed_method="DWT")

        watermarked, _ = authority.embed_watermark(img)
        extracted = authority.extract_watermark(watermarked)

        assert "decode_error" not in extracted
        assert "hash" in extracted
        assert len(extracted["signature"]) > 0


if __name__ == "__main__":
    pytest.main()
