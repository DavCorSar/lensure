"""
Tests for the sign_method parameter (RSA vs ECDSA).

Covers:
- Validation of the sign_method parameter.
- ECDSA key properties and signature length.
- Sign / verify correctness for ECDSA.
- DWT embedding roundtrip with ECDSA.
- User.verify acceptance with ECDSA.
- Payload size and PSNR improvements of ECDSA over RSA.
- Settings integration.
"""

import json
import os
import tempfile

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from lensure.roles.authority import Authority
from lensure.roles.user import User
from lensure.settings import Settings


IMAGE_PATH = "tests/data/"


@pytest.fixture
def image():
    return Image.open(IMAGE_PATH + os.listdir(IMAGE_PATH)[0])


@pytest.fixture
def authority_rsa(image):
    return Authority(embed_og_hash=True, embed_method="DWT", sign_method="RSA")


@pytest.fixture
def authority_ecdsa(image):
    return Authority(embed_og_hash=True, embed_method="DWT", sign_method="ECDSA")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_sign_method_raises_value_error():
    with pytest.raises(ValueError, match="is not a valid sign method"):
        Authority(embed_og_hash=True, sign_method="INVALID")


def test_default_sign_method_is_rsa():
    authority = Authority(embed_og_hash=True)
    assert authority.sign_method == "RSA"


# ---------------------------------------------------------------------------
# Signature length
# ---------------------------------------------------------------------------


def test_rsa_sig_len_matches_key_size(authority_rsa):
    assert authority_rsa._sig_len == authority_rsa.public_key.key_size // 8


def test_ecdsa_sig_len_is_64(authority_ecdsa):
    assert authority_ecdsa._sig_len == 64


def test_ecdsa_signature_bytes_are_64(authority_ecdsa, image):
    h = authority_ecdsa.compute_perceptual_hash(image)
    sig = authority_ecdsa.sign_hash(h)
    assert len(sig) == 64


# ---------------------------------------------------------------------------
# Sign / verify correctness
# ---------------------------------------------------------------------------


def test_ecdsa_verify_same_hash(authority_ecdsa, image):
    h = authority_ecdsa.compute_perceptual_hash(image)
    sig = authority_ecdsa.sign_hash(h)
    assert authority_ecdsa.verify_signature(h, sig)


def test_ecdsa_verify_empty_signature_returns_false(authority_ecdsa, image):
    h = authority_ecdsa.compute_perceptual_hash(image)
    assert not authority_ecdsa.verify_signature(h, b"")


def test_ecdsa_verify_wrong_hash_returns_false(authority_ecdsa, image):
    h = authority_ecdsa.compute_perceptual_hash(image)
    sig = authority_ecdsa.sign_hash(h)
    wrong_h = 1 - h  # flip all bits
    assert not authority_ecdsa.verify_signature(wrong_h, sig)


def test_ecdsa_verify_from_different_authority_returns_false(image):
    auth1 = Authority(embed_og_hash=True, sign_method="ECDSA")
    auth2 = Authority(embed_og_hash=True, sign_method="ECDSA")
    h = auth1.compute_perceptual_hash(image)
    sig = auth1.sign_hash(h)
    assert not auth2.verify_signature(h, sig)


# ---------------------------------------------------------------------------
# Embedding roundtrip
# ---------------------------------------------------------------------------


def test_ecdsa_dwt_roundtrip_no_decode_error(authority_ecdsa):
    for name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + name)
        watermarked, _ = authority_ecdsa.embed_watermark(img)
        result = authority_ecdsa.extract_watermark(watermarked)
        assert "decode_error" not in result, f"{name}: {result.get('decode_error')}"


def test_ecdsa_dwt_roundtrip_signature_valid(authority_ecdsa):
    for name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + name)
        watermarked, _ = authority_ecdsa.embed_watermark(img)
        result = authority_ecdsa.extract_watermark(watermarked)
        assert authority_ecdsa.verify_signature(result["hash"], result["signature"]), (
            f"{name}: signature invalid after roundtrip"
        )


# ---------------------------------------------------------------------------
# User.verify
# ---------------------------------------------------------------------------


def test_user_verify_accepts_ecdsa_watermarked_image(authority_ecdsa):
    for name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + name)
        watermarked, _ = authority_ecdsa.embed_watermark(img)
        result = User(authority_ecdsa).verify(watermarked)
        assert result["signature_valid"], f"{name}: signature_valid is False"
        assert result["accepted"], f"{name}: accepted is False"


def test_user_verify_ecdsa_distance_within_threshold(authority_ecdsa):
    for name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + name)
        watermarked, _ = authority_ecdsa.embed_watermark(img)
        result = User(authority_ecdsa).verify(watermarked)
        assert result["distance"] < 10, (
            f"{name}: distance {result['distance']} exceeds threshold"
        )


def test_user_verify_rejects_ecdsa_signature_from_different_authority(image):
    auth1 = Authority(embed_og_hash=True, embed_method="DWT", sign_method="ECDSA")
    auth2 = Authority(embed_og_hash=True, embed_method="DWT", sign_method="ECDSA")
    watermarked, _ = auth1.embed_watermark(image)
    result = User(auth2).verify(watermarked)
    assert not result["signature_valid"]
    assert not result["accepted"]


# ---------------------------------------------------------------------------
# Payload size: ECDSA < RSA
# ---------------------------------------------------------------------------


def test_ecdsa_encoded_payload_smaller_than_rsa():
    rsa_auth = Authority(embed_og_hash=True, sign_method="RSA", key_size=2048)
    ec_auth = Authority(embed_og_hash=True, sign_method="ECDSA")
    assert ec_auth._expected_encoded_length() < rsa_auth._expected_encoded_length()


# ---------------------------------------------------------------------------
# PSNR: ECDSA >= RSA (fewer coefficients modified)
# ---------------------------------------------------------------------------


def test_ecdsa_psnr_not_worse_than_rsa():
    """
    ECDSA embeds fewer bits so it modifies fewer DWT coefficients.
    PSNR should be equal or better than RSA on the same image.
    """
    for name in os.listdir(IMAGE_PATH):
        img = Image.open(IMAGE_PATH + name)
        orig = np.array(img.convert("RGB"))

        rsa_auth = Authority(embed_og_hash=True, sign_method="RSA", key_size=2048)
        ec_auth = Authority(embed_og_hash=True, sign_method="ECDSA")

        wm_rsa, _ = rsa_auth.embed_watermark(img)
        wm_ec, _ = ec_auth.embed_watermark(img)

        psnr_rsa = peak_signal_noise_ratio(
            orig, np.array(wm_rsa.convert("RGB")), data_range=255
        )
        psnr_ec = peak_signal_noise_ratio(
            orig, np.array(wm_ec.convert("RGB")), data_range=255
        )

        assert psnr_ec >= psnr_rsa - 0.5, (
            f"{name}: ECDSA PSNR {psnr_ec:.2f} dB worse than RSA {psnr_rsa:.2f} dB"
        )


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


def test_settings_default_sign_method_is_rsa():
    s = Settings()
    assert s.sign_method == "RSA"


def test_settings_from_json_ecdsa():
    data = {"sign_method": "ECDSA", "embed_og_hash": True}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        s = Settings.from_json(path)
        assert s.sign_method == "ECDSA"
    finally:
        os.unlink(path)


def test_settings_from_json_rsa_default():
    data = {"embed_og_hash": True}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        s = Settings.from_json(path)
        assert s.sign_method == "RSA"
    finally:
        os.unlink(path)
