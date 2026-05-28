"""
Tests for watermark visual quality metrics (PSNR / SSIM).
"""

import math
import os

import numpy as np
import polars as pl
import pytest
from PIL import Image

from lensure.utils.pipelines import compute_image_quality


IMAGE_PATH = "tests/data/"


@pytest.fixture
def image():
    return Image.open(IMAGE_PATH + os.listdir(IMAGE_PATH)[0]).convert("RGB")


def test_compute_image_quality_returns_expected_keys(image):
    result = compute_image_quality(image, image)
    assert "psnr" in result
    assert "ssim" in result


def test_identical_images_have_perfect_ssim(image):
    result = compute_image_quality(image, image)
    assert result["ssim"] == pytest.approx(1.0, abs=1e-4)


def test_identical_images_have_infinite_psnr(image):
    result = compute_image_quality(image, image)
    assert math.isinf(result["psnr"])


def test_modified_image_has_lower_psnr_than_original(image):
    arr = np.array(image).astype(np.int16)
    arr = np.clip(arr + np.random.randint(-30, 30, arr.shape), 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)

    result = compute_image_quality(image, noisy)
    assert result["psnr"] < 40.0


def test_ssim_is_between_zero_and_one(image):
    arr = np.array(image).astype(np.int16)
    arr = np.clip(arr + np.random.randint(-30, 30, arr.shape), 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)

    result = compute_image_quality(image, noisy)
    assert 0.0 <= result["ssim"] <= 1.0


def test_watermarked_image_psnr_is_finite_and_positive(image):
    from lensure.roles.authority import Authority

    authority = Authority(embed_og_hash=True, embed_method="DWT")
    watermarked, _ = authority.embed_watermark(image)
    result = compute_image_quality(image, watermarked)
    assert math.isfinite(result["psnr"])
    assert result["psnr"] > 0.0


def test_watermarked_image_ssim_is_less_than_original(image):
    from lensure.roles.authority import Authority

    authority = Authority(embed_og_hash=True, embed_method="DWT")
    watermarked, _ = authority.embed_watermark(image)
    result = compute_image_quality(image, watermarked)
    assert 0.0 <= result["ssim"] < 1.0


def test_analyze_results_reports_psnr_ssim(tmp_path):
    from lensure.utils.analysis import analyze_results

    csv_path = str(tmp_path / "results.csv")
    df = pl.DataFrame({
        "image": ["a.jpg", "a.jpg", "b.jpg", "b.jpg"],
        "attack": ["original", "jpeg", "original", "jpeg"],
        "distance": [1, 2, 1, 3],
        "signature_valid": [True, True, True, False],
        "accepted": [True, True, True, False],
        "psnr": [42.0, 42.0, 38.5, 38.5],
        "ssim": [0.995, 0.995, 0.991, 0.991],
    })
    df.write_csv(csv_path)

    analyze_results(csv_path, embed_og_hash=True, output_dir=str(tmp_path))

    analysis_text = (tmp_path / "analysis.txt").read_text()
    assert "PSNR" in analysis_text
    assert "SSIM" in analysis_text
