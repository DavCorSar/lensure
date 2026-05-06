"""
Tests for the Attacker role. The SD pipeline is mocked to avoid
loading the model during unit tests.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

from lensure.roles.attacker import Attacker


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))


@pytest.fixture
def mock_pipe(sample_image):
    pipe = MagicMock()
    pipe.return_value.images = [sample_image.copy()]
    return pipe


@pytest.fixture
def image_dir(tmp_path, sample_image):
    """Directory with two images, required by the 'change' attack."""
    rng = np.random.default_rng(1)
    sample_image.save(tmp_path / "img1.png")
    Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(
        tmp_path / "img2.png"
    )
    return tmp_path


@pytest.fixture
def attacker(image_dir, sample_image, mock_pipe):
    return Attacker(
        sample_image,
        original_image_path=str(image_dir / "img1.png"),
        pipe=mock_pipe,
    )


@pytest.mark.parametrize("attack_type", ["original", "jpeg", "resize", "blur", "noise"])
def test_basic_attack_returns_pil_image(attacker, attack_type):
    result = attacker.apply_attack(attack_type)
    assert isinstance(result, Image.Image)


def test_resize_preserves_original_dimensions(attacker, sample_image):
    result = attacker.apply_attack("resize")
    assert result.size == sample_image.size


def test_noise_modifies_pixel_values(attacker, sample_image):
    result = attacker.apply_attack("noise")
    assert not np.array_equal(np.array(result), np.array(sample_image))


def test_change_attack_returns_pil_image(attacker):
    result = attacker.apply_attack("change")
    assert isinstance(result, Image.Image)


def test_invalid_attack_type_raises_value_error(attacker):
    with pytest.raises(ValueError, match="Invalid attack type"):
        attacker.apply_attack("nonexistent")


@pytest.mark.parametrize(
    "attack_type", ["semantic-transformation-soft", "semantic-transformation-hard"]
)
def test_semantic_transformation_calls_pipe_and_returns_original_size(
    attacker, sample_image, mock_pipe, attack_type
):
    result = attacker.apply_attack(attack_type)
    assert mock_pipe.called
    assert isinstance(result, Image.Image)
    assert result.size == sample_image.size


if __name__ == "__main__":
    pytest.main()
