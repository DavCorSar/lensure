"""
Definition of the attacker of the system. Implements all
the possible perturbations that will be use to test the robustness
of the system.
"""

import io
import os

from PIL import Image, ImageFilter, Image
import numpy as np


class Attacker:
    """
    Implements multiple types of attacks
    """

    def __init__(self, image: Image, original_image_path: str):
        self.image = image
        self.original_image_path = original_image_path

    def apply_attack(self, attack_type) -> Image:
        """
        Applies the specifyied attack to the image
        """
        if attack_type == "original":
            return self.image
        if attack_type == "jpeg":
            return self.__convert_to_jpg()

        if attack_type == "resize":
            return self.__downsize_image(4)

        if attack_type == "blur":
            return self.image.filter(ImageFilter.GaussianBlur(radius=20))

        if attack_type == "noise":
            return self.__add_noise(0, 100)

        if attack_type == "change":
            return self.__select_different_image()

        raise ValueError(f"Invalid attack type: {attack_type}")

    def __convert_to_jpg(self) -> Image:
        """
        Returns the same image after being saved in JPEG,
        which applies a lossy compression
        """
        buffer = io.BytesIO()
        self.image.save(buffer, format="JPEG", quality=1)
        buffer.seek(0)
        return Image.open(buffer)

    def __downsize_image(self, factor: int) -> Image:
        """
        Resizes the image dividing the width and height by the specifyied factor
        """
        return self.image.resize(
            (self.image.width // factor, self.image.height // factor)
        ).resize(self.image.size)

    def __add_noise(self, noise_mean: int, noise_std: int) -> Image:
        """
        Includes gaussian noise in the image
        """
        arr = np.array(self.image)
        noise = np.random.normal(noise_mean, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __select_different_image(self) -> Image:
        """
        Returns a different image from the same folder
        """
        parent_folder = "/".join(self.original_image_path.split("/")[:-1])
        choice = parent_folder + "/" + np.random.choice(os.listdir(parent_folder))
        while choice == self.original_image_path:
            choice = parent_folder + np.random.choice(os.listdir(parent_folder))
        return Image.open(choice)
