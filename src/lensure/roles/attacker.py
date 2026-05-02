"""
Definition of the attacker of the system. Implements all
the possible perturbations that will be use to test the robustness
of the system.
"""

import io
import os

from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch

SEED = 42


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
        result = None
        if attack_type == "original":
            result = self.image

        if attack_type == "jpeg":
            result = self.__convert_to_jpg()

        if attack_type == "resize":
            result = self.__downsize_image(2)

        if attack_type == "blur":
            result = self.image.filter(ImageFilter.GaussianBlur(radius=5))

        if attack_type == "noise":
            result = self.__add_noise(0, 10)

        if attack_type == "semantic-transformation-soft":
            result = self.__apply_semantic_transformation(level=2)

        if attack_type == "semantic-transformation-hard":
            result = self.__apply_semantic_transformation(level=1)

        if attack_type == "change":
            result = self.__select_different_image()

        if result is None:
            raise ValueError(f"Invalid attack type: {attack_type}")

        return result

    def __convert_to_jpg(self) -> Image:
        """
        Returns the same image after being saved in JPEG,
        which applies a lossy compression
        """
        buffer = io.BytesIO()
        self.image.save(buffer, format="JPEG", quality=90)
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
        rng = np.random.default_rng(SEED)
        arr = np.array(self.image)
        noise = rng.normal(noise_mean, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __apply_semantic_transformation(self, level: int) -> Image:
        """
        Applyies a malicious semantic transformation to the image.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "runwayml/stable-diffusion-inpainting"

        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )

        if device == "cuda":
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        else:
            pipe = pipe.to(device)

        original_size = self.image.size

        working_size = (512, 512)

        image = self.image.convert("RGB").resize(working_size, Image.LANCZOS)

        mask = self.__generate_person_insertion_mask(image=image, level=level)

        prompt = (
            "a realistic full body person standing naturally in the scene, "
            "same lighting, same perspective, photorealistic"
        )

        negative_prompt = (
            "cartoon, anime, distorted body, extra limbs, missing limbs, "
            "deformed face, low quality, blurry, unrealistic, artifacts"
        )

        generator_device = device if device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(SEED)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        result = result.resize(original_size, Image.LANCZOS)

        return result

    def __select_different_image(self) -> Image:
        """
        Returns a different image from the same folder
        """
        rng = np.random.default_rng(SEED)
        parent_folder = "/".join(self.original_image_path.split("/")[:-1])
        choice = parent_folder + "/" + rng.choice(os.listdir(parent_folder))
        while choice == self.original_image_path:
            choice = parent_folder + "/" + rng.choice(os.listdir(parent_folder))
        return Image.open(choice)

    def __generate_person_insertion_mask(
        self, image: Image.Image, level: int
    ) -> Image.Image:
        """
        Generates an automatic mask where a person-like object can be inserted.

        The mask is placed in the lower half of the image and has a vertical
        elliptical shape, approximating the area occupied by a standing person.

        Parameters
        ----------
        image : Image.Image
            Input image.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        Image.Image
            Binary mask. White pixels indicate the area to modify.
        """

        rng = np.random.default_rng(SEED)

        width, height = image.size

        mask_width = int(width / level)
        mask_height = int(height / level)

        x_min = int(width * 0.15)
        x_max = int(width * 0.85) - mask_width

        y_min = int(height * 0.45)
        y_max = int(height * 0.85) - mask_height

        if x_max <= x_min:
            x0 = max(0, (width - mask_width) // 2)
        else:
            x0 = int(rng.integers(x_min, x_max))

        if y_max <= y_min:
            y0 = max(0, height - mask_height - int(height * 0.05))
        else:
            y0 = int(rng.integers(y_min, y_max))

        x1 = x0 + mask_width
        y1 = y0 + mask_height

        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        draw.ellipse([x0, y0, x1, y1], fill=255)

        return mask
