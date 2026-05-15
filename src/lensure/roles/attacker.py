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

from lensure.utils import pipelines
from lensure.utils import stable_diffusion_modifyier
from lensure.utils import social_media
from lensure.utils.social_media import BlueskyClientPool

SEED = 42


class Attacker:
    """
    Implements multiple types of attacks
    """

    def __init__(
        self,
        image: Image,
        original_image_path: str,
        pipe: StableDiffusionInpaintPipeline | None = None,
        bluesky_client_pool: BlueskyClientPool | None = None,
    ):
        self.image = image
        self.original_image_path = original_image_path
        if pipe is None:
            pipe = stable_diffusion_modifyier.create_dnn_pipeline()
        self.pipe = pipe
        self.bluesky_client_pool = bluesky_client_pool

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
            result = self.__resize_image(1.1)

        if attack_type == "blur":
            result = self.image.filter(ImageFilter.GaussianBlur(radius=1))

        if attack_type == "noise":
            result = self.__add_noise(0, 2)

        if attack_type == "semantic-transformation-soft":
            result = self.__apply_semantic_transformation(
                coverage=0.15,
                prompt=(
                    "a realistic human face looking naturally at the camera, "
                    "same lighting, same perspective, photorealistic"
                ),
                negative_prompt=(
                    "cartoon, anime, distorted face, extra features, deformed, "
                    "low quality, blurry, unrealistic, artifacts"
                ),
            )

        if attack_type == "semantic-transformation-hard":
            result = self.__apply_semantic_transformation(
                coverage=0.25,
                prompt=(
                    "a realistic full body person standing naturally in the scene, "
                    "same lighting, same perspective, photorealistic"
                ),
                negative_prompt=(
                    "cartoon, anime, distorted body, extra limbs, missing limbs, "
                    "deformed face, low quality, blurry, unrealistic, artifacts"
                ),
            )

        if attack_type == "change":
            result = self.__select_different_image()

        if attack_type == "social-bluesky":
            result = social_media.bluesky_attack(self.image, client_pool=self.bluesky_client_pool)

        if attack_type == "social-telegram":
            result = social_media.telegram_attack(self.image)

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

    def __resize_image(self, factor: float) -> Image:
        """
        Resizes the image dividing the width and height by the specifyied factor
        """
        return self.image.resize(
            (int(self.image.width * factor), int(self.image.height * factor))
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

    def __apply_semantic_transformation(
        self, coverage: float, prompt: str, negative_prompt: str
    ) -> Image:
        """
        Applyies a malicious semantic transformation to the image.
        """
        image = self.image.convert("RGB")

        mask = self.__generate_person_insertion_mask(image=image, coverage=coverage)

        bbox = mask.getbbox()
        if bbox is None:
            return image

        x0, y0, x1, y1 = bbox
        crop = image.crop(bbox)
        mask_crop = mask.crop(bbox)

        sd_size = (512, 512)
        crop_sd = crop.resize(sd_size, Image.LANCZOS)
        mask_sd = mask_crop.resize(sd_size, Image.NEAREST)

        generator_device = stable_diffusion_modifyier.get_device()
        generator = torch.Generator(device=generator_device).manual_seed(SEED)

        result_sd = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=crop_sd,
            mask_image=mask_sd,
            num_inference_steps=10,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        result_crop = result_sd.resize(crop.size, Image.LANCZOS)

        result = image.copy()
        result.paste(result_crop, (x0, y0), mask_crop)

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
        return pipelines.load_image_from_path(choice)

    def __generate_person_insertion_mask(
        self, image: Image.Image, coverage: float
    ) -> Image.Image:
        """
        Generates an automatic mask where a person-like object can be inserted.

        The mask is placed in the lower half of the image and has a vertical
        elliptical shape, approximating the area occupied by a standing person.

        Parameters
        ----------
        image : Image.Image
            Input image.
        coverage : float
            Fraction of the total image area covered by the mask ellipse
            bounding box (e.g. 0.10 = 10 %).

        Returns
        -------
        Image.Image
            Binary mask. White pixels indicate the area to modify.
        """

        rng = np.random.default_rng(SEED)

        width, height = image.size

        # mask_width * mask_height = coverage * width * height (aspect ratio preserved)
        mask_width = int(width * coverage**0.5)
        mask_height = int(height * coverage**0.5)

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
