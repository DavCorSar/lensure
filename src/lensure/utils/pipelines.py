import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from lensure.roles.authority import Authority
from lensure.roles.attacker import Attacker
from lensure.roles.user import User
from lensure.settings import Settings


def compute_image_quality(original: Image.Image, watermarked: Image.Image) -> dict:
    orig = np.array(original.convert("RGB"))
    wm = np.array(watermarked.convert("RGB"))
    psnr = peak_signal_noise_ratio(orig, wm, data_range=255)
    ssim = structural_similarity(orig, wm, channel_axis=2, data_range=255)
    return {"psnr": round(psnr, 4), "ssim": round(ssim, 6)}


def load_image_from_path(image_path: str, max_side: int = 1280) -> Image.Image:
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def run_single_image_execution(
    image_path: str,
    settings: Settings | None = None,
    save_results: bool = False,
    pipe: StableDiffusionInpaintPipeline | None = None,
    bluesky_client_pool=None,
) -> tuple[dict, plt.Figure] | None:
    """
    Performs an attack simulation over the specified image
    """
    if settings is None:
        settings = Settings()

    img = load_image_from_path(image_path)

    authority = Authority(
        settings.embed_og_hash,
        embed_method=settings.embed_method,
        hash_size=settings.hash_size,
        hash_type=settings.hash_type,
        delta_dwt=settings.delta_dwt,
        allow_retries=settings.allow_retries,
        key_size=settings.key_size,
        sign_method=settings.sign_method,
        dwt_size=settings.dwt_size,
    )

    watermarked_image, delta_used = authority.embed_watermark(img)
    quality = compute_image_quality(img, watermarked_image)
    quality["delta_used"] = delta_used
    attacker = Attacker(
        watermarked_image,
        original_image_path=image_path,
        pipe=pipe,
        bluesky_client_pool=bluesky_client_pool,
    )
    user = User(authority)

    attacks = settings.attacks

    ncols = 4
    nrows = math.ceil(len(attacks) / ncols)
    fig = plt.figure(figsize=(16, 3.6 * nrows), constrained_layout=True)

    metrics = {}
    for i, attack_type in enumerate(attacks):
        modifyied_image = attacker.apply_attack(attack_type)

        result = user.verify(modifyied_image)

        if not save_results:
            print(f"\n[{attack_type}]")
            print("distance:", result["distance"])
            print("signature valid:", result["signature_valid"])
            print("accepted:", result["accepted"])

            if "decode_error" in result:
                print("decode error:", result["decode_error"])

        ax = fig.add_subplot(math.ceil(len(attacks) / 4), 4, i + 1)
        ax.imshow(modifyied_image, cmap="gray")
        accepted_text = "accepted" if result["accepted"] else "rejected"
        ax.set_title(attack_type, fontsize=18, pad=6)
        ax.set_xlabel(
            f"D={result['distance']} | {accepted_text}",
            fontsize=18,
            labelpad=6,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        metrics[attack_type] = {**result, **quality}

    if save_results:
        return metrics, fig
    plt.show()
    return None
