import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from lensure.roles.authority import Authority
from lensure.roles.attacker import Attacker
from lensure.roles.user import User


def run_single_image_execution(
    image_path: str,
    embed_og_hash: bool = True,
    save_results: bool = False,
    embed_method: str = "DWT",
    pipe: StableDiffusionInpaintPipeline | None = None,
) -> tuple[dict, plt.Figure] | None:
    """
    Performs an attack simulation over the specifyied image
    """

    img = Image.open(image_path)

    authority = Authority(embed_og_hash, embed_method=embed_method)

    watermarked_image = authority.embed_watermark(img)
    attacker = Attacker(watermarked_image, original_image_path=image_path, pipe=pipe)
    user = User(authority)

    attacks = [
        "original",
        "jpeg",
        "resize",
        "blur",
        "noise",
        "semantic-transformation-soft",
        "semantic-transformation-hard",
        "change",
    ]

    fig = plt.figure(figsize=(15, 6))

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

        ax = fig.add_subplot(round(len(attacks) / 4), 4, i + 1)
        ax.imshow(modifyied_image, cmap="gray")
        ax.set_title(
            f"{attack_type}\nD={result['distance']}\nAccepted={result['accepted']}"
        )
        ax.axis("off")

        metrics[attack_type] = result

    if save_results:
        return metrics, fig
    plt.show()
    return None
