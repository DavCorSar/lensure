"""
Definition of the cli to execute the multiple functionalities of the package
"""

import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import typer

from lensure.roles.authority import Authority
from lensure.roles.attacker import Attacker
from lensure.roles.user import User

matplotlib.use("TkAgg")

app = typer.Typer()


@app.command()
def run_experiment(image_path: str, embed_og_hash: bool = True):
    """
    Performs an attack simulation over the specifyied image
    """

    img = Image.open(image_path)

    authority = Authority(embed_og_hash)
    attacker = Attacker(img, original_image_path=image_path)
    user = User(authority)

    watermarked_image = authority.include_watermarking(img)

    attacks = ["original", "jpeg", "resize", "blur", "noise", "change"]

    plt.figure(figsize=(15, 6))

    for i, attack_type in enumerate(attacks):
        modifyied_image = attacker.apply_attack(attack_type)

        result = user.verify(modifyied_image, watermarked_image)

        print(f"\n[{attack_type}]")
        print("distance:", result["distance"])
        print("signature valid:", result["signature_valid"])
        print("accepted:", result["accepted"])

        plt.subplot(1, len(attacks), i + 1)
        plt.imshow(modifyied_image, cmap="gray")
        plt.title(
            f"{attack_type}\nD={result['distance']}\nAccepted={result['accepted']}"
        )
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    app()
