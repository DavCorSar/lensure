"""
Definition of the cli to execute the multiple functionalities of the package
"""

import os

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import typer
from tqdm import tqdm
import polars as pl

from lensure.roles.authority import Authority
from lensure.roles.attacker import Attacker
from lensure.roles.user import User

matplotlib.use("TkAgg")

app = typer.Typer()


@app.command()
def run_experiment(
    image_path: str,
    embed_og_hash: bool = True,
    save_results: bool = False,
    embed_method: str = "DWT",
) -> tuple[dict, plt.Figure] | None:
    """
    Performs an attack simulation over the specifyied image
    """

    img = Image.open(image_path)

    authority = Authority(embed_og_hash, embed_method=embed_method)

    watermarked_image = authority.embed_watermark(img)
    attacker = Attacker(watermarked_image, original_image_path=image_path)
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


@app.command()
def complete_execution(
    images_path: str,
    output_path: str,
    embed_og_hash: bool = True,
    embed_method: str = "DWT",
):
    """
    Performs a complete execution and saves all the results
    to a posterior analysis.
    """
    plots_path = "plots/"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + plots_path, exist_ok=True)
    rows = []

    for image_name in tqdm(os.listdir(images_path)):
        metrics, fig = run_experiment(
            images_path + image_name,
            embed_og_hash,
            save_results=True,
            embed_method=embed_method,
        )
        fig.savefig(f"{output_path}/{plots_path}/{image_name}")
        plt.close()

        for attack, results in metrics.items():
            rows.append(
                {
                    "image": image_name,
                    "attack": attack,
                    "distance": results["distance"],
                    "signature_valid": results["signature_valid"],
                    "accepted": results["accepted"],
                }
            )

    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(output_path, "results.csv"))


if __name__ == "__main__":
    app()
