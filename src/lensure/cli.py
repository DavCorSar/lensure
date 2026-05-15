"""
Definition of the cli to execute the multiple functionalities of the package
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import typer
from tqdm import tqdm
import polars as pl

from lensure.utils import stable_diffusion_modifyier
from lensure.utils import pipelines
from lensure.utils.social_media import create_bluesky_client_pool

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

    pipelines.run_single_image_execution(
        image_path=image_path,
        embed_og_hash=embed_og_hash,
        save_results=save_results,
        embed_method=embed_method,
    )


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
    plots_path = "/plots/"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + plots_path, exist_ok=True)
    pipe = stable_diffusion_modifyier.create_dnn_pipeline()
    bluesky_client_pool = create_bluesky_client_pool()
    rows = []

    for image_name in tqdm(os.listdir(images_path)):
        try:
            metrics, fig = pipelines.run_single_image_execution(
                images_path + image_name,
                embed_og_hash,
                save_results=True,
                embed_method=embed_method,
                pipe=pipe,
                bluesky_client_pool=bluesky_client_pool,
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
        except Exception as e:
            print(f"Error {e} in image {image_name}")
            plt.close()

    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(output_path, "results.csv"))


if __name__ == "__main__":
    app()
