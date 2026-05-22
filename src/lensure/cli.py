"""
Definition of the cli to execute the multiple functionalities of the package
"""

import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import typer
from tqdm import tqdm
import polars as pl

from lensure.utils import stable_diffusion_modifyier
from lensure.utils import pipelines
from lensure.utils import analysis
from lensure.utils.social_media import create_bluesky_client_pool
from lensure.settings import Settings

matplotlib.use("TkAgg")

app = typer.Typer()


@app.command()
def run_experiment(
    image_path: str,
    settings_path: Optional[str] = typer.Option(
        None, "--settings-path", help="Path to JSON settings file"
    ),
) -> tuple[dict, plt.Figure] | None:
    """
    Performs an attack simulation over the specified image
    """
    settings = Settings.from_json(settings_path) if settings_path else Settings()
    pipelines.run_single_image_execution(
        image_path=image_path,
        settings=settings,
    )


@app.command()
def complete_execution(
    images_path: str,
    output_path: str,
    settings_path: Optional[str] = typer.Option(
        None, "--settings-path", help="Path to JSON settings file"
    ),
):
    """
    Performs a complete execution and saves all the results
    to a posterior analysis.
    """
    settings = Settings.from_json(settings_path) if settings_path else Settings()

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
                settings=settings,
                save_results=True,
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
                        "psnr": results["psnr"],
                        "ssim": results["ssim"],
                    }
                )
        except Exception as e:
            print(f"Error {e} in image {image_name}")
            plt.close()

    df = pl.DataFrame(rows)
    csv_path = os.path.join(output_path, "results.csv")
    df.write_csv(csv_path)

    analysis.analyze_results(csv_path, settings.embed_og_hash, output_path)
    if settings.embed_og_hash:
        analysis.compute_roc(csv_path, output_path)


if __name__ == "__main__":
    app()
