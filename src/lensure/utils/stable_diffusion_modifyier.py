from diffusers import StableDiffusionInpaintPipeline
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_dnn_pipeline() -> StableDiffusionInpaintPipeline:
    """
    TODO
    """
    device = get_device()

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
    return pipe
