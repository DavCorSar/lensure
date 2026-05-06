import huggingface_hub
import diffusers
import transformers
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import torch

diffusers.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()
diffusers.utils.logging.disable_progress_bar()
transformers.utils.logging.disable_progress_bar()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_dnn_pipeline() -> StableDiffusionInpaintPipeline:
    device = get_device()

    model_id = "runwayml/stable-diffusion-inpainting"

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
        use_safetensors=False,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()

    pipe.set_progress_bar_config(disable=True)
    return pipe
