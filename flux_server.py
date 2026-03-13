import os

# Set before any torch/cuda imports
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from datetime import datetime
import torch
import gc
import base64
import io
import logging
import asyncio
from PIL import Image
from diffusers import DiffusionPipeline

# Output directory — created at startup if it doesn't exist
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipe = None  # Global pipeline reference


def clear_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_pipeline():
    """Load the diffusion pipeline with memory-optimised settings for RTX A5000."""
    clear_gpu_memory()

    # Reserve at most 90 % of VRAM — leaves ~2.4 GB for driver/OS overhead
    torch.cuda.set_per_process_memory_fraction(0.90, device=0)

    logger.info(f"VRAM before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")

    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    # enable_sequential_cpu_offload offloads individual *layers* to CPU between
    # forward passes — far lower peak VRAM than enable_model_cpu_offload which
    # keeps whole submodels resident. Essential for FLUX on a 24 GB card.
    # NOTE: attention_slicing is not compatible with sequential offload.
    pipe.enable_sequential_cpu_offload()

    # Tile the VAE decode step so large images don't OOM at the final step
    pipe.vae.enable_tiling()

    logger.info(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global pipe
    logger.info("flux_server: Starting up")
    try:
        pipe = await asyncio.to_thread(load_pipeline)
        logger.info("flux_server: Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"flux_server: Failed to load pipeline: {e}")
        raise RuntimeError(f"Pipeline initialization failed: {e}")

    yield

    # Shutdown
    logger.info("flux_server: Shutting down")
    if pipe is not None:
        del pipe
        clear_gpu_memory()
    logger.info("flux_server: GPU memory cleared")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sequential lock — only one generation at a time on the GPU
_generation_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 20
    width: int = 768
    height: int = 768
    guidance_scale: Optional[float] = 2.5
    return_base64: bool = True


def _run_pipeline(prompt: str, steps: int, width: int, height: int,
                  guidance_scale: Optional[float]) -> Image.Image:
    """Run the pipeline synchronously (called via asyncio.to_thread)."""
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
        )
    image = result.images[0]
    # Free intermediate tensors immediately after generation
    clear_gpu_memory()
    return image


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate an image, save it to output/, and return it as base64-encoded PNG."""
    logger.info(f"Generating {req.width}x{req.height} — '{req.prompt[:60]}' steps={req.steps}")

    # Serialise GPU work — prevents concurrent requests from OOM-ing each other
    async with _generation_lock:
        try:
            image = await asyncio.to_thread(
                _run_pipeline,
                req.prompt, req.steps, req.width, req.height, req.guidance_scale,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM: {e}")
            clear_gpu_memory()
            return {"error": "GPU out of memory — try a smaller resolution or fewer steps"}
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"error": f"Generation failed: {str(e)}"}

    # Save to output/ with a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = OUTPUT_DIR / f"{timestamp}.png"
    await asyncio.to_thread(image.save, out_path)
    logger.info(f"Saved image to {out_path}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    logger.info("Generation complete, returning base64 image")
    return {"image_base64": img_b64, "saved_to": str(out_path)}


@app.get("/health")
async def health():
    vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "pipeline_loaded": pipe is not None,
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
    }


# uvicorn flux_server:app --host 0.0.0.0 --port 7001