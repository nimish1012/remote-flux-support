import os

# Set before any torch/cuda imports
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageFilter
import torch
import io
import base64
from diffusers import DiffusionPipeline, FluxPipeline
import asyncio
import uuid
import logging
import time
import gc

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
    """Load the diffusion pipeline with memory-optimised settings."""
    clear_gpu_memory()
    logger.info(f"VRAM before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")

    # pipe = DiffusionPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev",
    #     torch_dtype=torch.bfloat16,
    #     use_safetensors=True,
    # )


    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    # DO NOT call pipe.to("cuda") — cpu_offload handles device placement
    # enable_model_cpu_offload moves submodels to GPU only when needed,
    # keeping overall VRAM usage much lower than a full .to("cuda") load.
    pipe.enable_model_cpu_offload()

    # Slice attention computation to reduce peak VRAM usage
    pipe.enable_attention_slicing(slice_size="auto")

    # Tile the VAE decode step so large images don't OOM at the final step
    pipe.enable_vae_tiling()

    logger.info(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global pipe
    logger.info("Starting application")
    try:
        logger.info("Initializing DiffusionPipeline")
        pipe = await asyncio.to_thread(load_pipeline)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise RuntimeError(f"Pipeline initialization failed: {str(e)}")

    logger.info("Starting queue processor")
    asyncio.create_task(queue_processor())
    yield

    # Shutdown
    logger.info("Shutting down application")
    if pipe is not None:
        del pipe
        clear_gpu_memory()
    logger.info("Pipeline and CUDA cache cleared")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sequential request queue — only one generation runs at a time,
# preventing multiple concurrent jobs from fighting over VRAM.
request_queue = asyncio.Queue()
response_channels: dict[str, asyncio.Queue | WebSocket] = {}


class TextToImageRequest(BaseModel):
    prompt: str
    height: int = 768
    width: int = 768
    num_inference_steps: int = 20
    guidance_scale: float = 2.5


def _run_pipeline(prompt: str, height: int, width: int,
                  num_inference_steps: int, guidance_scale: float) -> Image.Image:
    """Run the pipeline synchronously (called via asyncio.to_thread)."""
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    image = result.images[0]
    # Free intermediate tensors immediately after generation
    clear_gpu_memory()
    return image


async def process_image(img: Image.Image, radius: int) -> Image.Image:
    """Apply Gaussian blur in a thread so the event loop stays free."""
    if radius == 0:
        return img
    return await asyncio.to_thread(lambda: img.filter(ImageFilter.GaussianBlur(radius)))


async def encode_image(img: Image.Image) -> str:
    """Encode PIL image to base64 JPEG in a thread."""
    def _encode():
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()

    return await asyncio.to_thread(_encode)


async def generate_and_stream(prompt: str, request_id: str,
                               height: int, width: int,
                               num_inference_steps: int, guidance_scale: float):
    """
    Generate one image and yield SSE frames:
      - 4 progressively sharper previews (blur radius 10 → 5 → 2 → 0)
      - a final 'done' event
    """
    logger.info(f"[{request_id}] Generating {width}x{height} image — '{prompt[:60]}'")    

    try:
        start = time.time()
        output_image = await asyncio.to_thread(
            _run_pipeline, prompt, height, width, num_inference_steps, guidance_scale
        )
        logger.info(f"[{request_id}] Generation took {time.time() - start:.2f}s")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"[{request_id}] OOM: {e}")
        clear_gpu_memory()
        yield f"event: error\ndata: GPU out of memory — try a smaller resolution\n\n"
        return
    except Exception as e:
        logger.error(f"[{request_id}] Generation failed: {e}")
        yield f"event: error\ndata: Image generation failed: {str(e)}\n\n"
        return

    for radius in [10, 5, 2, 0]:
        img = await process_image(output_image, radius)
        b64 = await encode_image(img)
        yield f"event: image\ndata: {b64}\n\n"
        logger.debug(f"[{request_id}] Sent frame blur={radius}")
        if radius != 0:
            await asyncio.sleep(0.4)

    yield "event: done\ndata: Image stream complete\n\n"
    logger.info(f"[{request_id}] Stream complete")


async def _send_to_channel(request_id: str, event: str):
    """Route an SSE event to either an asyncio.Queue or a WebSocket."""
    channel = response_channels.get(request_id)
    if channel is None:
        return
    if isinstance(channel, asyncio.Queue):
        await channel.put(event)
    elif isinstance(channel, WebSocket):
        await channel.send_text(event)


async def _close_channel(request_id: str):
    """Close and remove a response channel."""
    channel = response_channels.pop(request_id, None)
    if isinstance(channel, WebSocket):
        try:
            await channel.close()
        except Exception:
            pass
    logger.info(f"[{request_id}] Channel closed")


async def queue_processor():
    """Background loop — processes one request at a time."""
    while True:
        request_id, prompt, height, width, steps, guidance = await request_queue.get()
        logger.info(f"[{request_id}] Dequeued")
        try:
            async for event in generate_and_stream(
                prompt, request_id, height, width, steps, guidance
            ):
                await _send_to_channel(request_id, event)
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {e}")
            await _send_to_channel(
                request_id,
                f"event: error\ndata: Processing failed: {str(e)}\n\n"
            )
        finally:
            await _close_channel(request_id)
            request_queue.task_done()
            logger.info(f"[{request_id}] Done")


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@app.post("/text-to-image-stream")
async def text_to_image_stream(request: TextToImageRequest):
    """Enqueue a generation request and stream SSE frames back to the client."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] HTTP POST — '{request.prompt[:60]}'")

    response_queue: asyncio.Queue[str] = asyncio.Queue()
    response_channels[request_id] = response_queue

    await request_queue.put((
        request_id, request.prompt,
        request.height, request.width,
        request.num_inference_steps, request.guidance_scale,
    ))

    async def stream_response():
        try:
            while True:
                event = await response_queue.get()
                yield event
                if "event: done" in event or "event: error" in event:
                    break
                response_queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"[{request_id}] Client disconnected")
            response_channels.pop(request_id, None)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/text-to-image")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint — receive prompt as text, stream SSE-style frames."""
    await websocket.accept()
    request_id = str(uuid.uuid4())
    response_channels[request_id] = websocket
    logger.info(f"[{request_id}] WebSocket connected")
    try:
        prompt = await websocket.receive_text()
        logger.info(f"[{request_id}] WS prompt: '{prompt[:60]}'")
        # Use default generation params for WebSocket clients
        await request_queue.put((request_id, prompt, 768, 768, 20, 2.5))
        while request_id in response_channels:
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"[{request_id}] WebSocket error: {e}")
        if request_id in response_channels:
            await websocket.send_text(f"event: error\ndata: WebSocket error: {str(e)}\n\n")
            response_channels.pop(request_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)