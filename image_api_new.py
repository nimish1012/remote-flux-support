from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageFilter
import asyncio
import io
import base64
import uuid
import logging
import time

# Import the proxy pipeline
from remote_flux_pipeline import RemoteFluxPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipe = None  # Global proxy pipeline reference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global pipe
    logger.info("Starting application")
    try:
        pipe = RemoteFluxPipeline(base_url="http://127.0.0.1:7001")
        logger.info("✅ [READY] RemoteFluxPipeline proxy initialised")
    except Exception as e:
        logger.error(f"Failed to initialise RemoteFluxPipeline proxy: {e}")
        raise RuntimeError(f"Pipeline initialisation failed: {e}")

    # Start exactly ONE background queue processor
    queue_task = asyncio.create_task(queue_processor())

    yield

    # Shutdown
    logger.info("Shutting down application")
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        pass

    if pipe is not None:
        try:
            pipe.close()
        except Exception:
            pass

    logger.info("Pipeline reference cleared")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sequential request queue — only one generation runs at a time
request_queue: asyncio.Queue = asyncio.Queue()
response_channels: dict[str, asyncio.Queue | WebSocket] = {}


class TextToImageRequest(BaseModel):
    prompt: str
    height: int = 768
    width: int = 768
    num_inference_steps: int = 20
    guidance_scale: float = 2.5


async def process_image(img: Image.Image, radius: int) -> Image.Image:
    """Apply Gaussian blur in a thread so the event loop stays free."""
    if radius == 0:
        return img
    return await asyncio.to_thread(lambda: img.filter(ImageFilter.GaussianBlur(radius)))


async def encode_image(img: Image.Image) -> str:
    """Encode PIL image to base64 JPEG in a thread."""
    def _encode():
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    return await asyncio.to_thread(_encode)


async def generate_and_stream(prompt: str, request_id: str,
                               height: int, width: int,
                               num_inference_steps: int, guidance_scale: float):
    """
    Call the remote flux server and yield SSE frames:
      - 4 progressively sharper previews (blur radius 10 → 5 → 2 → 0)
      - a final 'done' event
    """
    logger.info(f"[{request_id}] Generating {width}x{height} — '{prompt[:60]}'")

    try:
        start = time.time()
        result = await asyncio.to_thread(
            lambda: pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        )
        output_image = result.images[0]
        logger.info(f"[{request_id}] Remote generation took {time.time() - start:.2f}s")
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
    return {
        "status": "ok",
        "pipeline_type": "remote_proxy",
        "remote_url": "http://127.0.0.1:7001",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
