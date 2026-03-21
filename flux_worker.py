"""
flux_worker.py
--------------
Standalone worker process.  Pull jobs from the Redis queue, generate images
via RemoteFluxPipeline (which calls flux_server.py), then publish each SSE
frame back on the job's private Pub/Sub channel.

Run one or more of these independently:
    python flux_worker.py

They share the same Redis job queue so work is distributed automatically.
"""

import base64
import gc
import io
import json
import logging
import os
import time
from PIL import Image, ImageFilter

from remote_flux_pipeline import RemoteFluxPipeline
from redis_queue import (
    JOB_QUEUE,
    RESULT_TTL,
    cache_key,
    get_sync_redis,
    job_channel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux_worker")

# ---------------------------------------------------------------------------
# Pipeline — swap RemoteFluxPipeline for the local DiffusionPipeline here
# if you want the worker to drive the GPU directly instead of via HTTP.
# ---------------------------------------------------------------------------
FLUX_SERVER_URL = os.getenv("FLUX_SERVER_URL", "http://127.0.0.1:7001")
pipe = RemoteFluxPipeline(base_url=FLUX_SERVER_URL)
logger.info(f"Worker using remote pipeline at {FLUX_SERVER_URL}")

r = get_sync_redis()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def publish(job_id: str, event: str, data: str) -> None:
    """Publish one SSE-style frame to the job's private channel."""
    r.publish(job_channel(job_id), json.dumps({"event": event, "data": data}))


def apply_blur(img: Image.Image, radius: int) -> Image.Image:
    if radius == 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius))


def encode_jpeg_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------------

def process_job(job: dict) -> None:
    job_id   = job["job_id"]
    prompt   = job["prompt"]
    height   = job.get("height", 768)
    width    = job.get("width", 768)
    steps    = job.get("num_inference_steps", 20)
    guidance = job.get("guidance_scale", 2.5)

    ck = cache_key(prompt, width, height, steps, guidance)

    # ---- cache hit -------------------------------------------------------
    cached = r.get(ck)
    if cached:
        logger.info(f"[{job_id}] Cache hit — serving cached result")
        # Replay the same 4-frame blur sequence from the cached final PNG
        cached_img = Image.open(io.BytesIO(base64.b64decode(cached))).convert("RGB")
        for radius in [10, 5, 2, 0]:
            frame = apply_blur(cached_img, radius)
            publish(job_id, "image", encode_jpeg_b64(frame))
        publish(job_id, "done", "Image stream complete")
        return

    # ---- generate --------------------------------------------------------
    logger.info(f"[{job_id}] Generating {width}x{height} — '{prompt[:60]}'")
    try:
        start = time.time()
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        output_image: Image.Image = result.images[0]
        logger.info(f"[{job_id}] Generation took {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"[{job_id}] Generation failed: {e}")
        publish(job_id, "error", f"Image generation failed: {e}")
        return

    # ---- cache final PNG (lossless so re-blurs stay sharp) ---------------
    png_buf = io.BytesIO()
    output_image.save(png_buf, format="PNG")
    r.setex(ck, RESULT_TTL, base64.b64encode(png_buf.getvalue()).decode())

    # ---- stream 4 progressive blur frames --------------------------------
    for radius in [10, 5, 2, 0]:
        frame = apply_blur(output_image, radius)
        publish(job_id, "image", encode_jpeg_b64(frame))
        logger.debug(f"[{job_id}] Published frame blur={radius}")

    publish(job_id, "done", "Image stream complete")
    logger.info(f"[{job_id}] Stream complete")

    # Free memory
    del output_image
    gc.collect()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Worker started — listening on queue '{JOB_QUEUE}'")
    while True:
        # BRPOP blocks up to 5 s then loops, so Ctrl-C is still responsive
        item = r.brpop(JOB_QUEUE, timeout=5)
        if item is None:
            continue
        _, payload = item
        try:
            job = json.loads(payload)
        except json.JSONDecodeError as e:
            logger.error(f"Bad payload: {e} — {payload!r}")
            continue

        try:
            process_job(job)
        except Exception as e:
            logger.error(f"Unhandled error in process_job: {e}")
            try:
                publish(job.get("job_id", "unknown"), "error",
                        f"Worker crashed: {e}")
            except Exception:
                pass
