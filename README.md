# Image Generator - FLUX Model API

A FastAPI-based image generation service using the FLUX diffusion model with support for both local and remote model execution.

## Conda Environment Setup

This project uses a conda environment named `krea_env`. To set it up:

```bash
# Activate the environment
conda activate krea_env

# Verify PyTorch is installed
python -c "import torch; print('PyTorch:', torch.__version__)"
```

**Installed packages include:**
- pytorch 2.6.0 (CPU)
- fastapi 0.135.1
- uvicorn 0.41.0
- pillow 12.1.1
- httpx 0.28.1
- pydantic 2.12.5
- diffusers 0.35.2

> **Note:** The current environment uses CPU-based PyTorch. For GPU support with CUDA, install the CUDA version:
> ```bash
> conda install -n krea_env pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
> ```

## Overview

This project provides a text-to-image generation API using the FLUX model from Black Forest Labs. It supports two deployment modes:

1. **Local Mode**: Loads the model directly on your GPU
2. **Remote Mode**: Uses a remote model server to reduce local resource usage

## Architecture

### File Structure

```
image_generator/
├── flux_server.py           # Remote FLUX model server
├── image_api.py             # Local deployment API (loads model on GPU)
├── image_api_new.py         # Remote deployment API (uses proxy)
└── remote_flux_pipeline.py  # HTTP client that mimics DiffusionPipeline
```

### How the Code Works

#### 1. [`flux_server.py`](flux_server.py) - Remote Model Server

This is a lightweight FastAPI server that hosts the FLUX model:

- Loads the `FLUX.1-schnell` model on CUDA at startup
- Exposes a single POST endpoint `/generate`
- Accepts parameters: `prompt`, `steps`, `width`, `height`
- Returns base64-encoded PNG images

**Run on port 7001:**
```bash
uvicorn flux_server:app --host 0.0.0.0 --port 7001
```

#### 2. [`image_api.py`](image_api.py) - Local API

Direct local deployment that loads the model on your GPU:

- Uses `black-forest-labs/FLUX.1-Krea-dev` model
- Loads model with `torch.bfloat16` dtype
- Enables GPU model offloading for memory efficiency
- Processes requests sequentially using an async queue

#### 3. [`image_api_new.py`](image_api_new.py) - Remote API Client

Lightweight client that connects to the remote server:

- Uses `RemoteFluxPipeline` proxy instead of loading model locally
- Connects to `http://127.0.0.1:7001`
- Supports both HTTP and WebSocket connections
- Streams progressive blur results to the client

#### 4. [`remote_flux_pipeline.py`](remote_flux_pipeline.py) - Pipeline Proxy

HTTP client that mimics the HuggingFace `DiffusionPipeline` API:

- Provides `.to()`, `.enable_model_cpu_offload()` for API compatibility
- `__call__()` sends requests to remote server
- Returns `_FluxResult` object with `.images` list
- Properly handles base64 encoding/decoding

## How Files Are Interconnected

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Request                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  image_api.py (local)  OR  image_api_new.py (remote)           │
│  - FastAPI Server on port 8000                                  │
│  - Handles HTTP POST /text-to-image-stream                      │
│  - Handles WebSocket /ws/text-to-image                          │
└───────────┬─────────────────────────────────────────────────────┘
            │
            │ (if using remote mode)
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  remote_flux_pipeline.py                                        │
│  - RemoteFluxPipeline proxy client                              │
│  - Mimics DiffusionPipeline API                                 │
│  - Sends HTTP requests to flux_server                           │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  flux_server.py                                                 │
│  - FastAPI Server on port 7001                                  │
│  - Runs FLUX model on GPU                                       │
│  - Returns base64 images                                        │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### HTTP POST (image_api.py / image_api_new.py)

**Endpoint:** `POST /text-to-image-stream`

**Request:**
```json
{
  "prompt": "A sunset over mountains"
}
```

**Response:** Server-Sent Events (SSE) stream

Events streamed:
- `event: image` - Base64 encoded image (progressively less blurred)
- `event: error` - Error message
- `event: done` - Stream complete

### WebSocket (image_api.py / image_api_new.py)

**Endpoint:** `WS /ws/text-to-image`

1. Connect to WebSocket
2. Send prompt as text message
3. Receive events as text messages (same format as HTTP)

## How to Use

### Option 1: Local Mode (Direct GPU Usage)

Run `image_api.py` directly - it loads the model on your GPU:

```bash
python image_api.py
```

The server starts on `http://0.0.0.0:8000`

### Option 2: Remote Mode (Lightweight Client)

1. **Start the remote model server:**
   ```bash
   uvicorn flux_server:app --host 0.0.0.0 --port 7001
   ```

2. **Start the API client (in another terminal):**
   ```bash
   python image_api_new.py
   ```

The server starts on `http://0.0.0.0:8000`

### Making Requests

#### Using cURL (HTTP):
```bash
curl -X POST http://localhost:8000/text-to-image-stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful landscape"}'
```

#### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/text-to-image-stream",
    json={"prompt": "A beautiful landscape"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

## Requirements

All requirements are pre-installed in the `krea_env` conda environment. See the [Conda Environment Setup](#conda-environment-setup) section above for details.

Alternatively, install manually:

```bash
pip install fastapi uvicorn torch diffusers pillow httpx pydantic
```

- Python 3.10 (recommended)
- CUDA-capable GPU (for local mode with GPU acceleration)

Install dependencies:
```bash
pip install fastapi uvicorn torch diffusers pillow httpx pydantic
```

## Configuration

### Changing Model Variants

In [`flux_server.py`](flux_server.py):
```python
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", ...)
```

In [`image_api.py`](image_api.py):
```python
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", ...)
```

### Adjusting Generation Parameters

- `num_inference_steps`: Higher = more detail (default: 20)
- `guidance_scale`: Higher = more prompt adherence (default: 2.5)
- `width`, `height`: Output dimensions (default: 768x768)

## Logging

The application uses Python's logging module. Configure by modifying:
```python
logging.basicConfig(level=logging.INFO)
```

Log levels: DEBUG, INFO, WARNING, ERROR
