# Remote Mode Architecture

The remote mode works as a **client-server architecture** where image generation is offloaded to a separate server.

## Architecture

### 1. Remote Server (`flux_server.py` - runs on port 7001)
- Hosts the actual ML model (Flux/DiffusionPipeline)
- Exposes a `/generate` endpoint that accepts prompt, steps, guidance, width, height
- Returns the generated image as base64

### 2. API Server (`image_api_new.py` - runs on port 8000)
- Acts as a **proxy client** to the remote server
- Uses `RemoteFluxPipeline` class which mimics the HuggingFace DiffusionPipeline API
- The proxy is initialized at startup with `RemoteFluxPipeline(base_url="http://127.0.0.1:7001")`

## How It Works

1. **Client sends request** to `POST /text-to-image-stream` or WebSocket `/ws/text-to-image`
2. **Request is queued** - only one generation runs at a time via `request_queue`
3. **Queue processor** calls `pipe()` (the RemoteFluxPipeline) which:
   - Sends HTTP POST to `http://127.0.0.1:7001/generate`
   - Receives base64-encoded image
   - Decodes and returns PIL Image
4. **Streams progressive previews** back to client with decreasing blur (10→5→2→0)

## Benefits
- **Lightweight API server** - no GPU required
- **Remote ML server** can be on a powerful GPU machine
- Client code stays the same whether using local or remote pipeline
