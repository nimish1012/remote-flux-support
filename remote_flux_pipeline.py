import httpx
import base64
import io
from typing import Optional
from PIL import Image


# ------------- Flux pipeline proxy (mimics DiffusionPipeline) -------------
class _FluxResult:
    """Wrapper to mimic the result object from DiffusionPipeline."""
    def __init__(self, images):
        self.images = images  # list[PIL.Image.Image]


class RemoteFluxPipeline:
    """
    Proxy client for a remote Flux model service.
    Mimics Hugging Face DiffusionPipeline API surface.
    """
    def __init__(self, base_url: str, timeout: float = 300.0):
        self._c = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def to(self, device: str):
        """
        Kept for API parity (no-op since remote handles device).
        Allows usage like: pipe.to("cuda")
        """
        return self

    def enable_model_cpu_offload(self):
        """
        API parity with DiffusionPipeline (no-op here).
        """
        return self

    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs
    ) -> _FluxResult:
        """
        Call the remote service to generate an image.
        Returns an object with `.images` (list of PIL.Image).
        """
        payload = {
            "prompt": prompt,
            "steps": num_inference_steps,
            "guidance": guidance_scale,
            "width": width,
            "height": height,
            "return_base64": True,
        }

        r = self._c.post("/generate", json=payload)
        r.raise_for_status()

        data = r.json()
        img_b64 = data.get("image_base64")
        if not img_b64:
            raise RuntimeError("Remote model service returned no image_base64")

        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        return _FluxResult([img])
    
    def close(self):
        self._c.close()
