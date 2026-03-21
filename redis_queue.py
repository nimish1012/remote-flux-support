import hashlib
import os

import redis.asyncio as aioredis
import redis as syncredis

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
JOB_QUEUE  = "flux:jobs"
RESULT_TTL = int(os.getenv("RESULT_TTL", 3600))   # seconds to cache results


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def job_channel(job_id: str) -> str:
    """Pub/Sub channel for a specific job's progress events."""
    return f"flux:progress:{job_id}"


def cache_key(prompt: str, width: int, height: int,
              steps: int, guidance: float) -> str:
    """
    Stable cache key that includes all generation params so different
    sizes / quality settings don't collide on the same prompt.
    """
    fingerprint = f"{prompt}|{width}|{height}|{steps}|{guidance}"
    return f"flux:cache:{hashlib.sha256(fingerprint.encode()).hexdigest()}"


# ---------------------------------------------------------------------------
# Connection factories
# ---------------------------------------------------------------------------

async def get_async_redis() -> aioredis.Redis:
    """Return a new async Redis client."""
    return await aioredis.from_url(REDIS_URL, decode_responses=True)


def get_sync_redis() -> syncredis.Redis:
    """Return a new synchronous Redis client (used inside the worker)."""
    return syncredis.from_url(REDIS_URL, decode_responses=True)
