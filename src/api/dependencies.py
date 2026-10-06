"""FastAPI dependencies: container access, API-key auth with roles, per-key rate limiting."""

from __future__ import annotations

import hmac
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, Header, Request

from src.pipeline.container import Container
from src.utils.exceptions import AuthError, RateLimitExceeded
from src.utils.helpers import sha256


@dataclass
class Principal:
    key_id: str  # hashed key - safe to log
    role: str


def get_container_dep(request: Request) -> Container:
    return request.app.state.container


class TokenBucketLimiter:
    """In-process token bucket per key. For multiple replicas move this to Redis / the API gateway."""

    def __init__(self, per_minute: int) -> None:
        self.capacity, self.rate = per_minute, per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            return True


def get_principal(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    container: Container = Depends(get_container_dep),
) -> Principal:
    sec = container.settings.security
    if not sec.auth_enabled:
        principal = Principal(key_id="anonymous", role=container.settings.app.default_role)
    else:
        if not x_api_key:
            raise AuthError("Missing X-API-Key header")
        role = None
        for key, key_role in container.settings.api_keys.items():
            if hmac.compare_digest(key, x_api_key):  # constant-time comparison
                role = key_role
                break
        if role is None:
            raise AuthError("Invalid API key")
        principal = Principal(key_id=sha256(x_api_key)[:12], role=role)

    limiter: TokenBucketLimiter = request.app.state.rate_limiter
    if not limiter.allow(principal.key_id):
        raise RateLimitExceeded("Rate limit exceeded, retry shortly")
    return principal


def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if principal.role != "hr_admin":
        err = AuthError("hr_admin role required")
        err.status_code = 403
        err.code = "forbidden"
        raise err
    return principal
