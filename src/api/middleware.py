"""Request-id correlation, access logging and Prometheus HTTP metrics (pure ASGI - streaming safe)."""

from __future__ import annotations

import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.utils.logger import get_logger, request_id_var
from src.utils.metrics import REQUEST_LATENCY, REQUESTS

logger = get_logger("api.access")


class RequestContextMiddleware:
    """Runs around EVERY request:

        request in ─► pick/create request_id ─► put it in a context variable (logger reads it)
                   ─► call the endpoint
                   ─► add "x-request-id" header to the response
                   ─► record metrics + one access-log line

    Written as "pure ASGI" (not BaseHTTPMiddleware) so it doesn't buffer streaming responses.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode()[:64]
        # Reuse the caller's id if they sent one (lets a frontend/gateway correlate its own logs)
        request_id = incoming or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = scope.get("route")
            path = getattr(route, "path", scope["path"])  # templated path keeps metric cardinality low
            REQUESTS.labels(scope["method"], path, str(status)).inc()
            REQUEST_LATENCY.labels(path).observe(elapsed)
            if path not in ("/health", "/metrics"):
                logger.info(
                    "request",
                    extra={
                        "method": scope["method"],
                        "path": path,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 1),
                    },
                )
            request_id_var.reset(token)
