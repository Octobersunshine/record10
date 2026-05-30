import time
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.concurrency import iterate_in_threadpool
from typing import List, Optional, Callable
import json
from .storage import MetricsStorage
from .alerts import AlertManager

logger = logging.getLogger(__name__)


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        storage: MetricsStorage,
        alert_manager: Optional[AlertManager] = None,
        exclude_paths: Optional[List[str]] = None,
        log_requests: bool = True,
        capture_body: bool = True,
        capture_headers: bool = True,
        max_body_size: int = 4096
    ):
        super().__init__(app)
        self.storage = storage
        self.alert_manager = alert_manager
        self.exclude_paths = exclude_paths or []
        self.log_requests = log_requests
        self.capture_body = capture_body
        self.capture_headers = capture_headers
        self.max_body_size = max_body_size

    def _should_exclude(self, path: str) -> bool:
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    async def _capture_request_body(self, request: Request) -> Optional[str]:
        if not self.capture_body:
            return None
        try:
            body = await request.body()
            if body and len(body) <= self.max_body_size:
                try:
                    decoded = body.decode("utf-8")
                    try:
                        parsed = json.loads(decoded)
                        return json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return decoded[:self.max_body_size]
                except Exception:
                    return f"<binary data, size={len(body)}>"
            elif body:
                return f"<body too large, size={len(body)}>"
        except Exception:
            return None
        return None

    def _capture_response_body(self, response: Response, response_body: bytes) -> Optional[str]:
        if not self.capture_body:
            return None
        if response_body and len(response_body) <= self.max_body_size:
            try:
                decoded = response_body.decode("utf-8")
                try:
                    parsed = json.loads(decoded)
                    return json.dumps(parsed, ensure_ascii=False)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return decoded[:self.max_body_size]
            except Exception:
                return f"<binary data, size={len(response_body)}>"
        elif response_body:
            return f"<body too large, size={len(response_body)}>"
        return None

    def _capture_headers(self, request: Request) -> dict:
        if not self.capture_headers:
            return {}
        headers = {}
        sensitive = {"authorization", "cookie", "set-cookie"}
        for key, value in request.headers.items():
            if key.lower() in sensitive:
                headers[key] = "***REDACTED***"
            else:
                headers[key] = value
        return headers

    def _evaluate_alerts(self):
        if self.alert_manager is None:
            return
        try:
            stats = self.storage.get_window_stats()
            if stats is None:
                return
            path_stats_dict = {
                f"{p['method']}:{p['path']}": p
                for p in stats.path_stats
            }
            self.alert_manager.evaluate(
                avg_response_time=stats.avg_response_time,
                p99_response_time=stats.p99_response_time,
                error_rate=stats.error_rate,
                throughput=stats.request_count / self.storage.window_seconds if self.storage.window_seconds > 0 else 0,
                path_stats=path_stats_dict
            )
        except Exception as e:
            logger.error(f"Alert evaluation error: {e}")

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        method = request.method

        if self._should_exclude(path):
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        headers = self._capture_headers(request)
        request_body = await self._capture_request_body(request)

        start_time = time.perf_counter()
        status_code = 500
        response_body = b""

        try:
            response = await call_next(request)
            status_code = response.status_code

            if self.capture_body:
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk

                original_headers = response.headers
                response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(original_headers),
                    media_type=response.media_type
                )
            else:
                response_body = b""

            return response
        except Exception as e:
            status_code = 500
            logger.error(f"Request failed: {method} {path} - {str(e)}")
            raise
        finally:
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000

            captured_response_body = self._capture_response_body(
                Response(status_code=status_code), response_body
            )

            self.storage.add_record(
                path=path,
                method=method,
                response_time=response_time,
                status_code=status_code,
                request_body=request_body,
                response_body=captured_response_body,
                request_id=request_id,
                headers=headers
            )

            self._evaluate_alerts()

            if self.log_requests:
                slow_marker = " [SLOW]" if response_time >= self.storage.slow_request_threshold_ms else ""
                logger.info(
                    f"[{method}] {path} - Status: {status_code} - "
                    f"Response Time: {response_time:.2f}ms"
                    f"{slow_marker} - RequestID: {request_id}"
                )
