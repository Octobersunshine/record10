import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

from masking import SensitiveDataMasker, mask_ip
from storage import BaseLogStorage, create_storage, DatabaseLogStorage


LARGE_REQUEST_THRESHOLD = 1024 * 1024
SKIP_BODY_CONTENT_TYPES = [
    "application/octet-stream",
    "image/",
    "video/",
    "audio/",
    "application/pdf",
    "application/zip",
    "application/x-rar-compressed",
]


def _get_content_length(request) -> Optional[int]:
    if hasattr(request, "headers"):
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                return int(content_length)
            except (ValueError, TypeError):
                pass
    if hasattr(request, "content_length"):
        content_length = request.content_length
        if isinstance(content_length, int) and content_length is not None:
            return content_length
    return None


def _is_file_upload(content_type: str) -> bool:
    if not content_type:
        return False
    content_type_lower = content_type.lower()
    return any(
        skip_type in content_type_lower for skip_type in SKIP_BODY_CONTENT_TYPES
    )


def _get_client_ip(request) -> str:
    if hasattr(request, "headers"):
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        x_real_ip = request.headers.get("X-Real-IP")
        if x_real_ip:
            return x_real_ip
    if hasattr(request, "remote_addr"):
        return request.remote_addr
    return "unknown"


def _parse_request_body(
    request,
    max_body_size: int = 1024,
    large_request_threshold: int = LARGE_REQUEST_THRESHOLD,
) -> Any:
    content_type = request.content_type if hasattr(request, "content_type") else ""

    if content_type and "multipart/form-data" in content_type.lower():
        return {"skipped": True, "reason": "file_upload", "content_type": content_type}

    content_length = _get_content_length(request)
    is_large = isinstance(content_length, int) and content_length > large_request_threshold

    if is_large and not (content_type and "application/x-www-form-urlencoded" in content_type.lower()):
        return {
            "skipped": True,
            "reason": "too_large",
            "content_length": content_length,
            "content_type": content_type,
        }

    if _is_file_upload(content_type):
        return {"skipped": True, "reason": "binary_content", "content_type": content_type}

    if content_type and "application/json" in content_type.lower():
        try:
            if hasattr(request, "get_json"):
                body = request.get_json(silent=True)
                if body is not None:
                    return body
            if hasattr(request, "json"):
                return request.json
        except Exception:
            pass

    if hasattr(request, "data") and request.data:
        data = request.data
        if len(data) > large_request_threshold and not is_large:
            return {
                "skipped": True,
                "reason": "too_large",
                "content_length": len(data),
                "content_type": content_type,
            }
        try:
            preview = data[:max_body_size]
            try:
                parsed = json.loads(data.decode("utf-8"))
                return parsed
            except Exception:
                return {"raw_data_preview": preview.decode("utf-8", errors="replace")}
        except Exception:
            return {"raw_data_preview": str(data[:max_body_size])}

    if hasattr(request, "form") and request.form:
        form_data = dict(request.form)
        if is_large:
            form_data["_skipped_files"] = True
        return form_data

    return None


def _parse_response_body(
    response,
    max_body_size: int = 1024,
    large_request_threshold: int = LARGE_REQUEST_THRESHOLD,
) -> Any:
    if not hasattr(response, "data") and not hasattr(response, "body"):
        return None

    try:
        data = getattr(response, "data", None) or getattr(response, "body", None)
        if data and isinstance(data, (bytes, bytearray)):
            if len(data) > large_request_threshold:
                return {
                    "skipped": True,
                    "reason": "too_large",
                    "content_length": len(data),
                }
            content_type = getattr(response, "content_type", "")
            if content_type and "application/json" in content_type.lower():
                try:
                    return json.loads(data.decode("utf-8"))
                except Exception:
                    return {"raw_data_preview": str(data[:max_body_size])}
            return {"raw_data_preview": str(data[:max_body_size])}
        if isinstance(data, (dict, list)):
            return data
    except Exception:
        pass

    return None


class RequestLoggerMiddleware:
    def __init__(
        self,
        storage: Optional[BaseLogStorage] = None,
        sensitive_fields: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_methods: Optional[List[str]] = None,
        anonymize_ip: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024,
        large_request_threshold: int = LARGE_REQUEST_THRESHOLD,
        skip_file_upload_body: bool = True,
        alert_manager: Optional[Any] = None,
        before_request: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
        after_request: Optional[Callable[[Any, Any, Dict[str, Any]], None]] = None,
        **storage_kwargs,
    ):
        self.storage = storage or create_storage(**storage_kwargs)
        self.masker = SensitiveDataMasker(sensitive_fields=sensitive_fields)
        self.exclude_paths = set(exclude_paths or [])
        self.exclude_methods = set(m.upper() for m in (exclude_methods or []))
        self.anonymize_ip = anonymize_ip
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.large_request_threshold = large_request_threshold
        self.skip_file_upload_body = skip_file_upload_body
        self.alert_manager = alert_manager
        self.before_request_hook = before_request
        self.after_request_hook = after_request

    def _should_log(self, request) -> bool:
        if hasattr(request, "method") and request.method in self.exclude_methods:
            return False
        if hasattr(request, "path") and request.path in self.exclude_paths:
            return False
        return True

    def _build_log_entry(
        self,
        request,
        response,
        start_time: float,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        duration_ms = (time.time() - start_time) * 1000

        request_body = _parse_request_body(
            request,
            max_body_size=self.max_body_size,
            large_request_threshold=self.large_request_threshold,
        )

        if request_body:
            is_skipped = isinstance(request_body, dict) and request_body.get("skipped", False)
            if not is_skipped:
                request_body = self._truncate_body(request_body)
                request_body = self.masker.mask(request_body)

        response_body = None
        if self.log_response_body and response is not None:
            response_body = _parse_response_body(
                response,
                max_body_size=self.max_body_size,
                large_request_threshold=self.large_request_threshold,
            )
            if response_body:
                is_skipped = isinstance(response_body, dict) and response_body.get("skipped", False)
                if not is_skipped:
                    response_body = self._truncate_body(response_body)
                    response_body = self.masker.mask(response_body)

        ip_address = _get_client_ip(request)
        if self.anonymize_ip:
            ip_address = mask_ip(ip_address, anonymize=True)

        headers = {}
        if hasattr(request, "headers"):
            headers = dict(request.headers)
            headers = self.masker.mask(headers)

        log_entry = {
            "timestamp": datetime.now(),
            "ip_address": ip_address,
            "request_method": request.method if hasattr(request, "method") else None,
            "request_path": request.path if hasattr(request, "path") else None,
            "request_headers": headers,
            "request_body": request_body,
            "response_status": response.status_code if hasattr(response, "status_code") else None,
            "response_body": response_body,
            "duration_ms": round(duration_ms, 2),
            "user_agent": headers.get("User-Agent") if headers else None,
            "referer": headers.get("Referer") if headers else None,
            "extra_data": extra_data or {},
        }

        return log_entry

    def _truncate_body(self, body: Any) -> Any:
        if isinstance(body, (dict, list)):
            body_str = json.dumps(body, ensure_ascii=False)
            if len(body_str) > self.max_body_size:
                return {"truncated": True, "preview": body_str[: self.max_body_size] + "..."}
        return body

    def log_request(self, request, response, start_time: float, **extra) -> None:
        if not self._should_log(request):
            return

        try:
            log_entry = self._build_log_entry(request, response, start_time, extra)
            self.storage.save(log_entry)

            if self.alert_manager:
                self.alert_manager.check(log_entry)
        except Exception as e:
            print(f"Failed to log request: {e}")
            traceback.print_exc()


class FlaskRequestLogger(RequestLoggerMiddleware):
    def __init__(self, app=None, **kwargs):
        super().__init__(**kwargs)
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        from flask import request

        @app.before_request
        def before_request():
            if not self._should_log(request):
                return
            request._start_time = time.time()
            request._extra_data = {}
            if self.before_request_hook:
                self.before_request_hook(request, request._extra_data)

        @app.after_request
        def after_request(response):
            if not self._should_log(request):
                return response
            start_time = getattr(request, "_start_time", time.time())
            extra_data = getattr(request, "_extra_data", {})
            if self.after_request_hook:
                self.after_request_hook(request, response, extra_data)
            self.log_request(request, response, start_time, **extra_data)
            return response


def log_api_request(
    logger: Optional[RequestLoggerMiddleware] = None,
    **middleware_kwargs,
) -> Callable:
    middleware = logger or RequestLoggerMiddleware(**middleware_kwargs)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            request = kwargs.get("request") or (args[0] if args else None)
            start_time = time.time()
            extra_data = {}

            try:
                if middleware.before_request_hook and request:
                    middleware.before_request_hook(request, extra_data)

                result = func(*args, **kwargs)

                if middleware.after_request_hook and request:
                    middleware.after_request_hook(request, result, extra_data)

                if request:
                    middleware.log_request(request, result, start_time, **extra_data)

                return result
            except Exception as e:
                if request:
                    class MockResponse:
                        status_code = 500

                    middleware.log_request(
                        request, MockResponse(), start_time, error=str(e), **extra_data
                    )
                raise

        return wrapper

    return decorator
