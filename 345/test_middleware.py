import unittest
import json
import os
import tempfile
from unittest.mock import Mock, MagicMock
from datetime import datetime

from masking import (
    SensitiveDataMasker,
    mask_sensitive_data,
    mask_email,
    mask_phone,
    mask_ip,
)
from storage import (
    FileLogStorage,
    DatabaseLogStorage,
    CompositeLogStorage,
    create_storage,
)
from middleware import (
    RequestLoggerMiddleware,
    FlaskRequestLogger,
    log_api_request,
)


class TestSensitiveDataMasker(unittest.TestCase):
    def test_mask_simple_dict(self):
        masker = SensitiveDataMasker()
        data = {"username": "testuser", "password": "secret123"}
        result = masker.mask(data)
        self.assertEqual(result["username"], "testuser")
        self.assertEqual(result["password"], "********")

    def test_mask_nested_dict(self):
        masker = SensitiveDataMasker()
        data = {"user": {"email": "test@example.com", "password": "secret"}}
        result = masker.mask(data)
        self.assertEqual(result["user"]["email"], "********")
        self.assertEqual(result["user"]["password"], "********")

    def test_mask_list(self):
        masker = SensitiveDataMasker()
        data = [{"password": "pass1"}, {"password": "pass2", "name": "alice"}]
        result = masker.mask(data)
        self.assertEqual(result[0]["password"], "********")
        self.assertEqual(result[1]["name"], "alice")

    def test_custom_sensitive_fields(self):
        masker = SensitiveDataMasker(sensitive_fields=["api_key"])
        data = {"api_key": "abc123", "password": "secret"}
        result = masker.mask(data)
        self.assertEqual(result["api_key"], "********")
        self.assertEqual(result["password"], "secret")

    def test_preserve_length(self):
        masker = SensitiveDataMasker(preserve_length=True)
        data = {"password": "secret12345"}
        result = masker.mask(data)
        self.assertEqual(result["password"], "***********")
        self.assertEqual(len(result["password"]), len("secret12345"))

    def test_mask_email(self):
        result = mask_email("test@example.com")
        self.assertTrue("@" in result)
        self.assertTrue("***" in result)
        self.assertEqual(result, "t***t@example.com")

    def test_mask_phone(self):
        result = mask_phone("13812345678")
        self.assertEqual(result, "138****5678")

    def test_mask_ip(self):
        result = mask_ip("192.168.1.100", anonymize=True)
        self.assertEqual(result, "192.168.***.***")

    def test_mask_none_value(self):
        masker = SensitiveDataMasker()
        result = masker.mask(None)
        self.assertIsNone(result)

    def test_mask_non_sensitive_keeps_value(self):
        masker = SensitiveDataMasker()
        data = {"name": "Alice", "age": 30, "is_active": True}
        result = masker.mask(data)
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["age"], 30)
        self.assertEqual(result["is_active"], True)


class TestFileLogStorage(unittest.TestCase):
    def test_file_storage_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            try:
                log_entry = {
                    "timestamp": datetime.now(),
                    "ip_address": "127.0.0.1",
                    "request_method": "GET",
                    "request_path": "/api/test",
                    "response_status": 200,
                    "duration_ms": 123.45,
                }
                storage.save(log_entry)

                log_file = os.path.join(tmpdir, "test.log")
                self.assertTrue(os.path.exists(log_file))

                with open(log_file, "r", encoding="utf-8") as f:
                    line = f.readline().strip()
                    saved = json.loads(line)
                    self.assertEqual(saved["ip_address"], "127.0.0.1")
                    self.assertEqual(saved["request_method"], "GET")
                    self.assertEqual(saved["response_status"], 200)
            finally:
                storage.close()


class TestDatabaseLogStorage(unittest.TestCase):
    def test_db_storage_save_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = DatabaseLogStorage(db_path=db_path)

            log_entry = {
                "timestamp": datetime.now(),
                "ip_address": "192.168.1.1",
                "request_method": "POST",
                "request_path": "/api/users",
                "request_headers": {"Content-Type": "application/json"},
                "request_body": {"name": "test"},
                "response_status": 201,
                "duration_ms": 45.67,
                "user_agent": "TestAgent",
                "referer": "http://example.com",
                "extra_data": {"user_id": 1},
            }
            storage.save(log_entry)

            results = storage.query(limit=10)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["ip_address"], "192.168.1.1")
            self.assertEqual(results[0]["request_method"], "POST")
            self.assertEqual(results[0]["response_status"], 201)
            self.assertAlmostEqual(results[0]["duration_ms"], 45.67)

            results_by_ip = storage.query(ip="192.168.1.1")
            self.assertEqual(len(results_by_ip), 1)

            results_by_path = storage.query(path="users")
            self.assertEqual(len(results_by_path), 1)

            results_no_match = storage.query(ip="10.0.0.1")
            self.assertEqual(len(results_no_match), 0)


class TestCompositeLogStorage(unittest.TestCase):
    def test_composite_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            db_storage = DatabaseLogStorage(
                db_path=os.path.join(tmpdir, "test.db")
            )
            composite = CompositeLogStorage([file_storage, db_storage])
            try:
                log_entry = {
                    "timestamp": datetime.now(),
                    "ip_address": "127.0.0.1",
                    "request_method": "GET",
                    "request_path": "/test",
                    "response_status": 200,
                    "duration_ms": 10.0,
                }
                composite.save(log_entry)

                results = db_storage.query()
                self.assertEqual(len(results), 1)

                log_file = os.path.join(tmpdir, "test.log")
                self.assertTrue(os.path.exists(log_file))
            finally:
                composite.close()


class TestCreateStorage(unittest.TestCase):
    def test_create_file_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage(
                use_file=True,
                use_database=False,
                file_config={"log_dir": tmpdir},
            )
            try:
                self.assertIsInstance(storage, FileLogStorage)
            finally:
                storage.close()

    def test_create_db_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage(
                use_file=False,
                use_database=True,
                db_config={"db_path": os.path.join(tmpdir, "test.db")},
            )
            try:
                self.assertIsInstance(storage, DatabaseLogStorage)
            finally:
                storage.close()

    def test_create_composite_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage(
                use_file=True,
                use_database=True,
                file_config={"log_dir": tmpdir},
                db_config={"db_path": os.path.join(tmpdir, "test.db")},
            )
            try:
                self.assertIsInstance(storage, CompositeLogStorage)
            finally:
                storage.close()


class TestRequestLoggerMiddleware(unittest.TestCase):
    def _create_mock_request(self, method="GET", path="/api/test", data=None, headers=None):
        request = Mock()
        request.method = method
        request.path = path
        request.remote_addr = "127.0.0.1"
        request.content_type = "application/json"
        request.data = json.dumps(data).encode("utf-8") if data else b""
        request.headers = headers or {}

        def get_json(silent=True):
            if data is not None:
                return data
            return None

        request.get_json = get_json
        request.form = {}
        return request

    def _create_mock_response(self, status_code=200, data=None):
        response = Mock()
        response.status_code = status_code
        response.content_type = "application/json"
        response.data = json.dumps(data).encode("utf-8") if data else b""
        return response

    def test_build_log_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir)
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                request_data = {"username": "test", "password": "secret123"}
                request = self._create_mock_request(
                    method="POST",
                    path="/api/login",
                    data=request_data,
                    headers={"User-Agent": "TestAgent", "X-Forwarded-For": "10.0.0.1"},
                )
                response = self._create_mock_response(
                    status_code=200, data={"token": "jwt-token"}
                )

                start_time = 0
                log_entry = middleware._build_log_entry(request, response, start_time)

                self.assertEqual(log_entry["ip_address"], "10.0.0.1")
                self.assertEqual(log_entry["request_method"], "POST")
                self.assertEqual(log_entry["request_path"], "/api/login")
                self.assertEqual(log_entry["response_status"], 200)
                self.assertIsNotNone(log_entry["duration_ms"])
                self.assertEqual(log_entry["user_agent"], "TestAgent")

                self.assertEqual(log_entry["request_body"]["username"], "test")
                self.assertEqual(log_entry["request_body"]["password"], "********")
            finally:
                storage.close()

    def test_log_response_body_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir)
            middleware = RequestLoggerMiddleware(
                storage=storage, log_response_body=False
            )
            try:
                request = self._create_mock_request(data={"test": "value"})
                response = self._create_mock_response(data={"result": "ok"})

                log_entry = middleware._build_log_entry(request, response, 0)
                self.assertIsNone(log_entry["response_body"])
            finally:
                storage.close()

    def test_log_response_body_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir)
            middleware = RequestLoggerMiddleware(
                storage=storage, log_response_body=True
            )
            try:
                request = self._create_mock_request(data={"test": "value"})
                response = self._create_mock_response(data={"result": "ok"})

                log_entry = middleware._build_log_entry(request, response, 0)
                self.assertIsNotNone(log_entry["response_body"])
                self.assertEqual(log_entry["response_body"]["result"], "ok")
            finally:
                storage.close()

    def test_exclude_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir)
            middleware = RequestLoggerMiddleware(
                storage=storage, exclude_paths=["/health"]
            )
            try:
                request = self._create_mock_request(path="/health")
                self.assertFalse(middleware._should_log(request))

                request = self._create_mock_request(path="/api/test")
                self.assertTrue(middleware._should_log(request))
            finally:
                storage.close()

    def test_exclude_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir)
            middleware = RequestLoggerMiddleware(
                storage=storage, exclude_methods=["OPTIONS"]
            )
            try:
                request = self._create_mock_request(method="OPTIONS")
                self.assertFalse(middleware._should_log(request))

                request = self._create_mock_request(method="GET")
                self.assertTrue(middleware._should_log(request))
            finally:
                storage.close()

    def test_anonymize_ip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir)
            middleware = RequestLoggerMiddleware(
                storage=storage, anonymize_ip=True
            )
            try:
                request = self._create_mock_request()
                response = self._create_mock_response()

                log_entry = middleware._build_log_entry(request, response, 0)
                self.assertEqual(log_entry["ip_address"], "127.0.***.***")
            finally:
                storage.close()

    def test_truncate_large_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir)
            middleware = RequestLoggerMiddleware(
                storage=storage, max_body_size=100
            )
            try:
                large_data = {"field": "x" * 200}
                request = self._create_mock_request(data=large_data)
                response = self._create_mock_response()

                log_entry = middleware._build_log_entry(request, response, 0)
                self.assertTrue(log_entry["request_body"].get("truncated"))
                self.assertIn("preview", log_entry["request_body"])
            finally:
                storage.close()

    def test_log_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                request = self._create_mock_request(
                    method="POST", data={"name": "test", "password": "secret"}
                )
                response = self._create_mock_response(status_code=201)

                middleware.log_request(request, response, 0, user_id=123)

                log_file = os.path.join(tmpdir, "test.log")
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self.assertEqual(len(lines), 1)
                    saved = json.loads(lines[0])
                    self.assertEqual(saved["response_status"], 201)
                    self.assertEqual(saved["extra_data"]["user_id"], 123)
                    self.assertEqual(saved["request_body"]["password"], "********")
            finally:
                storage.close()

    def test_file_upload_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                request = self._create_mock_request(
                    method="POST",
                    data={"file": "large_file_content"},
                    headers={"Content-Type": "multipart/form-data; boundary=----WebKitFormBoundary"},
                )
                request.content_type = "multipart/form-data; boundary=----WebKitFormBoundary"
                response = self._create_mock_response(status_code=200)

                log_entry = middleware._build_log_entry(request, response, 0)

                self.assertIsNotNone(log_entry["request_body"])
                self.assertTrue(log_entry["request_body"].get("skipped"))
                self.assertEqual(log_entry["request_body"].get("reason"), "file_upload")
                self.assertIn("multipart/form-data", log_entry["request_body"].get("content_type", ""))
            finally:
                storage.close()

    def test_large_content_length_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                large_size = 2 * 1024 * 1024
                request = self._create_mock_request(
                    method="POST",
                    data={"test": "data"},
                    headers={"Content-Length": str(large_size)},
                )
                response = self._create_mock_response(status_code=200)

                log_entry = middleware._build_log_entry(request, response, 0)

                self.assertIsNotNone(log_entry["request_body"])
                self.assertTrue(log_entry["request_body"].get("skipped"))
                self.assertEqual(log_entry["request_body"].get("reason"), "too_large")
                self.assertEqual(log_entry["request_body"].get("content_length"), large_size)
            finally:
                storage.close()

    def test_large_data_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                large_data = b"x" * (2 * 1024 * 1024)
                request = self._create_mock_request(
                    method="POST",
                    data={"test": "data"},
                )
                request.data = large_data
                request.content_type = "application/octet-stream"
                request.headers = {"Content-Length": str(len(large_data))}
                response = self._create_mock_response(status_code=200)

                log_entry = middleware._build_log_entry(request, response, 0)

                self.assertIsNotNone(log_entry["request_body"])
                self.assertTrue(log_entry["request_body"].get("skipped"))
                self.assertEqual(log_entry["request_body"].get("reason"), "too_large")
            finally:
                storage.close()

    def test_binary_content_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                request = self._create_mock_request(
                    method="POST",
                    data={"test": "data"},
                )
                request.data = b"small binary data"
                request.content_type = "image/png"
                request.headers = {"Content-Length": "100"}
                response = self._create_mock_response(status_code=200)

                log_entry = middleware._build_log_entry(request, response, 0)

                self.assertIsNotNone(log_entry["request_body"])
                self.assertTrue(log_entry["request_body"].get("skipped"))
                self.assertEqual(log_entry["request_body"].get("reason"), "binary_content")
            finally:
                storage.close()

    def test_non_json_body_preview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage, max_body_size=200)
            try:
                request = self._create_mock_request(
                    method="POST",
                    data={"test": "data"},
                )
                request.data = b"plain text data that is not json format"
                request.content_type = "text/plain"
                response = self._create_mock_response(status_code=200)

                log_entry = middleware._build_log_entry(request, response, 0)

                self.assertIsNotNone(log_entry["request_body"])
                self.assertIn("raw_data_preview", log_entry["request_body"])
                self.assertIn("plain text", log_entry["request_body"]["raw_data_preview"])
            finally:
                storage.close()

    def test_custom_large_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            custom_threshold = 500 * 1024
            middleware = RequestLoggerMiddleware(
                storage=storage,
                large_request_threshold=custom_threshold,
            )
            try:
                self.assertEqual(middleware.large_request_threshold, custom_threshold)
            finally:
                storage.close()

    def test_max_body_size_default_1kb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                self.assertEqual(middleware.max_body_size, 1024)
            finally:
                storage.close()

    def test_large_json_body_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage, max_body_size=100)
            try:
                large_json = {"data": "x" * 2000, "password": "secret123"}
                request = self._create_mock_request(
                    method="POST",
                    data=large_json,
                )
                response = self._create_mock_response(status_code=200)

                middleware.log_request(request, response, 0)

                log_file = os.path.join(tmpdir, "test.log")
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self.assertEqual(len(lines), 1)
                    saved = json.loads(lines[0])
                    self.assertTrue(saved["request_body"].get("truncated"))
                    self.assertIn("preview", saved["request_body"])
            finally:
                storage.close()

    def test_skipped_body_not_masked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                request = self._create_mock_request(
                    method="POST",
                    data={"file": "content"},
                    headers={"Content-Type": "multipart/form-data"},
                )
                request.content_type = "multipart/form-data"
                response = self._create_mock_response(status_code=200)

                log_entry = middleware._build_log_entry(request, response, 0)

                self.assertTrue(log_entry["request_body"].get("skipped"))
                self.assertEqual(log_entry["request_body"].get("reason"), "file_upload")
            finally:
                storage.close()


class TestLargeFileHandling(unittest.TestCase):
    def test_is_file_upload_detection(self):
        from middleware import _is_file_upload

        self.assertFalse(_is_file_upload("multipart/form-data"))
        self.assertFalse(_is_file_upload("multipart/form-data; boundary=xxx"))
        self.assertTrue(_is_file_upload("application/octet-stream"))
        self.assertTrue(_is_file_upload("image/png"))
        self.assertTrue(_is_file_upload("video/mp4"))
        self.assertTrue(_is_file_upload("audio/mpeg"))
        self.assertTrue(_is_file_upload("application/pdf"))
        self.assertTrue(_is_file_upload("application/zip"))
        self.assertFalse(_is_file_upload("application/json"))
        self.assertFalse(_is_file_upload("text/plain"))
        self.assertFalse(_is_file_upload("application/x-www-form-urlencoded"))
        self.assertFalse(_is_file_upload(""))
        self.assertFalse(_is_file_upload(None))

    def test_get_content_length(self):
        from middleware import _get_content_length

        request = Mock()
        request.headers = {"Content-Length": "1024"}
        self.assertEqual(_get_content_length(request), 1024)

        request.headers = {"Content-Length": "invalid"}
        self.assertIsNone(_get_content_length(request))

        request.headers = {}
        request.content_length = 2048
        self.assertEqual(_get_content_length(request), 2048)

        request.headers = {}
        request.content_length = None
        self.assertIsNone(_get_content_length(request))

    def test_large_response_body_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(
                storage=storage,
                log_response_body=True,
            )
            try:
                from middleware import _parse_response_body

                response = Mock()
                response.status_code = 200
                response.content_type = "application/octet-stream"
                response.data = b"x" * (2 * 1024 * 1024)

                result = _parse_response_body(response)
                self.assertIsNotNone(result)
                self.assertTrue(result.get("skipped"))
                self.assertEqual(result.get("reason"), "too_large")
            finally:
                storage.close()

    def test_form_data_with_large_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            middleware = RequestLoggerMiddleware(storage=storage)
            try:
                request = Mock()
                request.method = "POST"
                request.path = "/api/upload"
                request.remote_addr = "127.0.0.1"
                request.content_type = "application/x-www-form-urlencoded"
                request.data = b""
                request.headers = {"Content-Length": str(3 * 1024 * 1024)}
                request.get_json = lambda silent=True: None
                request.form = {"name": "test_file", "description": "test"}

                response = Mock()
                response.status_code = 200
                response.content_type = "application/json"
                response.data = b""

                log_entry = middleware._build_log_entry(request, response, 0)

                self.assertIsNotNone(log_entry["request_body"])
                self.assertEqual(log_entry["request_body"].get("name"), "test_file")
                self.assertTrue(log_entry["request_body"].get("_skipped_files"))
            finally:
                storage.close()


class TestLogApiRequestDecorator(unittest.TestCase):
    def test_decorator_with_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            logger = RequestLoggerMiddleware(storage=storage)
            try:
                @log_api_request(logger=logger)
                def test_func(request):
                    return Mock(status_code=200, content_type="application/json", data=b"")

                request = Mock()
                request.method = "GET"
                request.path = "/test"
                request.remote_addr = "127.0.0.1"
                request.content_type = "application/json"
                request.data = b""
                request.headers = {}
                request.get_json = lambda silent=True: None
                request.form = {}

                response = test_func(request=request)
                self.assertEqual(response.status_code, 200)

                log_file = os.path.join(tmpdir, "test.log")
                self.assertTrue(os.path.exists(log_file))
            finally:
                storage.close()

    def test_decorator_with_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileLogStorage(log_dir=tmpdir, filename="test.log")
            logger = RequestLoggerMiddleware(storage=storage)
            try:
                @log_api_request(logger=logger)
                def test_func(request):
                    raise ValueError("Test error")

                request = Mock()
                request.method = "GET"
                request.path = "/test"
                request.remote_addr = "127.0.0.1"
                request.content_type = "application/json"
                request.data = b""
                request.headers = {}
                request.get_json = lambda silent=True: None
                request.form = {}

                with self.assertRaises(ValueError):
                    test_func(request=request)

                log_file = os.path.join(tmpdir, "test.log")
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self.assertEqual(len(lines), 1)
                    saved = json.loads(lines[0])
                    self.assertEqual(saved["response_status"], 500)
                    self.assertEqual(saved["extra_data"]["error"], "Test error")
            finally:
                storage.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
