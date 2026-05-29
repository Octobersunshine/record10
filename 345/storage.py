import json
import csv
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
from unittest.mock import Mock


def _json_serialize_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Mock):
        return str(obj)[:100]
    if hasattr(obj, "__str__"):
        try:
            return str(obj)[:100]
        except Exception:
            return "<non-serializable>"
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class BaseLogStorage(ABC):
    @abstractmethod
    def save(self, log_entry: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class FileLogStorage(BaseLogStorage):
    _instance_counter = 0

    def __init__(
        self,
        log_dir: str = "logs",
        filename: str = "api_requests.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        FileLogStorage._instance_counter += 1
        self.instance_id = FileLogStorage._instance_counter

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename

        self.logger = logging.getLogger(f"api_request_logger_{self.instance_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.handler = None
        if not self.logger.handlers:
            from logging.handlers import RotatingFileHandler

            self.handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            formatter = logging.Formatter("%(message)s")
            self.handler.setFormatter(formatter)
            self.logger.addHandler(self.handler)

    def save(self, log_entry: Dict[str, Any]) -> None:
        log_line = json.dumps(log_entry, ensure_ascii=False, default=_json_serialize_default)
        self.logger.info(log_line)

    def close(self) -> None:
        if self.handler:
            self.handler.close()
            self.logger.removeHandler(self.handler)
            self.handler = None


class DatabaseLogStorage(BaseLogStorage):
    def __init__(
        self,
        db_path: str = "logs/api_requests.db",
        table_name: str = "api_request_logs",
    ):
        self.db_path = db_path
        self.table_name = table_name
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self._init_table()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_table(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    ip_address TEXT,
                    request_method TEXT NOT NULL,
                    request_path TEXT NOT NULL,
                    request_headers TEXT,
                    request_body TEXT,
                    response_status INTEGER,
                    response_body TEXT,
                    duration_ms REAL,
                    user_agent TEXT,
                    referer TEXT,
                    extra_data TEXT
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON {self.table_name}(timestamp)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_ip ON {self.table_name}(ip_address)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_path ON {self.table_name}(request_path)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_status ON {self.table_name}(response_status)
            """)

    def save(self, log_entry: Dict[str, Any]) -> None:
        timestamp = log_entry.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.table_name} (
                    timestamp, ip_address, request_method, request_path,
                    request_headers, request_body, response_status,
                    response_body, duration_ms, user_agent, referer, extra_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                log_entry.get("ip_address"),
                log_entry.get("request_method"),
                log_entry.get("request_path"),
                json.dumps(log_entry.get("request_headers", {}), ensure_ascii=False, default=_json_serialize_default) if log_entry.get("request_headers") else None,
                json.dumps(log_entry.get("request_body", {}), ensure_ascii=False, default=_json_serialize_default) if log_entry.get("request_body") else None,
                log_entry.get("response_status"),
                json.dumps(log_entry.get("response_body", {}), ensure_ascii=False, default=_json_serialize_default) if log_entry.get("response_body") else None,
                log_entry.get("duration_ms"),
                log_entry.get("user_agent"),
                log_entry.get("referer"),
                json.dumps(log_entry.get("extra_data", {}), ensure_ascii=False, default=_json_serialize_default) if log_entry.get("extra_data") else None,
            ))

    def query(
        self,
        ip: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        status_code_min: Optional[int] = None,
        status_code_max: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        query = f"SELECT * FROM {self.table_name} WHERE 1=1"
        params = []

        if ip:
            query += " AND ip_address = ?"
            params.append(ip)

        if path:
            query += " AND request_path LIKE ?"
            params.append(f"%{path}%")

        if method:
            query += " AND request_method = ?"
            params.append(method.upper())

        if status_code is not None:
            query += " AND response_status = ?"
            params.append(status_code)

        if status_code_min is not None:
            query += " AND response_status >= ?"
            params.append(status_code_min)

        if status_code_max is not None:
            query += " AND response_status <= ?"
            params.append(status_code_max)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def count(
        self,
        ip: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        status_code_min: Optional[int] = None,
        status_code_max: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        query = f"SELECT COUNT(*) as cnt FROM {self.table_name} WHERE 1=1"
        params = []

        if ip:
            query += " AND ip_address = ?"
            params.append(ip)

        if path:
            query += " AND request_path LIKE ?"
            params.append(f"%{path}%")

        if method:
            query += " AND request_method = ?"
            params.append(method.upper())

        if status_code is not None:
            query += " AND response_status = ?"
            params.append(status_code)

        if status_code_min is not None:
            query += " AND response_status >= ?"
            params.append(status_code_min)

        if status_code_max is not None:
            query += " AND response_status <= ?"
            params.append(status_code_max)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def get_status_distribution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        query = f"SELECT response_status, COUNT(*) as count FROM {self.table_name} WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " GROUP BY response_status ORDER BY response_status"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def export_csv(
        self,
        output_path: str,
        ip: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        status_code_min: Optional[int] = None,
        status_code_max: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> str:
        rows = self.query(
            ip=ip,
            path=path,
            method=method,
            status_code=status_code,
            status_code_min=status_code_min,
            status_code_max=status_code_max,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        csv_columns = [
            "id", "timestamp", "ip_address", "request_method",
            "request_path", "response_status", "duration_ms",
            "user_agent", "referer", "request_body", "response_body",
            "request_headers", "extra_data",
        ]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return str(output)


class CompositeLogStorage(BaseLogStorage):
    def __init__(self, storages: list):
        self.storages = storages

    def save(self, log_entry: Dict[str, Any]) -> None:
        for storage in self.storages:
            try:
                storage.save(log_entry)
            except Exception as e:
                print(f"Failed to save log to {type(storage).__name__}: {e}")

    def close(self) -> None:
        for storage in self.storages:
            try:
                storage.close()
            except Exception as e:
                print(f"Failed to close storage {type(storage).__name__}: {e}")


def create_storage(
    use_file: bool = True,
    use_database: bool = False,
    file_config: Optional[Dict[str, Any]] = None,
    db_config: Optional[Dict[str, Any]] = None,
) -> BaseLogStorage:
    storages = []

    if use_file:
        config = file_config or {}
        storages.append(FileLogStorage(**config))

    if use_database:
        config = db_config or {}
        storages.append(DatabaseLogStorage(**config))

    if len(storages) == 1:
        return storages[0]

    return CompositeLogStorage(storages)
