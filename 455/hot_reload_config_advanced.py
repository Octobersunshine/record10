import os
import json
import time
import threading
import logging
from typing import Any, Dict, Optional, Callable, Union, List
from pathlib import Path
from enum import Enum
from collections import deque


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    pass


class ConfigFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    AUTO = "auto"


class HotReloadConfigAdvanced:
    def __init__(
        self,
        config_path: str,
        reload_interval: float = 1.0,
        debounce_delay: float = 0.5,
        on_reload: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        validator: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_history: int = 10,
        config_format: ConfigFormat = ConfigFormat.AUTO,
        auto_start: bool = True,
        use_polling: bool = True,
        name: Optional[str] = None
    ):
        self.config_path = Path(config_path)
        self.reload_interval = reload_interval
        self.debounce_delay = debounce_delay
        self.on_reload = on_reload
        self.on_error = on_error
        self.validator = validator
        self.max_history = max_history
        self.config_format = config_format
        self.use_polling = use_polling
        self.name = name or self.config_path.stem
        
        self._config: Dict[str, Any] = {}
        self._version: int = 0
        self._last_modified: float = 0.0
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._observer = None
        
        self._debounce_timer: Optional[threading.Timer] = None
        self._pending_mtime: float = 0.0
        self._debounce_lock = threading.Lock()
        
        self._history: deque = deque(maxlen=max_history)
        
        self._determine_format()
        self._load_config()
        
        if auto_start:
            self.start_monitor()
    
    def _determine_format(self) -> None:
        if self.config_format == ConfigFormat.AUTO:
            suffix = self.config_path.suffix.lower()
            if suffix in ('.yaml', '.yml'):
                self.config_format = ConfigFormat.YAML
            else:
                self.config_format = ConfigFormat.JSON
    
    def _parse_config(self, content: str) -> Dict[str, Any]:
        if self.config_format == ConfigFormat.YAML:
            try:
                import yaml
                return yaml.safe_load(content) or {}
            except ImportError:
                raise RuntimeError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        else:
            return json.loads(content)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        if self.validator is not None:
            try:
                self.validator(config)
            except Exception as e:
                raise ConfigValidationError(f"Config validation failed: {e}") from e
    
    def _save_to_history(self) -> None:
        if self._version > 0:
            self._history.append({
                'version': self._version,
                'config': self._deep_copy(self._config),
                'timestamp': time.time(),
                'mtime': self._last_modified
            })
    
    def _deep_copy(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def _load_config(self) -> None:
        with self._lock:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_config = self._parse_config(content)
                
                self._validate_config(new_config)
                
                self._save_to_history()
                
                self._config = new_config
                self._version += 1
                self._last_modified = os.path.getmtime(self.config_path)
                
                if self.on_reload and self._version > 1:
                    self.on_reload(self._config)
                    
            except ConfigValidationError as e:
                error_msg = f"[{self.name}] Config validation failed, keeping old config: {e}"
                logger.warning(error_msg)
                if self.on_error:
                    self.on_error(e)
            except Exception as e:
                if self._version == 0:
                    raise RuntimeError(f"Failed to load initial config: {e}")
                error_msg = f"[{self.name}] Failed to load config, keeping old config: {e}"
                logger.warning(error_msg)
                if self.on_error:
                    self.on_error(e)
    
    def rollback(self, steps: int = 1) -> bool:
        with self._lock:
            if steps < 1 or steps > len(self._history):
                return False
            
            for _ in range(steps - 1):
                if self._history:
                    self._history.pop()
            
            if not self._history:
                return False
            
            history_entry = self._history.pop()
            self._config = history_entry['config']
            self._last_modified = history_entry['mtime']
            self._version += 1
            
            logger.info(f"[{self.name}] Rolled back to version {history_entry['version']}")
            
            if self.on_reload:
                self.on_reload(self._config)
            
            return True
    
    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    'version': entry['version'],
                    'timestamp': entry['timestamp'],
                    'config': self._deep_copy(entry['config'])
                }
                for entry in self._history
            ]
    
    def can_rollback(self, steps: int = 1) -> bool:
        with self._lock:
            return steps >= 1 and steps <= len(self._history)
    
    def _execute_debounced_load(self) -> None:
        with self._debounce_lock:
            self._debounce_timer = None
            self._load_config()
    
    def _schedule_debounced_load(self, new_mtime: float) -> None:
        with self._debounce_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            
            self._pending_mtime = new_mtime
            self._debounce_timer = threading.Timer(
                self.debounce_delay,
                self._execute_debounced_load
            )
            self._debounce_timer.start()
    
    def _cancel_debounced_load(self) -> None:
        with self._debounce_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None
    
    def _check_and_reload(self) -> None:
        try:
            current_mtime = os.path.getmtime(self.config_path)
            if current_mtime > self._last_modified:
                self._schedule_debounced_load(current_mtime)
        except FileNotFoundError:
            pass
    
    def _monitor_loop_polling(self) -> None:
        while not self._stop_event.is_set():
            self._check_and_reload()
            self._stop_event.wait(self.reload_interval)
    
    def _monitor_loop_observer(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class ConfigHandler(FileSystemEventHandler):
                def __init__(self, config_obj):
                    self.config_obj = config_obj
                
                def on_modified(self, event):
                    if not event.is_directory and Path(event.src_path) == self.config_obj.config_path:
                        try:
                            current_mtime = os.path.getmtime(event.src_path)
                            self.config_obj._schedule_debounced_load(current_mtime)
                        except FileNotFoundError:
                            pass
            
            self._observer = Observer()
            handler = ConfigHandler(self)
            self._observer.schedule(handler, str(self.config_path.parent), recursive=False)
            self._observer.start()
            
            while not self._stop_event.is_set():
                self._stop_event.wait(0.1)
            
            self._observer.stop()
            self._observer.join()
        except ImportError:
            self._monitor_loop_polling()
    
    def start_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_event.clear()
        
        if not self.use_polling:
            target = self._monitor_loop_observer
        else:
            target = self._monitor_loop_polling
        
        self._monitor_thread = threading.Thread(
            target=target,
            daemon=True,
            name=f"ConfigMonitor-{self.name}"
        )
        self._monitor_thread.start()
    
    def stop_monitor(self) -> None:
        self._stop_event.set()
        
        self._cancel_debounced_load()
        
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'HotReloadConfigAdvanced' object has no attribute '{name}'")
        with self._lock:
            if name in self._config:
                value = self._config[name]
                if isinstance(value, dict):
                    return _AttrDict(value)
                return value
            raise AttributeError(f"'HotReloadConfigAdvanced' object has no attribute '{name}'")
    
    @property
    def version(self) -> int:
        with self._lock:
            return self._version
    
    @property
    def config(self) -> Dict[str, Any]:
        with self._lock:
            return self._deep_copy(self._config)
    
    def reload_now(self) -> None:
        self._cancel_debounced_load()
        self._load_config()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitor()
        return False
    
    def __repr__(self) -> str:
        return f"HotReloadConfigAdvanced(name='{self.name}', path='{self.config_path}', version={self._version})"


class _AttrDict:
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _AttrDict(value)
            return value
        raise AttributeError(f"No attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()
