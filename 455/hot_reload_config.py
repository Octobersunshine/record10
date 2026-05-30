import os
import json
import time
import threading
import logging
from typing import Any, Dict, Optional, Callable, List, Tuple
from pathlib import Path
from collections import deque


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    pass


class HotReloadConfig:
    def __init__(
        self,
        config_path: str,
        reload_interval: float = 1.0,
        debounce_delay: float = 0.5,
        on_reload: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        validator: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_history: int = 10,
        auto_start: bool = True,
        name: Optional[str] = None
    ):
        self.config_path = Path(config_path)
        self.reload_interval = reload_interval
        self.debounce_delay = debounce_delay
        self.on_reload = on_reload
        self.on_error = on_error
        self.validator = validator
        self.max_history = max_history
        self.name = name or self.config_path.stem
        
        self._config: Dict[str, Any] = {}
        self._version: int = 0
        self._last_modified: float = 0.0
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        self._debounce_timer: Optional[threading.Timer] = None
        self._pending_mtime: float = 0.0
        self._debounce_lock = threading.Lock()
        
        self._history: deque = deque(maxlen=max_history)
        
        self._load_config()
        
        if auto_start:
            self.start_monitor()
    
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
                    new_config = json.load(f)
                
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
            except (FileNotFoundError, json.JSONDecodeError) as e:
                if self._version == 0:
                    raise RuntimeError(f"Failed to load initial config: {e}")
                error_msg = f"[{self.name}] Failed to load config, keeping old config: {e}"
                logger.warning(error_msg)
                if self.on_error:
                    self.on_error(e)
            except Exception as e:
                if self._version == 0:
                    raise RuntimeError(f"Failed to load initial config: {e}")
                error_msg = f"[{self.name}] Unexpected error loading config: {e}"
                logger.error(error_msg)
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
    
    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._check_and_reload()
            self._stop_event.wait(self.reload_interval)
    
    def start_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
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
            return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._config[key]
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f"'HotReloadConfig' object has no attribute '{name}'")
        with self._lock:
            if name in self._config:
                return self._config[name]
            raise AttributeError(f"'HotReloadConfig' object has no attribute '{name}'")
    
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
        return f"HotReloadConfig(name='{self.name}', path='{self.config_path}', version={self._version})"
