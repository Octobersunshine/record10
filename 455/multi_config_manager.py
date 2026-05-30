import os
import threading
import logging
from typing import Any, Dict, Optional, Callable, List, Union, Iterator
from pathlib import Path
from collections import defaultdict

from hot_reload_config import HotReloadConfig, ConfigValidationError
from hot_reload_config_advanced import HotReloadConfigAdvanced, ConfigFormat


logger = logging.getLogger(__name__)


class ConfigSpec:
    def __init__(
        self,
        config_path: str,
        name: Optional[str] = None,
        validator: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_reload: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        use_advanced: bool = True,
        **kwargs
    ):
        self.config_path = Path(config_path)
        self.name = name or self.config_path.stem
        self.validator = validator
        self.on_reload = on_reload
        self.on_error = on_error
        self.use_advanced = use_advanced
        self.kwargs = kwargs


class MultiConfigManager:
    def __init__(
        self,
        config_dir: Optional[str] = None,
        default_reload_interval: float = 1.0,
        default_debounce_delay: float = 0.5,
        default_max_history: int = 10,
        auto_discover: bool = False,
        discover_pattern: str = "*.json",
        on_any_reload: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_any_error: Optional[Callable[[str, Exception], None]] = None
    ):
        self.config_dir = Path(config_dir) if config_dir else None
        self.default_reload_interval = default_reload_interval
        self.default_debounce_delay = default_debounce_delay
        self.default_max_history = default_max_history
        self.auto_discover = auto_discover
        self.discover_pattern = discover_pattern
        self.on_any_reload = on_any_reload
        self.on_any_error = on_any_error
        
        self._configs: Dict[str, Union[HotReloadConfig, HotReloadConfigAdvanced]] = {}
        self._lock = threading.RLock()
        self._global_callbacks_lock = threading.Lock()
        
        if auto_discover and config_dir:
            self.discover_configs()
    
    def _wrap_callback(self, name: str, callback: Optional[Callable], is_reload: bool = True):
        def wrapped(config: Dict[str, Any]):
            try:
                if callback:
                    callback(config)
            finally:
                with self._global_callbacks_lock:
                    if is_reload and self.on_any_reload:
                        self.on_any_reload(name, config)
        return wrapped
    
    def _wrap_error_callback(self, name: str, callback: Optional[Callable]):
        def wrapped(error: Exception):
            try:
                if callback:
                    callback(error)
            finally:
                with self._global_callbacks_lock:
                    if self.on_any_error:
                        self.on_any_error(name, error)
        return wrapped
    
    def discover_configs(self) -> List[str]:
        if not self.config_dir or not self.config_dir.exists():
            return []
        
        discovered = []
        pattern = self.discover_pattern
        for config_file in self.config_dir.glob(pattern):
            if config_file.is_file() and config_file.stem not in self._configs:
                name = config_file.stem
                self.add_config(str(config_file), name=name)
                discovered.append(name)
        
        return discovered
    
    def add_config(
        self,
        config_path: str,
        name: Optional[str] = None,
        validator: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_reload: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        use_advanced: bool = True,
        **kwargs
    ) -> Union[HotReloadConfig, HotReloadConfigAdvanced]:
        with self._lock:
            path = Path(config_path)
            config_name = name or path.stem
            
            if config_name in self._configs:
                return self._configs[config_name]
            
            defaults = {
                'reload_interval': self.default_reload_interval,
                'debounce_delay': self.default_debounce_delay,
                'max_history': self.default_max_history,
                'auto_start': False
            }
            defaults.update(kwargs)
            
            wrapped_on_reload = self._wrap_callback(config_name, on_reload, is_reload=True)
            wrapped_on_error = self._wrap_error_callback(config_name, on_error)
            
            if use_advanced:
                config = HotReloadConfigAdvanced(
                    config_path=str(path),
                    on_reload=wrapped_on_reload,
                    on_error=wrapped_on_error,
                    validator=validator,
                    name=config_name,
                    **defaults
                )
            else:
                config = HotReloadConfig(
                    config_path=str(path),
                    on_reload=wrapped_on_reload,
                    on_error=wrapped_on_error,
                    validator=validator,
                    name=config_name,
                    **defaults
                )
            
            self._configs[config_name] = config
            config.start_monitor()
            
            return config
    
    def add_config_from_spec(self, spec: ConfigSpec) -> Union[HotReloadConfig, HotReloadConfigAdvanced]:
        return self.add_config(
            config_path=str(spec.config_path),
            name=spec.name,
            validator=spec.validator,
            on_reload=spec.on_reload,
            on_error=spec.on_error,
            use_advanced=spec.use_advanced,
            **spec.kwargs
        )
    
    def add_configs(self, specs: List[ConfigSpec]) -> Dict[str, Union[HotReloadConfig, HotReloadConfigAdvanced]]:
        result = {}
        for spec in specs:
            result[spec.name] = self.add_config_from_spec(spec)
        return result
    
    def get_config(self, name: str) -> Optional[Union[HotReloadConfig, HotReloadConfigAdvanced]]:
        with self._lock:
            return self._configs.get(name)
    
    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        config = self.get_config(config_name)
        if config is None:
            return default
        return config.get(key, default)
    
    def __getitem__(self, name: str) -> Union[HotReloadConfig, HotReloadConfigAdvanced]:
        config = self.get_config(name)
        if config is None:
            raise KeyError(f"No config found with name: {name}")
        return config
    
    def __getattr__(self, name: str) -> Union[HotReloadConfig, HotReloadConfigAdvanced]:
        if name.startswith('_'):
            raise AttributeError(f"'MultiConfigManager' object has no attribute '{name}'")
        config = self.get_config(name)
        if config is None:
            raise AttributeError(f"No config found with name: {name}")
        return config
    
    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._configs
    
    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._configs.keys()))
    
    def remove_config(self, name: str) -> bool:
        with self._lock:
            if name not in self._configs:
                return False
            config = self._configs.pop(name)
            config.stop_monitor()
            return True
    
    def get_all_configs(self) -> Dict[str, Union[HotReloadConfig, HotReloadConfigAdvanced]]:
        with self._lock:
            return self._configs.copy()
    
    def get_all_values(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {name: cfg.config for name, cfg in self._configs.items()}
    
    def get_versions(self) -> Dict[str, int]:
        with self._lock:
            return {name: cfg.version for name, cfg in self._configs.items()}
    
    def rollback(self, config_name: str, steps: int = 1) -> bool:
        config = self.get_config(config_name)
        if config is None:
            return False
        return config.rollback(steps)
    
    def rollback_all(self, steps: int = 1) -> Dict[str, bool]:
        results = {}
        for name in self:
            results[name] = self.rollback(name, steps)
        return results
    
    def reload_all(self) -> None:
        with self._lock:
            for config in self._configs.values():
                config.reload_now()
    
    def start_all(self) -> None:
        with self._lock:
            for config in self._configs.values():
                config.start_monitor()
    
    def stop_all(self) -> None:
        with self._lock:
            for config in self._configs.values():
                config.stop_monitor()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
        return False
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._configs)
    
    def __repr__(self) -> str:
        with self._lock:
            configs = list(self._configs.keys())
        return f"MultiConfigManager(configs={configs}, count={len(self._configs)})"
