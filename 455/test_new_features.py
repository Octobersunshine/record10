import os
import json
import time
import tempfile
import unittest
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from hot_reload_config import HotReloadConfig, ConfigValidationError
from hot_reload_config_advanced import HotReloadConfigAdvanced
from multi_config_manager import MultiConfigManager, ConfigSpec


class TestConfigRollback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self._write_config({
            "app_name": "TestApp",
            "version": "1.0.0",
            "port": 8080
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _write_config(self, config: dict):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    
    def test_rollback_single_step(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            self.assertEqual(config.version, 1)
            self.assertEqual(config.port, 8080)
            
            self._write_config({
                "app_name": "TestApp",
                "version": "1.0.0",
                "port": 9090
            })
            config.reload_now()
            
            self.assertEqual(config.version, 2)
            self.assertEqual(config.port, 9090)
            
            self._write_config({
                "app_name": "TestApp",
                "version": "1.0.0",
                "port": 10000
            })
            config.reload_now()
            
            self.assertEqual(config.version, 3)
            self.assertEqual(config.port, 10000)
            
            self.assertTrue(config.can_rollback(1))
            result = config.rollback(1)
            
            self.assertTrue(result)
            self.assertEqual(config.version, 4)
            self.assertEqual(config.port, 9090)
    
    def test_rollback_multiple_steps(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            for i in range(1, 5):
                self._write_config({
                    "app_name": f"App_{i}",
                    "version": f"1.{i}",
                    "step": i
                })
                config.reload_now()
            
            self.assertEqual(config.step, 4)
            
            self.assertTrue(config.can_rollback(3))
            result = config.rollback(3)
            
            self.assertTrue(result)
            self.assertEqual(config.step, 1)
    
    def test_rollback_no_history(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            self.assertFalse(config.can_rollback(1))
            result = config.rollback(1)
            self.assertFalse(result)
    
    def test_rollback_invalid_steps(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            self._write_config({"app_name": "Updated"})
            config.reload_now()
            
            self.assertFalse(config.rollback(0))
            self.assertFalse(config.rollback(-1))
            self.assertFalse(config.rollback(100))
    
    def test_get_history(self):
        with HotReloadConfig(self.config_path, auto_start=False, max_history=5) as config:
            for i in range(1, 4):
                self._write_config({"version": i})
                config.reload_now()
            
            history = config.get_history()
            self.assertEqual(len(history), 3)
            self.assertEqual(history[0]['version'], 1)
            self.assertEqual(history[1]['version'], 2)
            self.assertEqual(history[2]['version'], 3)
    
    def test_history_limit(self):
        with HotReloadConfig(self.config_path, auto_start=False, max_history=3) as config:
            for i in range(1, 8):
                self._write_config({"version": i})
                config.reload_now()
            
            history = config.get_history()
            self.assertEqual(len(history), 3)
            self.assertEqual(history[0]['config']['version'], 4)
            self.assertEqual(history[-1]['config']['version'], 6)


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self._write_config({
            "app_name": "TestApp",
            "port": 8080,
            "debug": True
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _write_config(self, config: dict):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    
    def test_validator_keeps_old_config_on_failure(self):
        def validator(cfg):
            if 'port' not in cfg or not isinstance(cfg['port'], int):
                raise ValueError("port must be an integer")
            if cfg['port'] < 1 or cfg['port'] > 65535:
                raise ValueError("port must be between 1 and 65535")
        
        errors = []
        
        with HotReloadConfig(
            self.config_path,
            auto_start=False,
            validator=validator,
            on_error=lambda e: errors.append(e)
        ) as config:
            self.assertEqual(config.port, 8080)
            self.assertEqual(len(errors), 0)
            
            self._write_config({
                "app_name": "TestApp",
                "port": "invalid",
                "debug": True
            })
            config.reload_now()
            
            self.assertEqual(config.port, 8080)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ConfigValidationError)
    
    def test_validator_passes_valid_config(self):
        def validator(cfg):
            if 'app_name' not in cfg:
                raise ValueError("app_name is required")
        
        reload_count = [0]
        
        with HotReloadConfig(
            self.config_path,
            auto_start=False,
            validator=validator,
            on_reload=lambda cfg: reload_count.__setitem__(0, reload_count[0] + 1)
        ) as config:
            self._write_config({
                "app_name": "ValidApp",
                "port": 9090
            })
            config.reload_now()
            
            self.assertEqual(config.app_name, "ValidApp")
            self.assertEqual(reload_count[0], 1)
    
    def test_invalid_json_keeps_old_config(self):
        errors = []
        
        with HotReloadConfig(
            self.config_path,
            auto_start=False,
            on_error=lambda e: errors.append(e)
        ) as config:
            self.assertEqual(config.app_name, "TestApp")
            
            with open(self.config_path, 'w') as f:
                f.write("{ invalid json }")
            
            config.reload_now()
            
            self.assertEqual(config.app_name, "TestApp")
            self.assertEqual(len(errors), 1)
    
    def test_validator_with_advanced_config(self):
        def validator(cfg):
            if not isinstance(cfg.get('server', {}).get('port'), int):
                raise ValueError("server.port must be integer")
        
        self._write_config({
            "server": {
                "host": "localhost",
                "port": 8080
            }
        })
        
        with HotReloadConfigAdvanced(
            self.config_path,
            auto_start=False,
            validator=validator
        ) as config:
            self.assertEqual(config.server.port, 8080)
            
            self._write_config({
                "server": {
                    "host": "localhost",
                    "port": "invalid"
                }
            })
            config.reload_now()
            
            self.assertEqual(config.server.port, 8080)
    
    def test_missing_required_fields(self):
        def validator(cfg):
            required_fields = ['app_name', 'version']
            for field in required_fields:
                if field not in cfg:
                    raise ValueError(f"Missing required field: {field}")
        
        self._write_config({
            "app_name": "TestApp",
            "version": "1.0.0",
            "port": 8080
        })
        
        errors = []
        
        with HotReloadConfig(
            self.config_path,
            auto_start=False,
            validator=validator,
            on_error=lambda e: errors.append(e)
        ) as config:
            self.assertEqual(config.get('app_name'), "TestApp")
            
            self._write_config({"app_name": "MissingVersion"})
            config.reload_now()
            
            self.assertEqual(config.get('app_name'), "TestApp")
            self.assertEqual(len(errors), 1)


class TestMultiConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        self.app_config_path = os.path.join(self.temp_dir, 'app.json')
        self.db_config_path = os.path.join(self.temp_dir, 'db.json')
        
        with open(self.app_config_path, 'w') as f:
            json.dump({"app_name": "MyApp", "port": 8080}, f)
        
        with open(self.db_config_path, 'w') as f:
            json.dump({"host": "localhost", "port": 5432, "database": "mydb"}, f)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_multiple_configs(self):
        with MultiConfigManager() as manager:
            app_config = manager.add_config(self.app_config_path, name='app')
            db_config = manager.add_config(self.db_config_path, name='db')
            
            self.assertIn('app', manager)
            self.assertIn('db', manager)
            self.assertEqual(len(manager), 2)
            
            self.assertEqual(app_config.app_name, "MyApp")
            self.assertEqual(db_config.host, "localhost")
    
    def test_access_configs_multiple_ways(self):
        with MultiConfigManager() as manager:
            manager.add_config(self.app_config_path, name='app')
            manager.add_config(self.db_config_path, name='db')
            
            self.assertEqual(manager['app'].port, 8080)
            self.assertEqual(manager.app.port, 8080)
            self.assertEqual(manager.get('db', 'port'), 5432)
            self.assertEqual(manager.get('db', 'nonexistent', 'default'), 'default')
    
    def test_add_config_from_spec(self):
        def db_validator(cfg):
            if 'host' not in cfg:
                raise ValueError("host is required")
        
        with MultiConfigManager() as manager:
            specs = [
                ConfigSpec(
                    config_path=self.app_config_path,
                    name='app'
                ),
                ConfigSpec(
                    config_path=self.db_config_path,
                    name='db',
                    validator=db_validator
                )
            ]
            
            configs = manager.add_configs(specs)
            
            self.assertEqual(len(configs), 2)
            self.assertIn('app', configs)
            self.assertIn('db', configs)
    
    def test_remove_config(self):
        with MultiConfigManager() as manager:
            manager.add_config(self.app_config_path, name='app')
            manager.add_config(self.db_config_path, name='db')
            
            self.assertTrue(manager.remove_config('app'))
            self.assertNotIn('app', manager)
            self.assertIn('db', manager)
            self.assertEqual(len(manager), 1)
            
            self.assertFalse(manager.remove_config('nonexistent'))
    
    def test_iterate_configs(self):
        with MultiConfigManager() as manager:
            manager.add_config(self.app_config_path, name='app')
            manager.add_config(self.db_config_path, name='db')
            
            names = [name for name in manager]
            self.assertEqual(sorted(names), ['app', 'db'])
    
    def test_get_all_values(self):
        with MultiConfigManager() as manager:
            manager.add_config(self.app_config_path, name='app')
            manager.add_config(self.db_config_path, name='db')
            
            all_values = manager.get_all_values()
            self.assertIn('app', all_values)
            self.assertIn('db', all_values)
            self.assertEqual(all_values['app']['port'], 8080)
            self.assertEqual(all_values['db']['database'], 'mydb')
    
    def test_get_versions(self):
        with MultiConfigManager() as manager:
            app_cfg = manager.add_config(self.app_config_path, name='app')
            db_cfg = manager.add_config(self.db_config_path, name='db')
            
            versions = manager.get_versions()
            self.assertEqual(versions['app'], 1)
            self.assertEqual(versions['db'], 1)
    
    def test_rollback_specific_config(self):
        with MultiConfigManager() as manager:
            app_cfg = manager.add_config(self.app_config_path, name='app', auto_start=False)
            manager.add_config(self.db_config_path, name='db', auto_start=False)
            
            with open(self.app_config_path, 'w') as f:
                json.dump({"app_name": "UpdatedApp", "port": 9090}, f)
            app_cfg.reload_now()
            
            self.assertEqual(manager.app.port, 9090)
            
            result = manager.rollback('app', 1)
            self.assertTrue(result)
            self.assertEqual(manager.app.port, 8080)
    
    def test_rollback_all(self):
        with MultiConfigManager() as manager:
            app_cfg = manager.add_config(self.app_config_path, name='app', auto_start=False)
            db_cfg = manager.add_config(self.db_config_path, name='db', auto_start=False)
            
            with open(self.app_config_path, 'w') as f:
                json.dump({"app_name": "UpdatedApp", "port": 9090}, f)
            app_cfg.reload_now()
            
            with open(self.db_config_path, 'w') as f:
                json.dump({"host": "remotehost", "port": 5432, "database": "mydb"}, f)
            db_cfg.reload_now()
            
            results = manager.rollback_all(1)
            self.assertEqual(results['app'], True)
            self.assertEqual(results['db'], True)
            
            self.assertEqual(manager.app.port, 8080)
            self.assertEqual(manager.db.host, "localhost")
    
    def test_global_callbacks(self):
        reloads = []
        errors = []
        
        def on_any_reload(name, config):
            reloads.append((name, config))
        
        def on_any_error(name, error):
            errors.append((name, error))
        
        with MultiConfigManager(
            on_any_reload=on_any_reload,
            on_any_error=on_any_error
        ) as manager:
            app_cfg = manager.add_config(self.app_config_path, name='app', auto_start=False)
            
            with open(self.app_config_path, 'w') as f:
                json.dump({"app_name": "UpdatedApp", "port": 9090}, f)
            app_cfg.reload_now()
            
            self.assertEqual(len(reloads), 1)
            self.assertEqual(reloads[0][0], 'app')
    
    def test_auto_discover(self):
        with MultiConfigManager(
            config_dir=self.temp_dir,
            auto_discover=True,
            discover_pattern='*.json'
        ) as manager:
            self.assertGreaterEqual(len(manager), 2)
            self.assertIn('app', manager)
            self.assertIn('db', manager)
    
    def test_reload_all(self):
        with MultiConfigManager() as manager:
            manager.add_config(self.app_config_path, name='app', auto_start=False)
            manager.add_config(self.db_config_path, name='db', auto_start=False)
            
            initial_versions = manager.get_versions()
            
            with open(self.app_config_path, 'w') as f:
                json.dump({"app_name": "UpdatedApp", "port": 9090}, f)
            
            with open(self.db_config_path, 'w') as f:
                json.dump({"host": "remotehost", "port": 5432, "database": "mydb"}, f)
            
            manager.reload_all()
            
            new_versions = manager.get_versions()
            self.assertEqual(new_versions['app'], initial_versions['app'] + 1)
            self.assertEqual(new_versions['db'], initial_versions['db'] + 1)
    
    def test_context_manager(self):
        manager = MultiConfigManager()
        app_cfg = manager.add_config(self.app_config_path, name='app')
        
        self.assertTrue(app_cfg._monitor_thread.is_alive())
        
        with manager:
            pass
        
        self.assertFalse(app_cfg._monitor_thread.is_alive())


def run_demo():
    print("=== 新功能综合演示 ===")
    print()
    
    temp_dir = tempfile.mkdtemp()
    
    app_config_path = os.path.join(temp_dir, 'app.json')
    db_config_path = os.path.join(temp_dir, 'db.json')
    
    with open(app_config_path, 'w', encoding='utf-8') as f:
        json.dump({"app_name": "MyApp", "port": 8080, "debug": True}, f, ensure_ascii=False)
    
    with open(db_config_path, 'w', encoding='utf-8') as f:
        json.dump({"host": "localhost", "port": 5432, "database": "prod_db"}, f)
    
    print("1. 配置校验功能演示")
    print("-" * 40)
    
    def app_validator(cfg):
        if 'port' not in cfg or not isinstance(cfg['port'], int):
            raise ValueError("port 必须是整数")
        if cfg['port'] < 1 or cfg['port'] > 65535:
            raise ValueError("port 必须在 1-65535 之间")
        if 'app_name' not in cfg or not cfg['app_name']:
            raise ValueError("app_name 不能为空")
    
    validation_errors = []
    
    with HotReloadConfig(
        app_config_path,
        auto_start=False,
        validator=app_validator,
        on_error=lambda e: validation_errors.append(e)
    ) as config:
        print(f"Initial config: port={config.port}, app_name={config.app_name}")
        print(f"Initial version: v{config.version}")
        
        print("\nWriting invalid config (port=99999)...")
        with open(app_config_path, 'w', encoding='utf-8') as f:
            json.dump({"app_name": "InvalidConfig", "port": 99999}, f, ensure_ascii=False)
        config.reload_now()
        
        print(f"After validation: port={config.port} (kept old value)")
        print(f"Validation errors: {len(validation_errors)}")
        print(f"Current version: v{config.version} (not incremented)")
    
    print()
    print("2. Config Rollback Demo")
    print("-" * 40)
    
    with open(app_config_path, 'w', encoding='utf-8') as f:
        json.dump({"app_name": "Version1", "port": 8080}, f, ensure_ascii=False)
    
    with HotReloadConfig(app_config_path, auto_start=False, max_history=10) as config:
        for i in range(2, 6):
            with open(app_config_path, 'w', encoding='utf-8') as f:
                json.dump({"app_name": f"Version{i}", "port": 8080 + i}, f, ensure_ascii=False)
            config.reload_now()
            print(f"  Updated to version {i}: app_name={config.app_name}, port={config.port}, config.version=v{config.version}")
        
        print(f"\nHistory count: {len(config.get_history())}")
        print(f"Can rollback 3 steps: {config.can_rollback(3)}")
        
        print("\nRolling back 3 steps...")
        config.rollback(3)
        print(f"After rollback: app_name={config.app_name}, port={config.port}, config.version=v{config.version}")
    
    print()
    print("3. Multi-Config Manager Demo")
    print("-" * 40)
    
    def db_validator(cfg):
        if 'host' not in cfg:
            raise ValueError("host field is required")
    
    with MultiConfigManager(
        on_any_reload=lambda name, cfg: print(f"  [Global Callback] {name} config updated"),
        on_any_error=lambda name, err: print(f"  [Global Error] {name}: {err}")
    ) as manager:
        manager.add_config(
            app_config_path,
            name='app',
            validator=app_validator,
            auto_start=False
        )
        manager.add_config(
            db_config_path,
            name='db',
            validator=db_validator,
            auto_start=False
        )
        
        print(f"Loaded configs: {list(manager)}")
        print(f"Config count: {len(manager)}")
        print()
        print(f"app.port = {manager.app.port}")
        print(f"db.host = {manager.db.host}")
        print(f"manager.get('db', 'database') = {manager.get('db', 'database')}")
        print()
        print("All versions:", manager.get_versions())
        
        print("\nUpdating app config...")
        with open(app_config_path, 'w', encoding='utf-8') as f:
            json.dump({"app_name": "UpdatedApp", "port": 9090}, f, ensure_ascii=False)
        manager.app.reload_now()
        
        print("\nUpdating db config...")
        with open(db_config_path, 'w', encoding='utf-8') as f:
            json.dump({"host": "192.168.1.100", "port": 5432, "database": "prod_db"}, f, ensure_ascii=False)
        manager.db.reload_now()
        
        print()
        print("Versions after update:", manager.get_versions())
        
        print("\nRolling back all configs...")
        rollback_results = manager.rollback_all(1)
        print(f"Rollback results: {rollback_results}")
        print(f"After rollback app.port = {manager.app.port}")
        print(f"After rollback db.host = {manager.db.host}")
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print()
    print("=== 演示完成 ===")


if __name__ == '__main__':
    run_demo()
    print()
    print("=" * 60)
    print("运行单元测试...")
    print("=" * 60)
    unittest.main(verbosity=2)
