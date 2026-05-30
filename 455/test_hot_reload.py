import os
import json
import time
import tempfile
import unittest
from pathlib import Path

from hot_reload_config import HotReloadConfig
from hot_reload_config_advanced import HotReloadConfigAdvanced, ConfigFormat


class TestHotReloadConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self._write_config({
            "app_name": "TestApp",
            "version": "1.0.0",
            "database": {
                "host": "localhost",
                "port": 5432
            }
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _write_config(self, config: dict):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    
    def test_initial_load(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            self.assertEqual(config.version, 1)
            self.assertEqual(config.app_name, "TestApp")
            self.assertEqual(config['app_name'], "TestApp")
            self.assertEqual(config.get('app_name'), "TestApp")
            self.assertEqual(config.database['host'], "localhost")
    
    def test_get_method(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            self.assertEqual(config.get('app_name'), "TestApp")
            self.assertEqual(config.get('nonexistent', 'default'), "default")
    
    def test_version_increment_on_reload(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            self.assertEqual(config.version, 1)
            
            self._write_config({
                "app_name": "UpdatedApp",
                "version": "2.0.0"
            })
            
            config.reload_now()
            self.assertEqual(config.version, 2)
            self.assertEqual(config.app_name, "UpdatedApp")
    
    def test_config_property_copy(self):
        with HotReloadConfig(self.config_path, auto_start=False) as config:
            cfg = config.config
            cfg['app_name'] = 'Modified'
            
            self.assertEqual(config.app_name, "TestApp")
            self.assertEqual(cfg['app_name'], "Modified")
    
    def test_context_manager(self):
        config = HotReloadConfig(self.config_path, auto_start=True)
        self.assertTrue(config._monitor_thread.is_alive())
        
        config.stop_monitor()
        config._monitor_thread.join(timeout=2)
        self.assertFalse(config._monitor_thread.is_alive())
    
    def test_debounce_prevents_multiple_loads(self):
        reload_count = [0]
        
        def on_reload(cfg):
            reload_count[0] += 1
        
        with HotReloadConfig(
            self.config_path,
            debounce_delay=0.3,
            auto_start=False,
            on_reload=on_reload
        ) as config:
            self.assertEqual(reload_count[0], 0)
            self.assertEqual(config.version, 1)
            
            for i in range(5):
                self._write_config({
                    "app_name": f"App_{i}",
                    "counter": i
                })
                config._check_and_reload()
                time.sleep(0.05)
            
            self.assertEqual(config.version, 1)
            
            time.sleep(0.5)
            
            self.assertEqual(config.version, 2)
            self.assertEqual(reload_count[0], 1)
            self.assertEqual(config.app_name, "App_4")
    
    def test_reload_now_cancels_debounce(self):
        with HotReloadConfig(
            self.config_path,
            debounce_delay=1.0,
            auto_start=False
        ) as config:
            self._write_config({"app_name": "Debounced"})
            config._check_and_reload()
            
            self.assertIsNotNone(config._debounce_timer)
            
            config.reload_now()
            
            self.assertIsNone(config._debounce_timer)
            self.assertEqual(config.version, 2)
            self.assertEqual(config.app_name, "Debounced")


class TestHotReloadConfigAdvanced(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self._write_config({
            "app_name": "TestApp",
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "debug": True
            }
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _write_config(self, config: dict):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    
    def test_dot_notation_access(self):
        with HotReloadConfigAdvanced(self.config_path, auto_start=False) as config:
            self.assertEqual(config.server.host, "0.0.0.0")
            self.assertEqual(config.server.port, 8080)
            self.assertTrue(config.server.debug)
    
    def test_nested_get(self):
        with HotReloadConfigAdvanced(self.config_path, auto_start=False) as config:
            self.assertEqual(config.get('server.host'), "0.0.0.0")
            self.assertEqual(config.get('server.port'), 8080)
            self.assertEqual(config.get('server.nonexistent', 'default'), "default")
            self.assertEqual(config.get('nonexistent.key', 'default'), "default")
    
    def test_on_reload_callback(self):
        reload_count = [0]
        last_config = [None]
        
        def on_reload(cfg):
            reload_count[0] += 1
            last_config[0] = cfg
        
        with HotReloadConfigAdvanced(
            self.config_path,
            auto_start=False,
            on_reload=on_reload
        ) as config:
            self.assertEqual(reload_count[0], 0)
            
            self._write_config({
                "app_name": "UpdatedApp",
                "server": {"port": 9090}
            })
            
            config.reload_now()
            self.assertEqual(reload_count[0], 1)
            self.assertEqual(last_config[0]['app_name'], "UpdatedApp")
    
    def test_attr_dict_methods(self):
        with HotReloadConfigAdvanced(self.config_path, auto_start=False) as config:
            server = config.server
            self.assertEqual(server.get('port'), 8080)
            self.assertEqual(server['port'], 8080)
            self.assertIsInstance(server.to_dict(), dict)
    
    def test_advanced_debounce_merges_changes(self):
        reload_count = [0]
        versions = []
        
        def on_reload(cfg):
            reload_count[0] += 1
            versions.append(cfg.get('version_num', 0))
        
        with HotReloadConfigAdvanced(
            self.config_path,
            debounce_delay=0.3,
            auto_start=False,
            on_reload=on_reload
        ) as config:
            for i in range(1, 6):
                self._write_config({
                    "app_name": f"App_{i}",
                    "version_num": i
                })
                config._check_and_reload()
                time.sleep(0.05)
            
            self.assertEqual(config.version, 1)
            
            time.sleep(0.5)
            
            self.assertEqual(config.version, 2)
            self.assertEqual(reload_count[0], 1)
            self.assertEqual(config.get('version_num'), 5)
            self.assertEqual(config.app_name, "App_5")
    
    def test_advanced_reload_now_cancels_debounce(self):
        with HotReloadConfigAdvanced(
            self.config_path,
            debounce_delay=1.0,
            auto_start=False
        ) as config:
            self._write_config({"app_name": "PendingChange"})
            config._check_and_reload()
            
            self.assertIsNotNone(config._debounce_timer)
            
            config.reload_now()
            
            self.assertIsNone(config._debounce_timer)
            self.assertEqual(config.version, 2)
            self.assertEqual(config.app_name, "PendingChange")


def run_demo():
    print("=== 配置热加载防抖机制演示 ===")
    print()
    
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'demo_config.json')
    
    initial_config = {
        "app_name": "防抖演示",
        "version": "1.0",
        "change_count": 0
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(initial_config, f, ensure_ascii=False, indent=2)
    
    print(f"配置文件: {config_path}")
    print()
    
    reload_count = [0]
    
    def on_reload(cfg):
        reload_count[0] += 1
        print(f"[回调 #{reload_count[0]}] 配置已更新: "
              f"change_count={cfg.get('change_count')}, "
              f"config.version={cfg.get('version')}")
    
    config = HotReloadConfigAdvanced(
        config_path,
        reload_interval=0.05,
        debounce_delay=0.3,
        on_reload=on_reload,
        auto_start=False
    )
    try:
        print(f"初始版本: v{config.version}")
        print(f"应用名称: {config.app_name}")
        print()
        
        print("连续快速修改配置文件5次...")
        for i in range(1, 6):
            updated_config = {
                "app_name": "防抖演示",
                "version": f"1.{i}",
                "change_count": i
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(updated_config, f, ensure_ascii=False, indent=2)
            config._check_and_reload()
            time.sleep(0.05)
            print(f"  已修改第 {i} 次")
        
        print()
        print(f"修改完成后立即检查: version={config.version}")
        print("等待防抖延迟（300ms）...")
        time.sleep(0.5)
        
        print()
        print(f"防抖延迟后: version={config.version}")
        print(f"最终 change_count: {config.change_count}")
        print(f"最终 version: {config.version}")
        print(f"实际触发重载次数: {reload_count[0]}")
        print()
        
        if reload_count[0] == 1 and config.version == 2:
            print("✓ 防抖机制成功: 5次修改合并为1次重载!")
        else:
            print(f"✗ 防抖机制可能有问题: reload_count={reload_count[0]}, config.version={config.version}")
    finally:
        config.stop_monitor()
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print()
    print("=== 演示完成 ===")


if __name__ == '__main__':
    run_demo()
    print()
    print("运行单元测试...")
    unittest.main(verbosity=2)
