import time
from hot_reload_config import HotReloadConfig


def on_config_reload(new_config: dict):
    print(f"\n=== 配置已重新加载 ===")
    print(f"新配置内容: {new_config}")
    print(f"=====================\n")


def main():
    print("配置文件热加载演示")
    print("修改 config.json 文件来观察热加载效果\n")
    
    with HotReloadConfig(
        config_path="config.json",
        reload_interval=1.0,
        on_reload=on_config_reload
    ) as config:
        print(f"初始配置版本: {config.version}")
        print(f"应用名称: {config.app_name}")
        print(f"服务器端口: {config['server']['port']}")
        print(f"调试模式: {config.get('server', {}).get('debug', False)}")
        
        print("\n程序运行中... 按 Ctrl+C 退出")
        print("定期显示当前配置信息:\n")
        
        try:
            while True:
                print(f"[v{config.version}] 应用: {config.app_name} | "
                      f"端口: {config.server.port} | "
                      f"日志级别: {config.logging.level}")
                time.sleep(3)
        except KeyboardInterrupt:
            print("\n程序退出")


if __name__ == "__main__":
    main()
