from cors_config_generator import (
    generate_all_configs, 
    generate_nginx_config, 
    generate_apache_config, 
    generate_spring_config, 
    generate_spring_boot_config, 
    check_wildcard_origin,
    audit_cors_config,
    format_audit_report
)
import json


def main():
    print("=" * 70)
    print("CORS CONFIGURATION GENERATOR")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("示例1: 指定具体源 + 动态CORS + 安全审计")
    print("=" * 70)
    
    cors_config1 = {
        "origins": [
            "https://app.example.com",
            "https://admin.example.com"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "headers": ["Content-Type", "Authorization"],
        "max_age": 86400
    }
    
    print(f"\nInput JSON Configuration:")
    print(json.dumps(cors_config1, indent=2))

    config_json1 = json.dumps(cors_config1)
    configs1 = generate_all_configs(config_json1, dynamic_cors=True)

    print(f"\n安全审计报告:")
    print(configs1['audit_report'])

    print(f"\nNginx 动态CORS配置:")
    print(configs1['nginx'])

    print("\n" + "=" * 70)
    print("示例2: 安全审计 - 高风险配置")
    print("=" * 70)
    
    cors_config2 = {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "headers": ["*"],
        "max_age": 604800
    }
    
    print(f"\nInput JSON Configuration:")
    print(json.dumps(cors_config2, indent=2))

    config_json2 = json.dumps(cors_config2)
    configs2 = generate_all_configs(config_json2)

    print(f"\n安全审计报告:")
    print(configs2['audit_report'])

    print("\n" + "=" * 70)
    print("示例3: 自定义预检缓存时间")
    print("=" * 70)
    
    cors_config3 = {
        "origins": ["https://example.com"],
        "methods": ["GET", "POST"],
        "headers": ["Content-Type"],
        "max_age": 3600
    }
    
    print(f"\n预检缓存时间: {cors_config3['max_age']} 秒 (1小时)")
    config_json3 = json.dumps(cors_config3)
    configs3 = generate_all_configs(config_json3)
    print(configs3['apache'])

    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    
    print("\n1. 生成动态CORS配置:")
    print('''
   config_json = json.dumps({
       "origins": ["https://app.example.com"],
       "methods": ["GET", "POST"],
       "headers": ["Content-Type"],
       "max_age": 86400
   })
   configs = generate_all_configs(config_json, dynamic_cors=True)
''')

    print("\n2. 单独执行安全审计:")
    print('''
   audit_result = audit_cors_config(
       origins=["*"],
       methods=["GET", "POST", "DELETE"],
       headers=["Content-Type"],
       allow_credentials=False,
       max_age=86400
   )
   print(format_audit_report(audit_result))
''')

    print("\n3. 命令行使用示例:")
    print('''
   # 生成所有配置 + 动态CORS + 审计
   python cors_config_generator.py --origins https://example.com \\
       --methods GET POST --dynamic-cors
   
   # 仅执行安全审计
   python cors_config_generator.py --origins "*" --format audit
   
   # 自定义预检缓存时间
   python cors_config_generator.py --max-age 3600
''')


if __name__ == '__main__':
    main()
