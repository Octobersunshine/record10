import json
import argparse
import sys
import re


def check_wildcard_origin(origins):
    has_wildcard = '*' in origins
    if has_wildcard:
        print("WARNING: 检测到允许所有源（*），浏览器安全策略禁止在允许所有源时携带Cookie凭证。", file=sys.stderr)
        print("         已自动将 allow_credentials 设置为 false。", file=sys.stderr)
    return has_wildcard


def audit_cors_config(origins, methods, headers, allow_credentials, max_age):
    audit_result = {
        'issues': [],
        'warnings': [],
        'infos': [],
        'security_score': 100
    }
    
    if '*' in origins:
        audit_result['issues'].append({
            'level': 'HIGH',
            'type': 'WILDCARD_ORIGIN',
            'message': '使用通配符 * 允许所有源，存在安全风险',
            'recommendation': '建议指定具体的允许源域名'
        })
        audit_result['security_score'] -= 30
    
    if allow_credentials and '*' in origins:
        audit_result['issues'].append({
            'level': 'CRITICAL',
            'type': 'CREDENTIALS_WITH_WILDCARD',
            'message': '允许所有源同时携带凭证，这违反浏览器安全策略',
            'recommendation': '必须禁用凭证或指定具体源'
        })
        audit_result['security_score'] -= 50
    
    if 'DELETE' in methods or 'PUT' in methods or 'PATCH' in methods:
        audit_result['warnings'].append({
            'level': 'MEDIUM',
            'type': 'DANGEROUS_METHODS',
            'message': f'允许修改性HTTP方法: {", ".join([m for m in methods if m in ["DELETE", "PUT", "PATCH"]])}',
            'recommendation': '仅允许必要的HTTP方法'
        })
        audit_result['security_score'] -= 10
    
    if '*' in headers:
        audit_result['warnings'].append({
            'level': 'MEDIUM',
            'type': 'WILDCARD_HEADERS',
            'message': '允许所有请求头',
            'recommendation': '指定具体需要的请求头'
        })
        audit_result['security_score'] -= 15
    
    if max_age > 86400:
        audit_result['infos'].append({
            'level': 'LOW',
            'type': 'LONG_MAX_AGE',
            'message': f'预检缓存时间过长 ({max_age}秒)',
            'recommendation': '建议设置为 86400 秒 (24小时) 或更短'
        })
    
    if max_age < 3600:
        audit_result['infos'].append({
            'level': 'LOW',
            'type': 'SHORT_MAX_AGE',
            'message': f'预检缓存时间较短 ({max_age}秒)',
            'recommendation': '可适当增加以减少预检请求数量'
        })
    
    if allow_credentials and '*' not in origins:
        audit_result['infos'].append({
            'level': 'INFO',
            'type': 'CREDENTIALS_ENABLED',
            'message': '已启用凭证携带',
            'recommendation': '确保仅在需要时启用'
        })
    
    has_http = any(o.startswith('http://') for o in origins if o != '*')
    if has_http:
        audit_result['warnings'].append({
            'level': 'MEDIUM',
            'type': 'HTTP_ORIGIN',
            'message': '包含HTTP协议的源，建议使用HTTPS',
            'recommendation': '将源升级为 HTTPS 协议'
        })
        audit_result['security_score'] -= 10
    
    audit_result['security_score'] = max(0, audit_result['security_score'])
    return audit_result


def format_audit_report(audit_result):
    report = []
    report.append("=" * 70)
    report.append("CORS 配置安全审计报告")
    report.append("=" * 70)
    
    report.append(f"\n安全评分: {audit_result['security_score']}/100")
    
    if audit_result['security_score'] >= 80:
        report.append("安全等级: 安全 ✓")
    elif audit_result['security_score'] >= 60:
        report.append("安全等级: 中等风险 ⚠")
    else:
        report.append("安全等级: 高风险 ✗")
    
    if audit_result['issues']:
        report.append("\n" + "-" * 70)
        report.append("问题 (需要修复):")
        report.append("-" * 70)
        for issue in audit_result['issues']:
            report.append(f"\n[{issue['level']}] {issue['type']}")
            report.append(f"  描述: {issue['message']}")
            report.append(f"  建议: {issue['recommendation']}")
    
    if audit_result['warnings']:
        report.append("\n" + "-" * 70)
        report.append("警告 (建议关注):")
        report.append("-" * 70)
        for warning in audit_result['warnings']:
            report.append(f"\n[{warning['level']}] {warning['type']}")
            report.append(f"  描述: {warning['message']}")
            report.append(f"  建议: {warning['recommendation']}")
    
    if audit_result['infos']:
        report.append("\n" + "-" * 70)
        report.append("信息:")
        report.append("-" * 70)
        for info in audit_result['infos']:
            report.append(f"\n[{info['level']}] {info['type']}")
            report.append(f"  描述: {info['message']}")
            report.append(f"  建议: {info['recommendation']}")
    
    report.append("\n" + "=" * 70)
    return "\n".join(report)


def generate_nginx_config(origins, methods, headers, max_age=86400, allow_credentials=True, dynamic_cors=False):
    origins_str = "|".join(origins) if len(origins) > 1 else origins[0]
    methods_str = " ".join(methods)
    headers_str = ", ".join(headers)
    
    credentials_line = "    add_header 'Access-Control-Allow-Credentials' 'true' always;" if allow_credentials else ""
    
    if dynamic_cors and '*' not in origins:
        origin_check = f'''
    map $http_origin $allowed_origin {{
        default "";
'''
        for origin in origins:
            origin_check += f'        "{origin}" "{origin}";\n'
        origin_check += '    }'
        
        origin_value = '$allowed_origin'
    else:
        origin_check = ''
        origin_value = '"$http_origin"' if '*' not in origins else '"*"'
    
    config = f'''# Nginx CORS Configuration
# 预检请求缓存时间: {max_age} 秒 ({max_age // 3600} 小时)
# 动态CORS: {'启用' if dynamic_cors else '禁用'}
{origin_check}
location / {{
    if ($request_method = 'OPTIONS') {{
        add_header 'Access-Control-Allow-Origin' {origin_value} always;
        add_header 'Access-Control-Allow-Methods' '{methods_str}' always;
        add_header 'Access-Control-Allow-Headers' '{headers_str}' always;
        add_header 'Access-Control-Max-Age' {max_age} always;
{credentials_line}
        return 204;
    }}

    if ($request_method = 'GET') {{
        add_header 'Access-Control-Allow-Origin' {origin_value} always;
        add_header 'Access-Control-Allow-Methods' '{methods_str}' always;
        add_header 'Access-Control-Allow-Headers' '{headers_str}' always;
{credentials_line}
    }}

    if ($request_method = 'POST') {{
        add_header 'Access-Control-Allow-Origin' {origin_value} always;
        add_header 'Access-Control-Allow-Methods' '{methods_str}' always;
        add_header 'Access-Control-Allow-Headers' '{headers_str}' always;
{credentials_line}
    }}

    if ($http_origin ~* "^({origins_str})$") {{
        set $cors "true";
    }}
}}'''
    return config


def generate_apache_config(origins, methods, headers, max_age=86400, allow_credentials=True, dynamic_cors=False):
    origins_str = "|".join(origins) if len(origins) > 1 else origins[0]
    methods_str = ", ".join(methods)
    headers_str = ", ".join(headers)
    
    credentials_line = "    Header always set Access-Control-Allow-Credentials \"true\"" if allow_credentials else ""
    
    if dynamic_cors and '*' not in origins:
        origin_env = 'SetEnvIf Origin "^(' + origins_str + ')$" AccessControlAllowOrigin=$0'
        origin_header = '%{AccessControlAllowOrigin}e env=AccessControlAllowOrigin'
    else:
        origin_env = ''
        origin_header = '"*"' if '*' in origins else '"%{HTTP_ORIGIN}e"'
    
    config = f'''# Apache CORS Configuration
# 预检请求缓存时间: {max_age} 秒 ({max_age // 3600} 小时)
# 动态CORS: {'启用' if dynamic_cors else '禁用'}
<IfModule mod_headers.c>
    {origin_env}
    Header always set Access-Control-Allow-Origin {origin_header}
    Header always set Access-Control-Allow-Methods "{methods_str}"
    Header always set Access-Control-Allow-Headers "{headers_str}"
    Header always set Access-Control-Max-Age "{max_age}"
{credentials_line}

    <IfModule mod_rewrite.c>
        RewriteEngine On
        RewriteCond %{{REQUEST_METHOD}} OPTIONS
        RewriteRule ^(.*)$ $1 [R=204,L]
    </IfModule>
</IfModule>'''
    return config


def generate_spring_config(origins, methods, headers, max_age=86400, allow_credentials=True, dynamic_cors=False):
    methods_str = ", ".join([f'RequestMethod.{m.upper()}' for m in methods])
    headers_str = ", ".join([f'"{h}"' for h in headers])
    credentials_value = "true" if allow_credentials else "false"
    
    if dynamic_cors and '*' not in origins:
        origins_code = '''
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOrigins(Arrays.asList(''' + ", ".join([f'"{o}"' for o in origins]) + '''));
        configuration.setAllowedMethods(Arrays.asList(''' + ", ".join([f'"{m}"' for m in methods]) + '''));
        configuration.setAllowedHeaders(Arrays.asList(''' + headers_str + '''));
        configuration.setMaxAge(Duration.ofSeconds(''' + str(max_age) + '''));
        configuration.setAllowCredentials(''' + credentials_value + ''');
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }'''
        config = f'''// Spring Dynamic CORS Configuration (Java)
// 预检请求缓存时间: {max_age} 秒 ({max_age // 3600} 小时)
// 动态CORS: 启用 - 使用 CorsConfigurationSource
@Configuration
public class CorsConfig {{
{origins_code}
}}'''
    else:
        origins_str = ", ".join([f'"{o}"' for o in origins])
        config = f'''// Spring CORS Configuration (Java)
// 预检请求缓存时间: {max_age} 秒 ({max_age // 3600} 小时)
// 动态CORS: 禁用
@Configuration
public class CorsConfig implements WebMvcConfigurer {{

    @Override
    public void addCorsMappings(CorsRegistry registry) {{
        registry.addMapping("/**")
            .allowedOrigins({origins_str})
            .allowedMethods({methods_str})
            .allowedHeaders({headers_str})
            .maxAge({max_age})
            .allowCredentials({credentials_value});
    }}
}}'''
    return config


def generate_spring_boot_config(origins, methods, headers, max_age=86400, allow_credentials=True, dynamic_cors=False):
    credentials_value = "true" if allow_credentials else "false"
    dynamic_note = "# 动态CORS: 请使用 CorsConfigurationSource Bean 实现动态配置" if dynamic_cors else ""
    
    config = f'''# Spring Boot application.properties CORS Configuration
# 预检请求缓存时间: {max_age} 秒 ({max_age // 3600} 小时)
# 动态CORS: {'需要编程式配置' if dynamic_cors else '标准配置'}
{dynamic_note}
spring.web.cors.allowed-origins={",".join(origins)}
spring.web.cors.allowed-methods={",".join(methods)}
spring.web.cors.allowed-headers={",".join(headers)}
spring.web.cors.max-age={max_age}
spring.web.cors.allow-credentials={credentials_value}'''
    return config


def generate_all_configs(config_json, dynamic_cors=False):
    config = json.loads(config_json)
    
    origins = config.get('origins', ['*'])
    methods = config.get('methods', ['GET', 'POST'])
    headers = config.get('headers', ['Content-Type', 'Authorization'])
    max_age = config.get('max_age', 86400)
    
    has_wildcard = check_wildcard_origin(origins)
    allow_credentials = not has_wildcard
    
    configs = {
        'nginx': generate_nginx_config(origins, methods, headers, max_age, allow_credentials, dynamic_cors),
        'apache': generate_apache_config(origins, methods, headers, max_age, allow_credentials, dynamic_cors),
        'spring': generate_spring_config(origins, methods, headers, max_age, allow_credentials, dynamic_cors),
        'spring_boot': generate_spring_boot_config(origins, methods, headers, max_age, allow_credentials, dynamic_cors)
    }
    
    audit_result = audit_cors_config(origins, methods, headers, allow_credentials, max_age)
    configs['audit'] = audit_result
    configs['audit_report'] = format_audit_report(audit_result)
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='CORS Configuration Generator')
    parser.add_argument('--origins', type=str, nargs='+', default=['*'],
                        help='Allowed origins (e.g., http://localhost:3000)')
    parser.add_argument('--methods', type=str, nargs='+', default=['GET', 'POST'],
                        help='Allowed HTTP methods')
    parser.add_argument('--headers', type=str, nargs='+', 
                        default=['Content-Type', 'Authorization'],
                        help='Allowed request headers')
    parser.add_argument('--max-age', type=int, default=86400,
                        help='Max age for preflight cache (seconds)')
    parser.add_argument('--dynamic-cors', action='store_true',
                        help='Enable dynamic CORS based on request Origin')
    parser.add_argument('--audit', action='store_true',
                        help='Run security audit only')
    parser.add_argument('--format', type=str, default='all',
                        choices=['nginx', 'apache', 'spring', 'spring_boot', 'all', 'audit'],
                        help='Output format')
    
    args = parser.parse_args()
    
    config_json = json.dumps({
        'origins': args.origins,
        'methods': args.methods,
        'headers': args.headers,
        'max_age': args.max_age
    })
    
    configs = generate_all_configs(config_json, args.dynamic_cors)
    
    if args.audit or args.format == 'audit':
        print(configs['audit_report'])
    elif args.format == 'all':
        print(configs['audit_report'])
        for name, config in configs.items():
            if name not in ['audit', 'audit_report']:
                print(f"\n{'='*60}")
                print(f"{name.upper()} CORS CONFIGURATION")
                print('='*60)
                print(config)
                print()
    else:
        print(configs[args.format])


if __name__ == '__main__':
    main()
