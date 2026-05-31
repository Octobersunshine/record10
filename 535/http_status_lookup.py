HTTP_STATUS_CODES = {
    100: {
        "meaning": "Continue",
        "category": "信息响应 (1xx)",
        "description": "服务器已收到请求头，客户端应继续发送请求体。",
        "scenarios": [
            "客户端发送 Expect: 100-continue 请求头时",
            "大文件上传前的预检查",
            "需要认证的请求预先确认权限"
        ]
    },
    101: {
        "meaning": "Switching Protocols",
        "category": "信息响应 (1xx)",
        "description": "服务器正在按照客户端请求切换协议。",
        "scenarios": [
            "WebSocket 连接升级",
            "HTTP/1.1 升级到 HTTP/2",
            "TLS 连接建立"
        ]
    },
    102: {
        "meaning": "Processing",
        "category": "信息响应 (1xx)",
        "description": "服务器已接受请求，正在处理，但尚无响应可用。",
        "scenarios": [
            "WebDAV 复杂操作",
            "需要长时间处理的请求"
        ]
    },
    200: {
        "meaning": "OK",
        "category": "成功响应 (2xx)",
        "description": "请求已成功，响应包含所请求的数据。",
        "scenarios": [
            "网页正常加载",
            "API 数据获取成功",
            "表单提交成功"
        ]
    },
    201: {
        "meaning": "Created",
        "category": "成功响应 (2xx)",
        "description": "请求成功，服务器创建了新的资源。",
        "scenarios": [
            "用户注册成功",
            "上传文件成功",
            "创建新的数据库记录"
        ]
    },
    202: {
        "meaning": "Accepted",
        "category": "成功响应 (2xx)",
        "description": "请求已接受，但尚未处理。",
        "scenarios": [
            "异步任务提交",
            "批量处理请求",
            "需要人工审核的操作"
        ]
    },
    203: {
        "meaning": "Non-Authoritative Information",
        "category": "成功响应 (2xx)",
        "description": "请求成功，但返回的内容可能来自第三方。",
        "scenarios": [
            "代理服务器返回缓存数据",
            "CDN 缓存响应",
            "数据聚合服务"
        ]
    },
    204: {
        "meaning": "No Content",
        "category": "成功响应 (2xx)",
        "description": "请求成功，但没有返回任何内容。",
        "scenarios": [
            "删除操作成功",
            "更新操作无需返回数据",
            "心跳检测"
        ]
    },
    205: {
        "meaning": "Reset Content",
        "category": "成功响应 (2xx)",
        "description": "请求成功，客户端应重置文档视图。",
        "scenarios": [
            "表单提交后重置表单",
            "清除用户输入"
        ]
    },
    206: {
        "meaning": "Partial Content",
        "category": "成功响应 (2xx)",
        "description": "服务器正在提供部分数据，用于断点续传。",
        "scenarios": [
            "视频/音频流媒体播放",
            "大文件断点续传",
            "多线程下载"
        ]
    },
    300: {
        "meaning": "Multiple Choices",
        "category": "重定向 (3xx)",
        "description": "请求有多个可能的响应，用户需要选择其一。",
        "scenarios": [
            "多语言内容选择",
            "多格式资源可选",
            "镜像站点列表"
        ]
    },
    301: {
        "meaning": "Moved Permanently",
        "category": "重定向 (3xx)",
        "description": "资源已永久移动到新位置，客户端应更新链接。",
        "scenarios": [
            "网站域名变更",
            "URL 结构永久调整",
            "SEO 优化的 URL 重写"
        ]
    },
    302: {
        "meaning": "Found (Temporary Redirect)",
        "category": "重定向 (3xx)",
        "description": "资源临时移动到其他位置，客户端应继续使用原 URL。",
        "scenarios": [
            "临时维护页面",
            "未登录用户跳转到登录页",
            "促销活动临时跳转"
        ]
    },
    303: {
        "meaning": "See Other",
        "category": "重定向 (3xx)",
        "description": "请求已处理，客户端应使用 GET 方法获取结果。",
        "scenarios": [
            "POST 表单提交后跳转到结果页",
            "支付完成后跳转到订单详情",
            "防止表单重复提交"
        ]
    },
    304: {
        "meaning": "Not Modified",
        "category": "重定向 (3xx)",
        "description": "资源未修改，客户端可使用缓存版本。",
        "scenarios": [
            "浏览器缓存命中",
            "静态资源 (CSS/JS/图片) 缓存",
            "条件请求验证"
        ]
    },
    307: {
        "meaning": "Temporary Redirect",
        "category": "重定向 (3xx)",
        "description": "临时重定向，保持请求方法不变。",
        "scenarios": [
            "API 端点临时迁移",
            "POST 请求需要重定向时",
            "保持请求方法的临时跳转"
        ]
    },
    308: {
        "meaning": "Permanent Redirect",
        "category": "重定向 (3xx)",
        "description": "永久重定向，保持请求方法不变。",
        "scenarios": [
            "API 版本升级永久迁移",
            "HTTPS 强制跳转",
            "保持 POST 方法的永久重定向"
        ]
    },
    400: {
        "meaning": "Bad Request",
        "category": "客户端错误 (4xx)",
        "description": "请求语法错误，服务器无法理解。",
        "scenarios": [
            "请求参数格式错误",
            "JSON 格式不正确",
            "缺少必要的请求参数"
        ]
    },
    401: {
        "meaning": "Unauthorized",
        "category": "客户端错误 (4xx)",
        "description": "请求需要用户认证，未提供或认证失败。",
        "scenarios": [
            "未登录访问受保护资源",
            "Token 过期或无效",
            "用户名密码错误"
        ]
    },
    402: {
        "meaning": "Payment Required",
        "category": "客户端错误 (4xx)",
        "description": "需要支付才能访问该资源。",
        "scenarios": [
            "付费内容访问",
            "API 调用配额用尽",
            "订阅到期"
        ]
    },
    403: {
        "meaning": "Forbidden",
        "category": "客户端错误 (4xx)",
        "description": "服务器理解请求，但拒绝执行。",
        "scenarios": [
            "权限不足，禁止访问",
            "IP 地址被封禁",
            "访问被管理员限制"
        ]
    },
    404: {
        "meaning": "Not Found",
        "category": "客户端错误 (4xx)",
        "description": "服务器找不到请求的资源。",
        "scenarios": [
            "URL 输入错误",
            "页面已被删除",
            "链接失效或过期"
        ]
    },
    405: {
        "meaning": "Method Not Allowed",
        "category": "客户端错误 (4xx)",
        "description": "请求方法不被服务器允许。",
        "scenarios": [
            "对只读资源使用 POST 请求",
            "API 只支持 GET 但使用了 DELETE",
            "CORS 预检请求失败"
        ]
    },
    406: {
        "meaning": "Not Acceptable",
        "category": "客户端错误 (4xx)",
        "description": "服务器无法生成客户端可接受的响应内容。",
        "scenarios": [
            "Accept 头要求的格式不支持",
            "请求特定语言但服务器不支持",
            "编码格式不兼容"
        ]
    },
    407: {
        "meaning": "Proxy Authentication Required",
        "category": "客户端错误 (4xx)",
        "description": "需要通过代理服务器认证。",
        "scenarios": [
            "企业代理需要认证",
            "VPN 代理登录"
        ]
    },
    408: {
        "meaning": "Request Timeout",
        "category": "客户端错误 (4xx)",
        "description": "服务器等待请求超时。",
        "scenarios": [
            "网络连接缓慢",
            "客户端发送数据太慢",
            "服务器负载过高"
        ]
    },
    409: {
        "meaning": "Conflict",
        "category": "客户端错误 (4xx)",
        "description": "请求与服务器当前状态冲突。",
        "scenarios": [
            "创建重复资源",
            "并发编辑冲突",
            "用户名已被注册"
        ]
    },
    410: {
        "meaning": "Gone",
        "category": "客户端错误 (4xx)",
        "description": "资源已永久删除，且没有转发地址。",
        "scenarios": [
            "内容已永久删除",
            "API 版本已废弃",
            "产品已下架"
        ]
    },
    411: {
        "meaning": "Length Required",
        "category": "客户端错误 (4xx)",
        "description": "请求需要 Content-Length 头。",
        "scenarios": [
            "POST 请求缺少长度声明",
            "分块编码不被支持"
        ]
    },
    412: {
        "meaning": "Precondition Failed",
        "category": "客户端错误 (4xx)",
        "description": "请求头中的前提条件失败。",
        "scenarios": [
            "If-Match 条件不满足",
            "资源已被他人修改",
            "并发更新冲突检测"
        ]
    },
    413: {
        "meaning": "Payload Too Large",
        "category": "客户端错误 (4xx)",
        "description": "请求体过大，服务器无法处理。",
        "scenarios": [
            "上传文件超过大小限制",
            "POST 数据过大",
            "图片上传尺寸超限"
        ]
    },
    414: {
        "meaning": "URI Too Long",
        "category": "客户端错误 (4xx)",
        "description": "请求 URL 过长，服务器无法处理。",
        "scenarios": [
            "GET 请求参数过多",
            "URL 编码后过长",
            "查询字符串过大"
        ]
    },
    415: {
        "meaning": "Unsupported Media Type",
        "category": "客户端错误 (4xx)",
        "description": "请求的媒体类型不被支持。",
        "scenarios": [
            "上传文件格式不支持",
            "Content-Type 不正确",
            "API 只接受 JSON 但发送了 XML"
        ]
    },
    416: {
        "meaning": "Range Not Satisfiable",
        "category": "客户端错误 (4xx)",
        "description": "请求的范围超出资源大小。",
        "scenarios": [
            "断点续传范围无效",
            "视频播放范围请求超限"
        ]
    },
    417: {
        "meaning": "Expectation Failed",
        "category": "客户端错误 (4xx)",
        "description": "Expect 请求头的期望无法满足。",
        "scenarios": [
            "服务器不支持 Expect 头",
            "100-continue 期望失败"
        ]
    },
    418: {
        "meaning": "I'm a teapot",
        "category": "客户端错误 (4xx)",
        "description": "一个愚人节玩笑状态码，表示服务器拒绝煮咖啡。",
        "scenarios": [
            "开发者彩蛋",
            "幽默的错误处理",
            "API 趣味功能"
        ]
    },
    421: {
        "meaning": "Misdirected Request",
        "category": "客户端错误 (4xx)",
        "description": "请求被发送到无法产生响应的服务器。",
        "scenarios": [
            "HTTP/2 连接复用错误",
            "虚拟主机配置错误"
        ]
    },
    422: {
        "meaning": "Unprocessable Entity",
        "category": "客户端错误 (4xx)",
        "description": "请求格式正确，但语义错误，无法处理。",
        "scenarios": [
            "数据验证失败",
            "必填字段为空",
            "邮箱格式不正确"
        ]
    },
    423: {
        "meaning": "Locked",
        "category": "客户端错误 (4xx)",
        "description": "资源被锁定，无法访问。",
        "scenarios": [
            "文件正在被编辑",
            "WebDAV 资源锁定",
            "并发访问保护"
        ]
    },
    424: {
        "meaning": "Failed Dependency",
        "category": "客户端错误 (4xx)",
        "description": "请求失败是因为前一个请求失败。",
        "scenarios": [
            "批量操作部分失败",
            "依赖的前置操作失败"
        ]
    },
    425: {
        "meaning": "Too Early",
        "category": "客户端错误 (4xx)",
        "description": "服务器不愿处理可能重放的请求。",
        "scenarios": [
            "TLS 0-RTT 重放保护",
            "防止重复提交"
        ]
    },
    426: {
        "meaning": "Upgrade Required",
        "category": "客户端错误 (4xx)",
        "description": "服务器要求客户端升级到更高的协议版本。",
        "scenarios": [
            "强制使用 HTTPS",
            "需要升级到 HTTP/2"
        ]
    },
    428: {
        "meaning": "Precondition Required",
        "category": "客户端错误 (4xx)",
        "description": "服务器要求请求是有条件的。",
        "scenarios": [
            "防止更新丢失",
            "要求 If-Match 头",
            "乐观并发控制"
        ]
    },
    429: {
        "meaning": "Too Many Requests",
        "category": "客户端错误 (4xx)",
        "description": "用户在给定时间内发送了太多请求。",
        "scenarios": [
            "API 调用频率超限",
            "爬虫被限流",
            "登录尝试次数过多"
        ]
    },
    431: {
        "meaning": "Request Header Fields Too Large",
        "category": "客户端错误 (4xx)",
        "description": "请求头字段过大，服务器拒绝处理。",
        "scenarios": [
            "Cookie 过大",
            "自定义请求头过多",
            "JWT Token 过长"
        ]
    },
    451: {
        "meaning": "Unavailable For Legal Reasons",
        "category": "客户端错误 (4xx)",
        "description": "因法律原因，资源不可访问。",
        "scenarios": [
            "版权侵权内容",
            "地区法律限制访问",
            "政府要求封禁"
        ]
    },
    500: {
        "meaning": "Internal Server Error",
        "category": "服务器错误 (5xx)",
        "description": "服务器内部发生未知错误。",
        "scenarios": [
            "程序代码异常抛出",
            "数据库连接失败",
            "内存溢出或崩溃"
        ]
    },
    501: {
        "meaning": "Not Implemented",
        "category": "服务器错误 (5xx)",
        "description": "服务器不支持请求的功能。",
        "scenarios": [
            "请求方法未实现",
            "API 端点尚未开发",
            "服务器功能不完整"
        ]
    },
    502: {
        "meaning": "Bad Gateway",
        "category": "服务器错误 (5xx)",
        "description": "作为网关或代理的服务器收到无效响应。",
        "scenarios": [
            "后端服务崩溃",
            "PHP-FPM 挂掉",
            "微服务调用失败"
        ]
    },
    503: {
        "meaning": "Service Unavailable",
        "category": "服务器错误 (5xx)",
        "description": "服务器暂时不可用，通常是由于过载或维护。",
        "scenarios": [
            "服务器维护中",
            "流量峰值导致过载",
            "DDoS 攻击防护中"
        ]
    },
    504: {
        "meaning": "Gateway Timeout",
        "category": "服务器错误 (5xx)",
        "description": "网关或代理服务器等待上游响应超时。",
        "scenarios": [
            "后端程序执行时间过长",
            "数据库查询太慢",
            "第三方 API 响应超时"
        ]
    },
    505: {
        "meaning": "HTTP Version Not Supported",
        "category": "服务器错误 (5xx)",
        "description": "服务器不支持请求使用的 HTTP 版本。",
        "scenarios": [
            "使用过时的 HTTP/0.9 协议",
            "服务器配置不支持 HTTP/2"
        ]
    },
    506: {
        "meaning": "Variant Also Negotiates",
        "category": "服务器错误 (5xx)",
        "description": "服务器内部配置错误，内容协商失败。",
        "scenarios": [
            "服务器配置错误",
            "内容协商循环引用"
        ]
    },
    507: {
        "meaning": "Insufficient Storage",
        "category": "服务器错误 (5xx)",
        "description": "服务器存储空间不足。",
        "scenarios": [
            "磁盘空间已满",
            "数据库存储空间不足",
            "云存储配额用尽"
        ]
    },
    508: {
        "meaning": "Loop Detected",
        "category": "服务器错误 (5xx)",
        "description": "服务器检测到请求处理循环。",
        "scenarios": [
            "重定向死循环",
            "递归调用过深"
        ]
    },
    510: {
        "meaning": "Not Extended",
        "category": "服务器错误 (5xx)",
        "description": "需要进一步扩展请求才能完成。",
        "scenarios": [
            "HTTP 扩展协议要求",
            "需要额外的认证扩展"
        ]
    },
    511: {
        "meaning": "Network Authentication Required",
        "category": "服务器错误 (5xx)",
        "description": "需要网络认证才能访问。",
        "scenarios": [
            "公共 WiFi 需要登录",
            "酒店/机场网络认证",
            "校园网登录"
        ]
    }
}

NON_STANDARD_CODES = {418}

CUSTOM_STATUS_CODES = {}

RFC_AND_DEBUG_INFO = {
    100: {
        "rfc_refs": ["RFC 7231, Section 6.2.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.2.1"],
        "debug_tips": [
            "检查客户端是否发送了 Expect: 100-continue 请求头",
            "确认服务器支持该请求头并正确响应",
            "大文件上传时可考虑禁用该特性以避免延迟"
        ]
    },
    101: {
        "rfc_refs": ["RFC 7231, Section 6.2.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.2.2"],
        "debug_tips": [
            "检查 Upgrade 和 Connection 请求头是否正确设置",
            "验证服务器是否支持请求的协议",
            "WebSocket 连接失败时检查 Sec-WebSocket-Key 响应"
        ]
    },
    102: {
        "rfc_refs": ["RFC 2518, Section 10.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc2518#section-10.1"],
        "debug_tips": [
            "这是 WebDAV 扩展状态码，确认客户端期望 WebDAV 响应",
            "长时间运行的 WebDAV 操作可返回此状态防止超时"
        ]
    },
    200: {
        "rfc_refs": ["RFC 7231, Section 6.3.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.3.1"],
        "debug_tips": [
            "这是最常见的成功响应，通常无需特殊处理",
            "检查响应体内容是否符合预期格式",
            "确认 Content-Type 头与实际内容一致"
        ]
    },
    201: {
        "rfc_refs": ["RFC 7231, Section 6.3.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.3.2"],
        "debug_tips": [
            "检查响应中是否包含 Location 头指向新资源",
            "确认资源确实已创建（可通过 GET Location 验证）",
            "POST 或 PUT 创建资源后应返回此状态"
        ]
    },
    202: {
        "rfc_refs": ["RFC 7231, Section 6.3.3"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.3.3"],
        "debug_tips": [
            "响应应包含任务状态检查方式或预计完成时间",
            "客户端需要轮询或使用 WebSocket 获取最终结果",
            "注意：此状态不保证任务最终会成功"
        ]
    },
    203: {
        "rfc_refs": ["RFC 7231, Section 6.3.4"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.3.4"],
        "debug_tips": [
            "检查数据来源是否可信",
            "代理服务器可能修改了原始响应",
            "如需原始数据考虑直连源服务器"
        ]
    },
    204: {
        "rfc_refs": ["RFC 7231, Section 6.3.5"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.3.5"],
        "debug_tips": [
            "确认响应体确实为空（不应有任何内容）",
            "DELETE 操作成功常用此状态",
            "浏览器会保留当前页面而不会刷新"
        ]
    },
    205: {
        "rfc_refs": ["RFC 7231, Section 6.3.6"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.3.6"],
        "debug_tips": [
            "用于表单提交后重置表单",
            "响应体必须为空",
            "浏览器会清除表单输入但不导航"
        ]
    },
    206: {
        "rfc_refs": ["RFC 7233, Section 4.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7233#section-4.1"],
        "debug_tips": [
            "检查 Range 请求头格式是否正确",
            "确认 Content-Range 响应头与实际数据匹配",
            "多部分范围响应使用 multipart/byteranges"
        ]
    },
    300: {
        "rfc_refs": ["RFC 7231, Section 6.4.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.4.1"],
        "debug_tips": [
            "响应应包含可用选项列表供用户选择",
            "如果有首选资源应设置 Location 头",
            "现代浏览器很少触发用户选择"
        ]
    },
    301: {
        "rfc_refs": ["RFC 7231, Section 6.4.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.4.2"],
        "debug_tips": [
            "检查 Location 响应头是否正确",
            "永久重定向会被浏览器缓存，注意清除缓存测试",
            "SEO 友好，会传递链接权重",
            "注意：POST 请求可能被改为 GET"
        ]
    },
    302: {
        "rfc_refs": ["RFC 7231, Section 6.4.3"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.4.3"],
        "debug_tips": [
            "临时重定向，搜索引擎应保留原 URL",
            "历史上客户端常将 POST 改为 GET",
            "如需保持方法使用 307，需永久使用 308"
        ]
    },
    303: {
        "rfc_refs": ["RFC 7231, Section 6.4.4"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.4.4"],
        "debug_tips": [
            "总是使用 GET 方法获取结果",
            "常用于 POST/PUT/DELETE 后的结果页重定向",
            "防止刷新时重复提交表单"
        ]
    },
    304: {
        "rfc_refs": ["RFC 7232, Section 4.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7232#section-4.1"],
        "debug_tips": [
            "检查 If-None-Match 或 If-Modified-Since 请求头",
            "验证 ETag 和 Last-Modified 配置是否正确",
            "响应体必须为空，只返回头信息"
        ]
    },
    307: {
        "rfc_refs": ["RFC 7231, Section 6.4.7"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.4.7"],
        "debug_tips": [
            "临时重定向，保持原请求方法和请求体",
            "与 302 不同：POST 重定向后仍为 POST",
            "可安全用于表单提交重定向"
        ]
    },
    308: {
        "rfc_refs": ["RFC 7538"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7538"],
        "debug_tips": [
            "永久重定向，保持原请求方法",
            "与 301 不同：POST 重定向后仍为 POST",
            "会被浏览器永久缓存"
        ]
    },
    400: {
        "rfc_refs": ["RFC 7231, Section 6.5.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.1"],
        "debug_tips": [
            "检查请求参数格式是否正确",
            "验证 JSON/XML 请求体语法",
            "查看服务器日志获取详细错误信息",
            "检查 Content-Type 与实际提交数据是否匹配",
            "确认 URL 编码是否正确"
        ]
    },
    401: {
        "rfc_refs": ["RFC 7235, Section 3.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7235#section-3.1"],
        "debug_tips": [
            "检查 WWW-Authenticate 响应头了解认证方式",
            "验证 Token 是否有效且未过期",
            "确认 Authorization 请求头格式正确",
            "Basic Auth 检查用户名密码是否正确编码",
            "OAuth2 检查 scope 权限是否足够"
        ]
    },
    402: {
        "rfc_refs": ["RFC 7231, Section 6.5.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.2"],
        "debug_tips": [
            "检查账户余额或订阅状态",
            "查看 API 调用配额是否用尽",
            "确认支付方式是否有效",
            "此状态码目前较少使用，主要保留给未来"
        ]
    },
    403: {
        "rfc_refs": ["RFC 7231, Section 6.5.3"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.3"],
        "debug_tips": [
            "认证已通过但权限不足，检查用户角色",
            "查看文件系统权限配置",
            "检查 IP 地址是否在白名单/黑名单中",
            "验证 CSRF Token 是否有效",
            "与 401 不同：重新认证也无法解决，需要提升权限"
        ]
    },
    404: {
        "rfc_refs": ["RFC 7231, Section 6.5.4"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.4"],
        "debug_tips": [
            "检查 URL 拼写是否正确，注意大小写",
            "确认资源是否已被删除或移动",
            "查看服务器路由配置",
            "检查文件是否存在于正确位置",
            "REST API 确认资源 ID 是否有效"
        ]
    },
    405: {
        "rfc_refs": ["RFC 7231, Section 6.5.5"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.5"],
        "debug_tips": [
            "检查 Allow 响应头查看支持的方法",
            "确认使用了正确的 HTTP 方法",
            "REST API：创建用 POST，读取用 GET，更新用 PUT，删除用 DELETE",
            "检查 CORS 预检请求是否成功",
            "验证服务器路由配置"
        ]
    },
    406: {
        "rfc_refs": ["RFC 7231, Section 6.5.6"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.6"],
        "debug_tips": [
            "检查 Accept 请求头的值",
            "确认服务器支持请求的 MIME 类型",
            "尝试移除 Accept 头使用默认值",
            "检查 Accept-Language、Accept-Encoding 等头"
        ]
    },
    407: {
        "rfc_refs": ["RFC 7235, Section 3.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7235#section-3.2"],
        "debug_tips": [
            "检查 Proxy-Authenticate 响应头",
            "配置代理服务器认证信息",
            "确认 Proxy-Authorization 请求头已正确设置",
            "检查代理服务器地址和端口配置"
        ]
    },
    408: {
        "rfc_refs": ["RFC 7231, Section 6.5.7"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.7"],
        "debug_tips": [
            "检查网络连接是否稳定",
            "确认服务器负载是否过高",
            "大请求考虑增加超时时间",
            "检查客户端发送数据速度",
            "可尝试重试请求"
        ]
    },
    409: {
        "rfc_refs": ["RFC 7231, Section 6.5.8"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.8"],
        "debug_tips": [
            "检查是否存在重复资源",
            "查看数据版本控制冲突",
            "使用乐观锁或悲观锁防止并发冲突",
            "获取最新资源状态后重试",
            "用户注册时常见：用户名已存在"
        ]
    },
    410: {
        "rfc_refs": ["RFC 7231, Section 6.5.9"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.9"],
        "debug_tips": [
            "资源已永久删除，不要重试",
            "更新书签或链接",
            "与 404 不同：明确告知资源曾存在但已删除",
            "搜索引擎应从索引中移除该 URL"
        ]
    },
    411: {
        "rfc_refs": ["RFC 7231, Section 6.5.10"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.10"],
        "debug_tips": [
            "添加 Content-Length 请求头",
            "或使用 Transfer-Encoding: chunked",
            "确认不是空的 POST/PUT 请求",
            "检查 HTTP 客户端库配置"
        ]
    },
    412: {
        "rfc_refs": ["RFC 7232, Section 4.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7232#section-4.2"],
        "debug_tips": [
            "检查 If-Match/If-None-Match 头的 ETag 值",
            "确认资源是否已被他人修改",
            "获取最新资源版本后重试",
            "用于防止更新丢失的乐观并发控制"
        ]
    },
    413: {
        "rfc_refs": ["RFC 7231, Section 6.5.11"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.11"],
        "debug_tips": [
            "压缩请求体或拆分大文件",
            "检查服务器上传大小限制配置",
            "使用分块上传或断点续传",
            "Nginx: client_max_body_size, PHP: upload_max_filesize"
        ]
    },
    414: {
        "rfc_refs": ["RFC 7231, Section 6.5.12"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.12"],
        "debug_tips": [
            "减少查询参数数量或长度",
            "将参数移到请求体中（使用 POST）",
            "检查服务器 URL 长度限制配置",
            "避免在 URL 中传递大量数据"
        ]
    },
    415: {
        "rfc_refs": ["RFC 7231, Section 6.5.13"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.13"],
        "debug_tips": [
            "检查 Content-Type 请求头是否正确",
            "确认服务器支持该媒体类型",
            "验证请求体格式是否匹配 Content-Type",
            "上传文件检查文件扩展名和 MIME 类型",
            "API 可能只支持 application/json"
        ]
    },
    416: {
        "rfc_refs": ["RFC 7233, Section 4.4"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7233#section-4.4"],
        "debug_tips": [
            "检查 Range 请求头格式",
            "确认请求范围不超过资源实际大小",
            "先通过 HEAD 请求获取资源大小",
            "断点续传前验证文件是否完整"
        ]
    },
    417: {
        "rfc_refs": ["RFC 7231, Section 6.5.14"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.14"],
        "debug_tips": [
            "移除 Expect 请求头",
            "确认服务器支持 Expect: 100-continue",
            "代理服务器可能不支持此特性",
            "大文件上传可考虑直接发送"
        ]
    },
    418: {
        "rfc_refs": ["RFC 2324 (April Fools' RFC)"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc2324"],
        "debug_tips": [
            "这是一个愚人节玩笑状态码",
            "某些框架或 API 用作彩蛋",
            "检查是否有特殊的 API 限制或幽默响应",
            "实际开发中应避免使用"
        ]
    },
    421: {
        "rfc_refs": ["RFC 7540, Section 9.1.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7540#section-9.1.2"],
        "debug_tips": [
            "HTTP/2 特定错误，检查连接复用配置",
            "确认请求发送到了正确的虚拟主机",
            "验证 TLS SNI 配置",
            "尝试使用 HTTP/1.1 复现问题"
        ]
    },
    422: {
        "rfc_refs": ["RFC 4918, Section 11.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc4918#section-11.2"],
        "debug_tips": [
            "请求格式正确但语义错误",
            "检查数据验证规则",
            "必填字段是否都已提供",
            "验证邮箱、电话等格式",
            "查看响应体中的具体字段错误信息"
        ]
    },
    423: {
        "rfc_refs": ["RFC 4918, Section 11.3"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc4918#section-11.3"],
        "debug_tips": [
            "WebDAV 资源锁定，稍后重试",
            "检查是否有其他用户正在编辑",
            "确认锁的超时时间",
            "必要时可强制释放锁"
        ]
    },
    424: {
        "rfc_refs": ["RFC 4918, Section 11.4"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc4918#section-11.4"],
        "debug_tips": [
            "批量操作中某个步骤失败",
            "检查前置操作的错误信息",
            "修复依赖问题后重试",
            "WebDAV 多状态响应常见"
        ]
    },
    425: {
        "rfc_refs": ["RFC 8470"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc8470"],
        "debug_tips": [
            "TLS 0-RTT 重放保护触发",
            "使用完整 TLS 握手重试",
            "避免在 0-RTT 中发送非幂等请求",
            "检查服务器 Early-Data 配置"
        ]
    },
    426: {
        "rfc_refs": ["RFC 7231, Section 6.5.15"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.5.15"],
        "debug_tips": [
            "检查 Upgrade 响应头了解需要的协议",
            "升级到 HTTPS",
            "使用更高版本的 HTTP 协议",
            "确认客户端支持请求的协议版本"
        ]
    },
    428: {
        "rfc_refs": ["RFC 6585, Section 3"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc6585#section-3"],
        "debug_tips": [
            "服务器要求条件请求",
            "添加 If-Match 或 If-Unmodified-Since 头",
            "获取资源当前 ETag 后重试",
            "用于防止更新丢失"
        ]
    },
    429: {
        "rfc_refs": ["RFC 6585, Section 4"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc6585#section-4"],
        "debug_tips": [
            "检查 Retry-After 响应头了解何时可重试",
            "实现请求退避策略（指数退避）",
            "减少请求频率",
            "检查 API 配额使用情况",
            "考虑批量请求减少调用次数"
        ]
    },
    431: {
        "rfc_refs": ["RFC 6585, Section 5"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc6585#section-5"],
        "debug_tips": [
            "减少 Cookie 数量和大小",
            "移除不必要的自定义请求头",
            "检查服务器请求头大小限制",
            "JWT Token 过长可考虑存在 Cookie 或请求体"
        ]
    },
    451: {
        "rfc_refs": ["RFC 7725"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7725"],
        "debug_tips": [
            "法律原因无法访问，通常无法技术解决",
            "检查是否有地区限制",
            "尝试使用不同的网络环境",
            "联系网站管理员了解详情"
        ]
    },
    500: {
        "rfc_refs": ["RFC 7231, Section 6.6.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.6.1"],
        "debug_tips": [
            "查看服务器错误日志获取堆栈跟踪",
            "检查数据库连接和查询",
            "验证第三方服务调用是否成功",
            "检查内存使用和磁盘空间",
            "近期代码部署可能引入 bug"
        ]
    },
    501: {
        "rfc_refs": ["RFC 7231, Section 6.6.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.6.2"],
        "debug_tips": [
            "确认服务器支持该 HTTP 方法",
            "检查 API 文档确认端点是否实现",
            "确认使用了正确的 HTTP 版本",
            "WebDAV 扩展方法需要服务器支持"
        ]
    },
    502: {
        "rfc_refs": ["RFC 7231, Section 6.6.3"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.6.3"],
        "debug_tips": [
            "检查后端服务是否正常运行",
            "查看反向代理配置（Nginx/Apache）",
            "确认 PHP-FPM/uWSGI 等应用服务器状态",
            "检查防火墙是否阻断后端连接",
            "验证后端服务端口是否正确监听"
        ]
    },
    503: {
        "rfc_refs": ["RFC 7231, Section 6.6.4"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.6.4"],
        "debug_tips": [
            "检查 Retry-After 头（如果有）",
            "确认服务器是否在维护中",
            "查看服务器负载和资源使用情况",
            "检查是否被 DDoS 攻击或流量突增",
            "自动伸缩组实例是否足够"
        ]
    },
    504: {
        "rfc_refs": ["RFC 7231, Section 6.6.5"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.6.5"],
        "debug_tips": [
            "检查后端服务性能瓶颈",
            "优化慢查询或耗时操作",
            "增加代理超时时间（临时方案）",
            "考虑异步处理长时间任务",
            "检查第三方 API 响应时间"
        ]
    },
    505: {
        "rfc_refs": ["RFC 7231, Section 6.6.6"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc7231#section-6.6.6"],
        "debug_tips": [
            "升级 HTTP 客户端版本",
            "配置服务器支持请求的 HTTP 版本",
            "HTTP/2 需要 ALPN 支持和正确的 SSL 配置",
            "回退到 HTTP/1.1 测试"
        ]
    },
    506: {
        "rfc_refs": ["RFC 2295, Section 8.1"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc2295#section-8.1"],
        "debug_tips": [
            "检查服务器内容协商配置",
            "修复循环引用的变体配置",
            "这通常是服务器配置错误",
            "联系服务器管理员"
        ]
    },
    507: {
        "rfc_refs": ["RFC 4918, Section 11.5"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc4918#section-11.5"],
        "debug_tips": [
            "检查磁盘空间使用情况",
            "清理旧文件释放空间",
            "增加存储配额",
            "数据库检查表空间是否足够",
            "归档历史数据"
        ]
    },
    508: {
        "rfc_refs": ["RFC 5842, Section 7.2"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc5842#section-7.2"],
        "debug_tips": [
            "检查重定向规则是否有循环",
            "验证资源绑定是否形成循环",
            "修复 WebDAV 配置中的循环引用",
            "检查递归逻辑是否有终止条件"
        ]
    },
    510: {
        "rfc_refs": ["RFC 2774"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc2774"],
        "debug_tips": [
            "检查需要的 HTTP 扩展",
            "确认客户端支持请求的扩展",
            "在请求中声明所需的扩展",
            "这是一个较少使用的状态码"
        ]
    },
    511: {
        "rfc_refs": ["RFC 6585, Section 6"],
        "rfc_links": ["https://datatracker.ietf.org/doc/html/rfc6585#section-6"],
        "debug_tips": [
            "需要网络认证，打开浏览器访问任意 HTTP 网站",
            "完成 captive portal 登录",
            "检查 WiFi/网络连接条款",
            "酒店/机场网络常见"
        ]
    }
}


def get_category(code):
    if 100 <= code < 200:
        return "信息响应 (1xx)"
    elif 200 <= code < 300:
        return "成功响应 (2xx)"
    elif 300 <= code < 400:
        return "重定向 (3xx)"
    elif 400 <= code < 500:
        return "客户端错误 (4xx)"
    elif 500 <= code < 600:
        return "服务器错误 (5xx)"
    else:
        return None


def lookup_status_code(code):
    if not isinstance(code, int):
        try:
            code = int(code)
        except (ValueError, TypeError):
            return {
                "success": True,
                "code": str(code),
                "meaning": "未定义",
                "category": "无效输入",
                "description": f"'{code}' 不是有效的 HTTP 状态码，状态码应为 0-599 之间的整数。",
                "scenarios": [],
                "standard": False,
                "rfc_refs": [],
                "rfc_links": [],
                "debug_tips": []
            }

    category = get_category(code)

    if category is None:
        return {
            "success": True,
            "code": code,
            "meaning": "未定义",
            "category": "超出范围",
            "description": f"状态码 {code} 不在 HTTP 标准范围内（100-599），属于未定义的状态码。",
            "scenarios": [],
            "standard": False,
            "rfc_refs": [],
            "rfc_links": [],
            "debug_tips": []
        }

    rfc_info = RFC_AND_DEBUG_INFO.get(code, {"rfc_refs": [], "rfc_links": [], "debug_tips": []})

    if code in CUSTOM_STATUS_CODES:
        info = CUSTOM_STATUS_CODES[code]
        return {
            "success": True,
            "code": code,
            "meaning": info["meaning"],
            "category": info.get("category", category),
            "description": info.get("description", ""),
            "scenarios": info.get("scenarios", []),
            "standard": False,
            "rfc_refs": info.get("rfc_refs", []),
            "rfc_links": info.get("rfc_links", []),
            "debug_tips": info.get("debug_tips", [])
        }

    if code in HTTP_STATUS_CODES:
        info = HTTP_STATUS_CODES[code]
        is_standard = code not in NON_STANDARD_CODES
        return {
            "success": True,
            "code": code,
            "meaning": info["meaning"],
            "category": info["category"],
            "description": info["description"],
            "scenarios": info["scenarios"],
            "standard": is_standard,
            "rfc_refs": rfc_info.get("rfc_refs", []),
            "rfc_links": rfc_info.get("rfc_links", []),
            "debug_tips": rfc_info.get("debug_tips", [])
        }

    return {
        "success": True,
        "code": code,
        "meaning": "未定义",
        "category": category,
        "description": f"这是一个 {category} 范围内的状态码，但不是标准定义的状态码。可能是自定义或实验性状态码。",
        "scenarios": ["非标准状态码", "自定义扩展", "实验性功能"],
        "standard": False,
        "rfc_refs": [],
        "rfc_links": [],
        "debug_tips": []
    }


def lookup_status_codes(codes=None, category_filter=None):
    if codes is not None:
        results = []
        for code in codes:
            results.append(lookup_status_code(code))
        return results
    elif category_filter is not None:
        all_codes = list_status_codes(category_filter)
        results = []
        for entry in all_codes:
            results.append(lookup_status_code(entry["code"]))
        return results
    else:
        return []


def batch_lookup(category_filter=None):
    return lookup_status_codes(category_filter=category_filter)


def add_status_code(code, meaning, description="", scenarios=None, category=None):
    try:
        code = int(code)
    except (ValueError, TypeError):
        return {"success": False, "error": "无效的状态码，请输入一个有效的整数"}

    if code < 0 or code > 599:
        return {"success": False, "error": "状态码必须在 0-599 之间"}

    if code in HTTP_STATUS_CODES and code not in NON_STANDARD_CODES:
        return {"success": False, "error": f"状态码 {code} 已是标准状态码（{HTTP_STATUS_CODES[code]['meaning']}），无需自定义添加"}

    if code in CUSTOM_STATUS_CODES:
        return {"success": False, "error": f"自定义状态码 {code} 已存在（{CUSTOM_STATUS_CODES[code]['meaning']}），如需修改请先删除"}

    entry = {"meaning": meaning, "description": description}
    if scenarios is not None:
        entry["scenarios"] = scenarios
    else:
        entry["scenarios"] = []
    if category is not None:
        entry["category"] = category
    else:
        auto_category = get_category(code)
        entry["category"] = auto_category if auto_category else "自定义"

    CUSTOM_STATUS_CODES[code] = entry
    return {"success": True, "code": code, "meaning": meaning, "message": f"自定义状态码 {code} ({meaning}) 添加成功"}


def remove_status_code(code):
    try:
        code = int(code)
    except (ValueError, TypeError):
        return {"success": False, "error": "无效的状态码，请输入一个有效的整数"}

    if code in CUSTOM_STATUS_CODES:
        meaning = CUSTOM_STATUS_CODES[code]["meaning"]
        del CUSTOM_STATUS_CODES[code]
        return {"success": True, "code": code, "message": f"自定义状态码 {code} ({meaning}) 已删除"}
    else:
        return {"success": False, "error": f"自定义状态码 {code} 不存在，只能删除自定义添加的状态码"}


def list_status_codes(category_filter=None):
    results = []
    for code in sorted(HTTP_STATUS_CODES.keys()):
        info = HTTP_STATUS_CODES[code]
        is_standard = code not in NON_STANDARD_CODES
        entry = {
            "code": code,
            "meaning": info["meaning"],
            "category": info["category"],
            "standard": is_standard
        }
        if category_filter is None or category_filter in info["category"]:
            results.append(entry)
    for code in sorted(CUSTOM_STATUS_CODES.keys()):
        info = CUSTOM_STATUS_CODES[code]
        entry = {
            "code": code,
            "meaning": info["meaning"],
            "category": info.get("category", get_category(code) or "自定义"),
            "standard": False
        }
        if category_filter is None or category_filter in entry["category"]:
            results.append(entry)
    return results


def format_result(result, show_rfc=True, show_debug=True):
    if not result.get("success", True):
        return f"❌ 错误: {result['error']}"

    output = []
    output.append("📋 HTTP 状态码查询结果")
    output.append("=" * 50)
    output.append(f"状态码: {result['code']}")
    output.append(f"含义:   {result['meaning']}")
    output.append(f"分类:   {result['category']}")
    standard_label = "✅ 标准状态码" if result.get("standard", False) else "⚠️ 非标准/自定义状态码"
    output.append(f"类型:   {standard_label}")
    output.append("=" * 50)

    if result.get("description"):
        output.append("📝 描述:")
        output.append(f"  {result['description']}")

    if result.get("scenarios"):
        output.append("")
        output.append("🔍 常见场景:")
        for i, scenario in enumerate(result["scenarios"], 1):
            output.append(f"  {i}. {scenario}")

    if show_rfc and result.get("rfc_refs"):
        output.append("")
        output.append("📚 RFC 参考:")
        for rfc in result["rfc_refs"]:
            output.append(f"  • {rfc}")
        if result.get("rfc_links"):
            output.append("  链接:")
            for link in result["rfc_links"]:
                output.append(f"    {link}")

    if show_debug and result.get("debug_tips"):
        output.append("")
        output.append("🛠️ 调试建议:")
        for i, tip in enumerate(result["debug_tips"], 1):
            output.append(f"  {i}. {tip}")

    return "\n".join(output)


def format_batch_results(results, compact=False):
    if compact:
        lines = [f"{'状态码':<8}{'含义':<35}{'分类':<25}{'类型'}"]
        lines.append("-" * 80)
        for r in results:
            std = "标准" if r.get("standard", False) else "非标准"
            lines.append(f"{r.get('code', '?'):<8}{r.get('meaning', ''):<35}{r.get('category', ''):<25}{std}")
        lines.append(f"\n共 {len(results)} 个状态码")
        return "\n".join(lines)
    else:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(format_result(r))
            if i < len(results):
                parts.append("\n" + "~" * 50 + "\n")
        return "\n".join(parts)


def _parse_add_command(parts):
    if len(parts) < 3:
        print("用法: add <状态码> <含义> [描述] [场景1,场景2,...]")
        print("示例: add 999 CustomError 自定义错误 这是一个自定义错误 用户自定义,扩展功能")
        return
    code = parts[1]
    meaning = parts[2]
    description = parts[3] if len(parts) > 3 else ""
    scenarios = parts[4].split(",") if len(parts) > 4 else []
    result = add_status_code(code, meaning, description, scenarios)
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['error']}")


def _parse_remove_command(parts):
    if len(parts) < 2:
        print("用法: remove <状态码>")
        return
    result = remove_status_code(parts[1])
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['error']}")


def _handle_list_command(parts):
    category_filter = parts[1] if len(parts) > 1 else None
    codes = list_status_codes(category_filter)
    if not codes:
        print("没有找到匹配的状态码")
        return
    print(f"{'状态码':<8}{'含义':<35}{'分类':<20}{'类型'}")
    print("-" * 80)
    for entry in codes:
        std = "标准" if entry["standard"] else "非标准"
        print(f"{entry['code']:<8}{entry['meaning']:<35}{entry['category']:<20}{std}")
    print(f"\n共 {len(codes)} 个状态码")


def _handle_batch_command(parts):
    if len(parts) < 2:
        print("用法: batch <分类> [--compact]")
        print("分类: 1xx, 2xx, 3xx, 4xx, 5xx 或完整分类名")
        print("示例: batch 4xx")
        print("      batch 客户端错误 --compact")
        return
    category_filter = parts[1]
    compact = "--compact" in parts or "-c" in parts
    results = lookup_status_codes(category_filter=category_filter)
    if not results:
        print(f"没有找到分类为 '{category_filter}' 的状态码")
        return
    print(format_batch_results(results, compact=compact))


def main():
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "add" and len(sys.argv) >= 4:
            code = sys.argv[2]
            meaning = sys.argv[3]
            description = sys.argv[4] if len(sys.argv) > 4 else ""
            scenarios = sys.argv[5].split(",") if len(sys.argv) > 5 else []
            result = add_status_code(code, meaning, description, scenarios)
            if result["success"]:
                print(f"✅ {result['message']}")
            else:
                print(f"❌ {result['error']}")
            return
        elif command == "remove" and len(sys.argv) >= 3:
            result = remove_status_code(sys.argv[2])
            if result["success"]:
                print(f"✅ {result['message']}")
            else:
                print(f"❌ {result['error']}")
            return
        elif command == "list":
            category_filter = sys.argv[2] if len(sys.argv) > 2 else None
            _handle_list_command(["list", category_filter] if category_filter else ["list"])
            return
        elif command == "batch":
            _handle_batch_command(sys.argv[1:])
            return

        for code in sys.argv[1:]:
            result = lookup_status_code(code)
            print(format_result(result))
            print()
    else:
        print("HTTP 状态码查询工具 (增强版)")
        print("=" * 50)
        print("命令:")
        print("  <状态码>              - 查询状态码（可多个，空格分隔）")
        print("  batch <分类> [-c]     - 按分类批量查询详情 (-c 简洁模式)")
        print("  list [分类]           - 列出状态码简要列表")
        print("  add <码> <含义> [描述] [场景1,场景2] - 添加自定义状态码")
        print("  remove <码>           - 删除自定义状态码")
        print("  q / quit              - 退出")
        print()
        print("分类选项: 1xx, 2xx, 3xx, 4xx, 5xx")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ('q', 'quit', 'exit'):
                    print("再见！")
                    break

                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd == "add":
                    _parse_add_command(parts)
                elif cmd == "remove":
                    _parse_remove_command(parts)
                elif cmd == "list":
                    _handle_list_command(parts)
                elif cmd == "batch":
                    _handle_batch_command(parts)
                else:
                    results = lookup_status_codes(codes=parts)
                    print(format_batch_results(results))
                    print()

            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break


if __name__ == "__main__":
    main()
