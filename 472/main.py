import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from jinja2 import Environment, DictLoader, TemplateError, Undefined
from typing import Dict, Any, Optional, List

app = FastAPI(
    title="邮件模板渲染API",
    description="基于Jinja2引擎的邮件模板渲染服务，支持HTML模板和变量替换，内置常用邮件模板库",
    version="2.0.0"
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


class DefaultUndefined(Undefined):
    _default_value = ""

    def __str__(self):
        return self._default_value

    def __repr__(self):
        return f"DefaultUndefined({self._undefined_name!r})"

    def __bool__(self):
        return False

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __eq__(self, other):
        if isinstance(other, DefaultUndefined):
            return self._undefined_name == other._undefined_name
        return False

    def __hash__(self):
        return hash(self._undefined_name)


class PlaceholderUndefined(Undefined):
    def __str__(self):
        return "{{ " + self._undefined_name + " }}"

    def __repr__(self):
        return f"PlaceholderUndefined({self._undefined_name!r})"

    def __bool__(self):
        return False

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __eq__(self, other):
        if isinstance(other, PlaceholderUndefined):
            return self._undefined_name == other._undefined_name
        return False

    def __hash__(self):
        return hash(self._undefined_name)


def _load_builtin_templates() -> Dict[str, str]:
    templates = {}
    if TEMPLATES_DIR.exists():
        for filepath in TEMPLATES_DIR.glob("*.html"):
            templates[filepath.name] = filepath.read_text(encoding="utf-8")
    return templates


BUILTIN_TEMPLATES: Dict[str, str] = _load_builtin_templates()

TEMPLATE_METADATA: Dict[str, Dict[str, Any]] = {
    "base.html": {
        "name": "基础布局模板",
        "description": "邮件基础布局，包含头部、正文、尾部区域，供其他模板继承使用",
        "variables": ["site_name", "user_name", "year"],
        "is_base": True
    },
    "welcome.html": {
        "name": "欢迎邮件",
        "description": "新用户注册后发送的欢迎及账户激活邮件",
        "variables": ["site_name", "user_name", "activate_url", "expire_hours", "invite_code"],
        "extends": "base.html",
        "is_base": False
    },
    "reset_password.html": {
        "name": "密码重置邮件",
        "description": "用户请求重置密码时发送的邮件",
        "variables": ["site_name", "user_name", "reset_url", "expire_minutes"],
        "extends": "base.html",
        "is_base": False
    },
    "order_notification.html": {
        "name": "订单通知邮件",
        "description": "订单状态变更通知，包含商品明细和物流信息",
        "variables": ["site_name", "user_name", "order_id", "order_items", "total_amount", "order_status", "order_time", "shipping_address", "tracking_url"],
        "extends": "base.html",
        "is_base": False
    }
}


def _create_env(
    keep_placeholder: bool = False,
    default_value: str = "",
    extra_templates: Optional[Dict[str, str]] = None
) -> Environment:
    all_templates = dict(BUILTIN_TEMPLATES)
    if extra_templates:
        all_templates.update(extra_templates)

    if keep_placeholder:
        undefined_class = PlaceholderUndefined
    else:
        undefined_class = type(
            "CustomDefaultUndefined",
            (DefaultUndefined,),
            {"_default_value": default_value}
        )

    return Environment(
        loader=DictLoader(all_templates),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=undefined_class
    )


def _extract_preview(rendered: str) -> Optional[str]:
    if not rendered:
        return None
    text_only = re.sub(r'<[^>]+>', '', rendered).strip()
    return text_only[:200] + ("..." if len(text_only) > 200 else "")


class RenderRequest(BaseModel):
    template: str = Field(..., description="Jinja2模板内容，包含变量占位符", json_schema_extra={"example": """
<!DOCTYPE html>
<html>
<body>
    <h1>您好, {{ name }}!</h1>
    <p>您的订单号: <strong>{{ order_id }}</strong></p>
    <p>金额: ¥{{ amount }}</p>
</body>
</html>
    """})
    variables: Dict[str, Any] = Field(default_factory=dict, description="模板变量字典", json_schema_extra={"example": {"name": "张三", "order_id": "ORD2024001", "amount": 299.99}})
    keep_placeholder: bool = Field(default=False, description="变量缺失时是否保留原始占位符（如 {{ name }}），为False则使用default_for_missing替换", json_schema_extra={"example": False})
    default_for_missing: str = Field(default="", description="变量未提供时的默认替换值，仅在keep_placeholder=False时生效", json_schema_extra={"example": ""})


class NamedRenderRequest(BaseModel):
    template_name: str = Field(..., description="内置模板名称（如 welcome.html、reset_password.html）", json_schema_extra={"example": "welcome.html"})
    variables: Dict[str, Any] = Field(default_factory=dict, description="模板变量字典", json_schema_extra={"example": {"site_name": "MyApp", "user_name": "张三", "activate_url": "https://example.com/activate?token=abc", "year": "2026"}})
    keep_placeholder: bool = Field(default=False, description="变量缺失时是否保留原始占位符", json_schema_extra={"example": False})
    default_for_missing: str = Field(default="", description="变量未提供时的默认替换值", json_schema_extra={"example": ""})
    extra_templates: Optional[Dict[str, str]] = Field(None, description="额外模板映射（用于自定义继承模板），key为模板名，value为模板内容", json_schema_extra={"example": {"base.html": "<html>{% block content %}{% endblock %}</html>"}})


class RenderResponse(BaseModel):
    success: bool = True
    rendered_content: str = Field(..., description="渲染后的邮件正文内容")
    template_preview: Optional[str] = Field(None, description="模板预览（纯文本前200字符）")


class TemplateInfo(BaseModel):
    name: str = Field(..., description="模板文件名")
    display_name: str = Field(..., description="模板显示名称")
    description: str = Field(..., description="模板描述")
    variables: List[str] = Field(..., description="模板所需变量列表")
    extends: Optional[str] = Field(None, description="继承的父模板")
    is_base: bool = Field(..., description="是否为基础布局模板")


class TemplateListResponse(BaseModel):
    templates: List[TemplateInfo] = Field(..., description="模板列表")
    total: int = Field(..., description="模板总数")


class TemplatePreviewResponse(BaseModel):
    name: str = Field(..., description="模板文件名")
    display_name: str = Field(..., description="模板显示名称")
    description: str = Field(..., description="模板描述")
    variables: List[str] = Field(..., description="模板所需变量列表")
    extends: Optional[str] = Field(None, description="继承的父模板")
    sample_rendered: Optional[str] = Field(None, description="使用示例变量渲染的结果预览")
    sample_preview: Optional[str] = Field(None, description="纯文本预览")


@app.post("/api/render/email", response_model=RenderResponse, summary="渲染邮件模板（自定义模板内容）")
async def render_email_template(request: RenderRequest):
    """
    接收Jinja2模板内容和变量字典，渲染生成最终邮件正文。

    - **template**: Jinja2模板字符串，支持 {{ variable }}、{% for %}、{% if %} 等语法
    - **variables**: 变量字典，key为模板中使用的变量名
    - **keep_placeholder**: 变量缺失时保留原始占位符（如 {{ name }}），优先级高于default_for_missing
    - **default_for_missing**: 变量未提供时的默认替换值，仅在keep_placeholder=False时生效，默认空字符串
    """
    try:
        env = _create_env(
            keep_placeholder=request.keep_placeholder,
            default_value=request.default_for_missing
        )
        template = env.from_string(request.template)
        rendered = template.render(**request.variables)

        return RenderResponse(
            success=True,
            rendered_content=rendered,
            template_preview=_extract_preview(rendered)
        )

    except TemplateError as e:
        raise HTTPException(status_code=400, detail=f"模板语法错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"渲染失败: {str(e)}")


@app.post("/api/render/named", response_model=RenderResponse, summary="渲染内置模板（按模板名）")
async def render_named_template(request: NamedRenderRequest):
    """
    使用内置模板名称渲染邮件，支持模板继承。

    - **template_name**: 内置模板名称，如 welcome.html、reset_password.html、order_notification.html
    - **variables**: 模板变量字典
    - **keep_placeholder**: 变量缺失时保留原始占位符
    - **default_for_missing**: 变量未提供时的默认替换值
    - **extra_templates**: 额外模板映射（用于自定义继承模板）
    """
    if request.template_name not in BUILTIN_TEMPLATES:
        available = [k for k in BUILTIN_TEMPLATES if not TEMPLATE_METADATA.get(k, {}).get("is_base", False)]
        raise HTTPException(
            status_code=404,
            detail=f"模板 '{request.template_name}' 不存在。可用模板: {', '.join(available)}"
        )

    try:
        env = _create_env(
            keep_placeholder=request.keep_placeholder,
            default_value=request.default_for_missing,
            extra_templates=request.extra_templates
        )
        template = env.get_template(request.template_name)
        rendered = template.render(**request.variables)

        return RenderResponse(
            success=True,
            rendered_content=rendered,
            template_preview=_extract_preview(rendered)
        )

    except TemplateError as e:
        raise HTTPException(status_code=400, detail=f"模板语法错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"渲染失败: {str(e)}")


@app.get("/api/templates", response_model=TemplateListResponse, summary="获取模板列表")
async def list_templates():
    """返回所有内置邮件模板的信息列表，包括名称、描述、所需变量等。"""
    templates = []
    for name, meta in TEMPLATE_METADATA.items():
        if name in BUILTIN_TEMPLATES:
            templates.append(TemplateInfo(
                name=name,
                display_name=meta["name"],
                description=meta["description"],
                variables=meta["variables"],
                extends=meta.get("extends"),
                is_base=meta.get("is_base", False)
            ))
    return TemplateListResponse(templates=templates, total=len(templates))


@app.get("/api/templates/{template_name}/preview", response_model=TemplatePreviewResponse, summary="预览模板")
async def preview_template(template_name: str):
    """
    预览指定模板，使用示例变量渲染生成预览内容。

    - **template_name**: 模板文件名，如 welcome.html
    """
    if template_name not in BUILTIN_TEMPLATES:
        raise HTTPException(status_code=404, detail=f"模板 '{template_name}' 不存在")

    if template_name not in TEMPLATE_METADATA:
        raise HTTPException(status_code=404, detail=f"模板 '{template_name}' 缺少元数据")

    meta = TEMPLATE_METADATA[template_name]

    sample_variables: Dict[str, Any] = {
        "site_name": "示例平台",
        "user_name": "示例用户",
        "year": "2026",
        "activate_url": "https://example.com/activate?token=sample_token",
        "expire_hours": 24,
        "invite_code": "WELCOME2026",
        "reset_url": "https://example.com/reset?token=sample_token",
        "expire_minutes": 30,
        "order_id": "ORD-2026-001",
        "order_status": "确认",
        "order_time": "2026-05-30 14:00:00",
        "total_amount": 486,
        "order_items": [
            {"name": "无线耳机", "price": 299, "quantity": 1},
            {"name": "手机壳", "price": 49, "quantity": 2},
            {"name": "充电器", "price": 89, "quantity": 1}
        ],
        "shipping_address": "北京市朝阳区xxx路xxx号",
        "tracking_url": "https://example.com/track/ORD-2026-001"
    }

    sample_rendered = None
    sample_preview = None

    if not meta.get("is_base", False):
        try:
            env = _create_env(keep_placeholder=False, default_value="[示例]")
            template = env.get_template(template_name)
            sample_rendered = template.render(**sample_variables)
            sample_preview = _extract_preview(sample_rendered)
        except Exception:
            sample_rendered = None
            sample_preview = None

    return TemplatePreviewResponse(
        name=template_name,
        display_name=meta["name"],
        description=meta["description"],
        variables=meta["variables"],
        extends=meta.get("extends"),
        sample_rendered=sample_rendered,
        sample_preview=sample_preview
    )


@app.get("/api/health", summary="健康检查")
async def health_check():
    return {"status": "healthy", "service": "email-template-renderer", "version": "2.0.0", "templates_loaded": len(BUILTIN_TEMPLATES)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
