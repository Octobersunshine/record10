"""
API Documentation Generator
从代码注释或 JSON Schema 生成 Markdown 格式的 API 文档
"""

import ast
import json
import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ApiParam:
    name: str
    type_: str = ""
    required: bool = False
    description: str = ""
    default: Optional[str] = None
    in_: str = "query"
    type_inferred: bool = False
    original_type: str = ""


@dataclass
class ApiResponse:
    name: str = ""
    type_: str = ""
    description: str = ""
    type_inferred: bool = False
    original_type: str = ""


@dataclass
class ApiEndpoint:
    method: str = ""
    path: str = ""
    title: str = ""
    description: str = ""
    params: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    success_example: str = ""
    error_example: str = ""
    tags: list = field(default_factory=list)


def _normalize_type(raw: str) -> str:
    mapping = {
        "str": "string", "string": "string",
        "int": "integer", "integer": "integer", "long": "integer",
        "float": "number", "number": "number", "double": "number", "decimal": "number",
        "bool": "boolean", "boolean": "boolean",
        "list": "array", "array": "array",
        "dict": "object", "object": "object",
    }
    return mapping.get(raw.lower().strip(), raw.lower().strip())


def parse_comment_doc(source: str) -> list:
    endpoints = []
    current = None
    current_section = None
    example_buffer = []

    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        cleaned = re.sub(r"^[#/]+\s*", "", stripped)
        use = cleaned if cleaned.startswith("@api") else stripped

        if use.startswith("@api "):
            if current:
                _flush_example(current, current_section, example_buffer)
                endpoints.append(current)
            current = ApiEndpoint()
            current_section = None
            example_buffer = []
            m = re.match(
                r"@api\s+\{(\w+)\}\s+(\S+)\s+(.*)", use
            )
            if m:
                current.method = m.group(1).upper()
                current.path = m.group(2)
                current.title = m.group(3).strip()

        elif current and use.startswith("@apiParam"):
            _flush_example(current, current_section, example_buffer)
            current_section = "param"
            current.params.append(_parse_param_line(use))

        elif current and use.startswith("@apiSuccessExample"):
            _flush_example(current, current_section, example_buffer)
            current_section = "success_example"
            example_buffer = []

        elif current and use.startswith("@apiErrorExample"):
            _flush_example(current, current_section, example_buffer)
            current_section = "error_example"
            example_buffer = []

        elif current and use.startswith("@apiSuccess"):
            _flush_example(current, current_section, example_buffer)
            current_section = "success"
            current.responses.append(_parse_response_line(use))

        elif current and use.startswith("@apiDescription"):
            _flush_example(current, current_section, example_buffer)
            current_section = None
            m = re.match(r"@apiDescription\s+(.*)", use)
            if m:
                current.description = m.group(1).strip()

        elif current and use.startswith("@apiName"):
            m = re.match(r"@apiName\s+(.*)", use)
            if m:
                current.title = m.group(1).strip() or current.title

        elif current and use.startswith("@apiGroup"):
            m = re.match(r"@apiGroup\s+(.*)", use)
            if m:
                current.tags.append(m.group(1).strip())

        elif current and current_section and current_section.endswith("_example"):
            example_buffer.append(cleaned if cleaned else stripped)

    if current:
        _flush_example(current, current_section, example_buffer)
        endpoints.append(current)

    return endpoints


def _flush_example(endpoint: ApiEndpoint, section: str, buffer: list):
    if not buffer:
        return
    text = "\n".join(buffer).rstrip()
    if section == "success_example":
        endpoint.success_example = text
    elif section == "error_example":
        endpoint.error_example = text


def _parse_param_line(line: str) -> ApiParam:
    m = re.match(
        r"@apiParam\s+\{(\w+)\}\s+\[([^\]]+)\]\s+(.*)",
        line,
    )
    if m:
        p = ApiParam(
            type_=m.group(1),
            name=m.group(2).split("=")[0].strip(),
            required=False,
            description=m.group(3).strip(),
        )
        eq = m.group(2).split("=")
        if len(eq) > 1:
            p.default = eq[1].strip()
        return p

    m = re.match(
        r"@apiParam\s+\{(\w+)\}\s+(\S+)\s+(.*)",
        line,
    )
    if m:
        return ApiParam(
            type_=m.group(1),
            name=m.group(2),
            required=True,
            description=m.group(3).strip(),
        )

    m = re.match(r"@apiParam\s+\[(\S+)\]\s+(.*)", line)
    if m:
        p = ApiParam(name=m.group(1).split("=")[0].strip(), required=False, description=m.group(2).strip())
        eq = m.group(1).split("=")
        if len(eq) > 1:
            p.default = eq[1].strip()
        return p

    m = re.match(r"@apiParam\s+(\S+)\s+(.*)", line)
    if m:
        return ApiParam(name=m.group(1), required=True, description=m.group(2).strip())

    return ApiParam()


def _parse_response_line(line: str) -> ApiResponse:
    m = re.match(
        r"@apiSuccess\s+\{(\w+)\}\s+(\S+)\s+(.*)",
        line,
    )
    if m:
        return ApiResponse(type_=m.group(1), name=m.group(2), description=m.group(3).strip())

    m = re.match(r"@apiSuccess\s+(\S+)\s+(.*)", line)
    if m:
        return ApiResponse(name=m.group(1), description=m.group(2).strip())

    return ApiResponse()


def parse_python_docstrings(source: str) -> list:
    endpoints = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return endpoints

    route_re = re.compile(r"@?(?:app|router)\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]")

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        method = ""
        path = ""
        for deco in node.decorator_list:
            deco_src = ast.get_source_segment(source, deco) or ""
            m = route_re.match(deco_src)
            if m:
                method = m.group(1).upper()
                path = m.group(2)
                break

        if not method:
            continue

        docstring = ast.get_docstring(node) or ""
        ep = _parse_function_docstring(docstring, method, path)
        endpoints.append(ep)

    return endpoints


def _parse_function_docstring(docstring: str, method: str, path: str) -> ApiEndpoint:
    ep = ApiEndpoint(method=method, path=path)
    lines = docstring.splitlines()
    section = "desc"
    desc_lines = []
    param_lines = []
    resp_lines = []
    return_lines = []
    example_buffer = []
    example_type = ""

    for line in lines:
        s = line.strip()
        lower = s.lower()

        if lower.startswith("args:") or lower.startswith("参数:"):
            section = "args"
            continue
        if lower.startswith("returns:") or lower.startswith("返回:") or lower.startswith("return:"):
            section = "returns"
            continue
        if lower.startswith("example:") or lower.startswith("示例:") or lower.startswith("返回示例:"):
            section = "example"
            example_buffer = []
            continue

        if section == "desc":
            desc_lines.append(s)
        elif section == "args":
            param_lines.append(s)
        elif section == "returns":
            resp_lines.append(s)
        elif section == "example":
            example_buffer.append(line)

    ep.title = " ".join(desc_lines[0:1])
    ep.description = "\n".join(desc_lines[1:]).strip() if len(desc_lines) > 1 else ""
    if not ep.title and desc_lines:
        ep.title = desc_lines[0]

    for pl in param_lines:
        if not pl:
            continue
        m = re.match(r"(\w+)\s*[\((]\s*([^)\]]*?)\s*[)\]]\s*[:-]\s*(.*)", pl)
        if m:
            raw_type = m.group(2).strip()
            base_type = re.split(r"[,\s]+", raw_type, maxsplit=1)[0].strip()
            p = ApiParam(
                name=m.group(1),
                type_=_normalize_type(base_type),
                description=m.group(3).strip(),
            )
            if "required" in m.group(2).lower() or "必填" in m.group(2):
                p.required = True
            if "optional" in m.group(2).lower() or "可选" in m.group(2):
                p.required = False
            ep.params.append(p)
            continue
        m = re.match(r"(\w+)\s*[:-]\s*(.*)", pl)
        if m:
            ep.params.append(ApiParam(name=m.group(1), description=m.group(2).strip()))

    for rl in resp_lines:
        if not rl:
            continue
        m = re.match(r"(\S+)\s*[\((]\s*([^)\]]*?)\s*[)\]]\s*[:-]\s*(.*)", rl)
        if m:
            base_type = re.split(r"[,\s]+", m.group(2).strip(), maxsplit=1)[0].strip()
            ep.responses.append(ApiResponse(name=m.group(1), type_=_normalize_type(base_type), description=m.group(3).strip()))
            continue
        m = re.match(r"(\S+)\s*[:-]\s*(.*)", rl)
        if m:
            ep.responses.append(ApiResponse(name=m.group(1), description=m.group(2).strip()))

    if example_buffer:
        ep.success_example = "\n".join(example_buffer).rstrip()

    return ep


def parse_openapi_schema(data: dict) -> list:
    endpoints = []
    paths = data.get("paths", {})

    for path, methods in paths.items():
        for method, info in methods.items():
            if method.lower() in ("summary", "description", "parameters"):
                continue
            ep = ApiEndpoint(
                method=method.upper(),
                path=path,
                title=info.get("summary", ""),
                description=info.get("description", ""),
                tags=info.get("tags", []),
            )

            for param in info.get("parameters", []):
                schema = param.get("schema", {})
                p = ApiParam(
                    name=param.get("name", ""),
                    type_=schema.get("type", param.get("type", "")),
                    required=param.get("required", False),
                    description=param.get("description", ""),
                    default=str(schema.get("default")) if "default" in schema else None,
                    in_=param.get("in", "query"),
                )
                if "enum" in schema:
                    p.description += f" (可选值: {', '.join(str(v) for v in schema['enum'])})"
                ep.params.append(p)

            request_body = info.get("requestBody", {})
            if request_body:
                content = request_body.get("content", {})
                for ct, ct_info in content.items():
                    schema = ct_info.get("schema", {})
                    if "properties" in schema:
                        for pname, pinfo in schema["properties"].items():
                            p = ApiParam(
                                name=pname,
                                type_=pinfo.get("type", ""),
                                required=pname in schema.get("required", []),
                                description=pinfo.get("description", ""),
                                in_="body",
                            )
                            ep.params.append(p)

            responses_info = info.get("responses", {})
            for code, resp in responses_info.items():
                desc = resp.get("description", "")
                ep.responses.append(ApiResponse(name=code, description=desc))

                content = resp.get("content", {})
                for ct, ct_info in content.items():
                    example = ct_info.get("example")
                    if example:
                        if code.startswith("2"):
                            ep.success_example = f"HTTP/1.1 {code}\n{json.dumps(example, ensure_ascii=False, indent=2)}"
                        else:
                            ep.error_example = f"HTTP/1.1 {code}\n{json.dumps(example, ensure_ascii=False, indent=2)}"

                    schema = ct_info.get("schema", {})
                    if "properties" in schema:
                        for rname, rinfo in schema["properties"].items():
                            ep.responses.append(
                                ApiResponse(
                                    name=f"{code}.{rname}",
                                    type_=rinfo.get("type", ""),
                                    description=rinfo.get("description", ""),
                                )
                            )

            endpoints.append(ep)

    return endpoints


def render_markdown(endpoints: list, title: str = "API Documentation") -> str:
    sections = []

    sections.append(f"# {title}\n")

    grouped = {}
    for ep in endpoints:
        tag = ep.tags[0] if ep.tags else "默认"
        grouped.setdefault(tag, []).append(ep)

    sections.append("## 目录\n")
    for tag, eps in grouped.items():
        sections.append(f"### {tag}\n")
        for ep in eps:
            anchor = _make_anchor(ep.method, ep.path)
            sections.append(f"- [{ep.method} {ep.path}](#{anchor}) — {ep.title}")
        sections.append("")

    for tag, eps in grouped.items():
        sections.append(f"## {tag}\n")
        for ep in eps:
            sections.append(_render_endpoint(ep))
            sections.append("")

    return "\n".join(sections)


def _make_anchor(method: str, path: str) -> str:
    raw = f"{method}-{path}"
    return re.sub(r"[^a-zA-Z0-9_-]", "", raw.replace("/", "-")).lower()


def _format_type_cell(type_: str, inferred: bool, original: str) -> str:
    if not inferred:
        return type_
    if original:
        return f"{type_} ⚙ *(原: {original})*"
    return f"{type_} ⚙"


def _render_endpoint(ep: ApiEndpoint) -> str:
    lines = []

    lines.append(f"### {ep.method} `{ep.path}`\n")
    if ep.title:
        lines.append(f"**{ep.title}**\n")
    if ep.description:
        lines.append(f"{ep.description}\n")

    lines.append(f"- **方法**: {ep.method}")
    lines.append(f"- **路径**: `{ep.path}`")
    lines.append("")

    if ep.params:
        lines.append("#### 参数\n")
        has_inferred = any(p.type_inferred for p in ep.params)
        if has_inferred:
            lines.append("| 参数名 | 类型 | 必填 | 位置 | 默认值 | 说明 |")
            lines.append("|--------|------|------|------|--------|------|")
            for p in ep.params:
                req = "是" if p.required else "否"
                default = p.default or "-"
                type_display = _format_type_cell(p.type_, p.type_inferred, p.original_type)
                lines.append(
                    f"| `{p.name}` | {type_display} | {req} | {p.in_} | {default} | {p.description} |"
                )
        else:
            lines.append("| 参数名 | 类型 | 必填 | 位置 | 默认值 | 说明 |")
            lines.append("|--------|------|------|------|--------|------|")
            for p in ep.params:
                req = "是" if p.required else "否"
                default = p.default or "-"
                lines.append(
                    f"| `{p.name}` | {p.type_} | {req} | {p.in_} | {default} | {p.description} |"
                )
        lines.append("")
        if has_inferred:
            lines.append("> *标记 ⚙ 的类型为从代码/示例自动推断，可能与原始标注不同*")
            lines.append("")

    if ep.responses:
        lines.append("#### 返回字段\n")
        has_inferred = any(r.type_inferred for r in ep.responses)
        lines.append("| 字段 | 类型 | 说明 |")
        lines.append("|------|------|------|")
        for r in ep.responses:
            type_display = _format_type_cell(r.type_, r.type_inferred, r.original_type)
            lines.append(f"| `{r.name}` | {type_display} | {r.description} |")
        if has_inferred:
            lines.append("")
            lines.append("> *标记 ⚙ 的类型为从代码/示例自动推断，可能与原始标注不同*")
        lines.append("")

    if ep.success_example:
        lines.append("#### 成功响应示例\n")
        lines.append("```json")
        code = _extract_json(ep.success_example)
        lines.append(code)
        lines.append("```\n")

    if ep.error_example:
        lines.append("#### 错误响应示例\n")
        lines.append("```json")
        code = _extract_json(ep.error_example)
        lines.append(code)
        lines.append("```\n")

    return "\n".join(lines)


def _extract_json(text: str) -> str:
    json_start = text.find("{")
    json_start2 = text.find("[")
    if json_start < 0:
        json_start = json_start2
    if json_start2 >= 0 and json_start2 < json_start:
        json_start = json_start2
    if json_start < 0:
        return text
    candidate = text[json_start:]
    try:
        obj = json.loads(candidate)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        pass
    depth = 0
    in_str = False
    end = -1
    for i, ch in enumerate(candidate):
        if ch == '"' and (i == 0 or candidate[i - 1] != "\\"):
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in ("{", "["):
            depth += 1
        elif ch in ("}", "]"):
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end > 0:
        candidate2 = candidate[:end]
        try:
            obj = json.loads(candidate2)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return candidate2
    return candidate


def _python_type_to_schema(py_type: str) -> str:
    mapping = {
        "str": "string", "int": "integer", "float": "number",
        "bool": "boolean", "list": "array", "dict": "object",
        "Optional": "", "Union": "", "Any": "",
    }
    return mapping.get(py_type, "")


def infer_types_from_ast(source: str, endpoints: list) -> list:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return endpoints

    func_map = {}
    ordered_funcs = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_map[node.name] = node
            ordered_funcs.append(node)

    route_re = re.compile(r"@?(?:app|router)\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]")

    api_comment_lines = {}
    for i, line in enumerate(source.splitlines()):
        cleaned = re.sub(r"^[#/]+\s*", "", line.strip())
        m = re.match(r"@api\s+\{(\w+)\}\s+(\S+)", cleaned)
        if m:
            api_comment_lines[(m.group(1).upper(), m.group(2))] = i

    for ep in endpoints:
        matched_node = None

        for raw_name, func_node in func_map.items():
            for deco in func_node.decorator_list:
                deco_src = ast.get_source_segment(source, deco) or ""
                m = route_re.match(deco_src)
                if m and m.group(1).upper() == ep.method and m.group(2) == ep.path:
                    matched_node = func_node
                    break
            if matched_node:
                break

        if not matched_node:
            key = (ep.method, ep.path)
            if key in api_comment_lines:
                comment_line = api_comment_lines[key]
                best_func = None
                best_dist = float("inf")
                for func_node in ordered_funcs:
                    if func_node.lineno > comment_line:
                        dist = func_node.lineno - comment_line
                        if dist < best_dist:
                            best_dist = dist
                            best_func = func_node
                if best_func and best_dist <= 30:
                    matched_node = best_func

        if not matched_node:
            path_parts = ep.path.strip("/").split("/")
            for part in reversed(path_parts):
                candidate = re.sub(r"[{}\-]", "", part).replace("-", "_")
                if candidate and candidate in func_map:
                    matched_node = func_map[candidate]
                    break

        if not matched_node:
            continue

        type_hints = _extract_type_hints(matched_node)
        isinstance_checks = _extract_isinstance_checks(matched_node, source)
        defaults = _extract_defaults(matched_node, source)
        comparisons = _extract_numeric_comparisons(matched_node, source)

        for param in ep.params:
            inferred = ""

            if param.name in type_hints:
                inferred = _python_type_to_schema(type_hints[param.name])

            if not inferred and param.name in isinstance_checks:
                inferred = isinstance_checks[param.name]

            if not inferred and param.name in comparisons:
                inferred = comparisons[param.name]

            if not inferred and param.name in defaults:
                inferred = _type_from_literal(defaults[param.name])

            if inferred:
                normalized_orig = _normalize_type(param.type_)
                normalized_inf = _normalize_type(inferred)
                if normalized_orig != normalized_inf and normalized_inf:
                    param.original_type = param.type_
                    param.type_ = normalized_inf
                    param.type_inferred = True
                elif not param.type_ and normalized_inf:
                    param.type_ = normalized_inf
                    param.type_inferred = True

        for resp in ep.responses:
            base_name = resp.name.split(".")[-1] if "." in resp.name else resp.name
            if base_name in type_hints and not resp.type_:
                schema_type = _python_type_to_schema(type_hints[base_name])
                if schema_type:
                    resp.type_ = schema_type
                    resp.type_inferred = True

    return endpoints


def _extract_type_hints(func_node) -> dict:
    hints = {}
    args = func_node.args
    for arg in args.args + args.posonlyargs + args.kwonlyargs:
        if arg.annotation:
            type_str = _ast_type_to_str(arg.annotation)
            if type_str:
                hints[arg.arg] = type_str
    if func_node.returns:
        ret = _ast_type_to_str(func_node.returns)
        if ret:
            hints["__return__"] = ret
    return hints


def _ast_type_to_str(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Attribute):
        val = _ast_type_to_str(node.value)
        return f"{val}.{node.attr}" if val else node.attr
    if isinstance(node, ast.Subscript):
        base = _ast_type_to_str(node.value)
        return base if base else ""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _ast_type_to_str(node.left)
        right = _ast_type_to_str(node.right)
        if left and right:
            return f"Union[{left}, {right}]"
        return left or right
    return ""


def _extract_isinstance_checks(func_node, source: str) -> dict:
    checks = {}
    python_to_schema = {
        "int": "integer", "float": "number", "str": "string",
        "bool": "boolean", "list": "array", "dict": "object",
        "List": "array", "Dict": "object",
    }

    for child in ast.walk(func_node):
        if not isinstance(child, ast.Call):
            continue
        if not (isinstance(child.func, ast.Name) and child.func.id == "isinstance"):
            continue
        if len(child.args) < 2:
            continue

        var_name = ""
        if isinstance(child.args[0], ast.Name):
            var_name = child.args[0].id
        elif isinstance(child.args[0], ast.Subscript) and isinstance(child.args[0].value, ast.Name):
            var_name = child.args[0].value.id

        if not var_name:
            continue

        type_name = ""
        type_arg = child.args[1]
        if isinstance(type_arg, ast.Name):
            type_name = type_arg.id
        elif isinstance(type_arg, ast.Attribute):
            type_name = type_arg.attr
        elif isinstance(type_arg, ast.Tuple):
            types = []
            for elt in type_arg.elts:
                if isinstance(elt, ast.Name):
                    types.append(elt.id)
                elif isinstance(elt, ast.Attribute):
                    types.append(elt.attr)
            type_name = types[0] if types else ""

        if type_name and type_name in python_to_schema:
            existing = checks.get(var_name)
            new_type = python_to_schema[type_name]
            if existing and existing != new_type:
                checks[var_name] = "number" if {existing, new_type} == {"integer", "number"} else existing
            else:
                checks[var_name] = new_type

    return checks


def _extract_defaults(func_node, source: str) -> dict:
    defaults = {}
    args = func_node.args

    all_args = args.args + args.posonlyargs + args.kwonlyargs
    kw_defaults_start = len(args.args) + len(args.posonlyargs) - len(args.defaults)
    for i, default in enumerate(args.defaults):
        arg_idx = kw_defaults_start + i
        if arg_idx < len(all_args):
            defaults[all_args[arg_idx].arg] = default

    for i, default in enumerate(args.kw_defaults):
        if default and i < len(args.kwonlyargs):
            defaults[args.kwonlyargs[i].arg] = default

    return defaults


def _type_from_literal(node) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "boolean"
        if isinstance(node.value, int):
            return "integer"
        if isinstance(node.value, float):
            return "number"
        if isinstance(node.value, str):
            return "string"
        if isinstance(node.value, (list, tuple)):
            return "array"
        if isinstance(node.value, dict):
            return "object"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _type_from_literal(node.operand)
        return inner
    if isinstance(node, ast.List):
        return "array"
    if isinstance(node, ast.Dict):
        return "object"
    if isinstance(node, (ast.NameConstant,)):
        return "boolean"
    return ""


def _extract_numeric_comparisons(func_node, source: str) -> dict:
    comparisons = {}
    for child in ast.walk(func_node):
        if not isinstance(child, ast.Compare):
            continue
        left = child.left
        if isinstance(left, ast.Name):
            var_name = left.id
        elif isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name):
            var_name = left.value.id
        else:
            continue

        has_numeric = False
        for comparator in child.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                has_numeric = True
                break
            if isinstance(comparator, ast.UnaryOp) and isinstance(comparator.op, ast.USub):
                if isinstance(comparator.operand, ast.Constant) and isinstance(comparator.operand.value, (int, float)):
                    has_numeric = True
                    break
            if isinstance(comparator, ast.Call):
                if isinstance(comparator.func, ast.Name) and comparator.func.id in ("int", "float", "len"):
                    has_numeric = True
                    break

        if has_numeric:
            if var_name not in comparisons:
                comparisons[var_name] = "integer"

    return comparisons


def infer_types_from_examples(endpoints: list) -> list:
    for ep in endpoints:
        if ep.success_example:
            example_data = _try_parse_json(ep.success_example)
            if example_data:
                for resp in ep.responses:
                    inferred = _infer_type_from_data(resp.name, example_data)
                    if inferred:
                        normalized_orig = _normalize_type(resp.type_)
                        normalized_inf = _normalize_type(inferred)
                        if normalized_orig != normalized_inf and normalized_inf:
                            resp.original_type = resp.type_
                            resp.type_ = normalized_inf
                            resp.type_inferred = True
                        elif not resp.type_ and normalized_inf:
                            resp.type_ = normalized_inf
                            resp.type_inferred = True

                for param in ep.params:
                    if param.name in example_data:
                        inferred = _json_value_type(example_data[param.name])
                        normalized_orig = _normalize_type(param.type_)
                        normalized_inf = _normalize_type(inferred)
                        if normalized_orig != normalized_inf and normalized_inf:
                            param.original_type = param.type_
                            param.type_ = normalized_inf
                            param.type_inferred = True

        if ep.error_example:
            error_data = _try_parse_json(ep.error_example)
            if error_data:
                for param in ep.params:
                    if param.name in error_data:
                        inferred = _json_value_type(error_data[param.name])
                        normalized_orig = _normalize_type(param.type_)
                        normalized_inf = _normalize_type(inferred)
                        if normalized_orig != normalized_inf and normalized_inf:
                            param.original_type = param.type_
                            param.type_ = normalized_inf
                            param.type_inferred = True

    return endpoints


def _try_parse_json(text: str):
    json_start = text.find("{")
    json_start2 = text.find("[")
    if json_start < 0 and json_start2 < 0:
        return None
    if json_start < 0:
        json_start = json_start2
    if json_start2 >= 0 and json_start2 < json_start:
        json_start = json_start2

    candidate = text[json_start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    depth = 0
    in_str = False
    end = -1
    for i, ch in enumerate(candidate):
        if ch == '"' and (i == 0 or candidate[i - 1] != "\\"):
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in ("{", "["):
            depth += 1
        elif ch in ("}", "]"):
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end > 0:
        try:
            return json.loads(candidate[:end])
        except json.JSONDecodeError:
            return None
    return None


def _infer_type_from_data(name_path: str, data) -> str:
    parts = name_path.split(".")
    if parts[0].isdigit():
        if len(parts) > 1:
            return _infer_type_from_data(".".join(parts[1:]), data)
        return ""

    current = data
    for part in parts:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return ""
        elif isinstance(current, list) and len(current) > 0:
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                current = current[0]
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return ""
        else:
            return ""

    return _json_value_type(current)


def _json_value_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return ""


def render_html(endpoints: list, title: str = "API Documentation", base_url: str = "http://localhost:8000") -> str:
    grouped = {}
    for ep in endpoints:
        tag = ep.tags[0] if ep.tags else "默认"
        grouped.setdefault(tag, []).append(ep)

    endpoints_json = json.dumps(
        [_endpoint_to_debug_dict(ep) for ep in endpoints], ensure_ascii=False
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f7fa;color:#333;display:flex;min-height:100vh}}
.sidebar{{width:260px;background:#1a1a2e;color:#fff;padding:20px 0;position:fixed;top:0;bottom:0;overflow-y:auto}}
.sidebar h2{{padding:0 20px 16px;font-size:18px;border-bottom:1px solid rgba(255,255,255,.1)}}
.sidebar .group{{margin-top:12px}}
.sidebar .group-title{{padding:8px 20px;font-size:13px;color:#8b8fa3;text-transform:uppercase;letter-spacing:.5px}}
.sidebar a{{display:block;padding:8px 20px 8px 28px;color:#c3c7d4;text-decoration:none;font-size:13px;transition:all .2s}}
.sidebar a:hover,.sidebar a.active{{background:#16213e;color:#fff}}
.sidebar a .method{{display:inline-block;width:50px;font-size:11px;font-weight:700;padding:2px 6px;border-radius:3px;text-align:center;margin-right:8px}}
.sidebar a .method.get{{background:#61affe;color:#fff}}
.sidebar a .method.post{{background:#49cc90;color:#fff}}
.sidebar a .method.put{{background:#fca130;color:#fff}}
.sidebar a .method.delete{{background:#f93e3e;color:#fff}}
.sidebar a .method.patch{{background:#50e3c2;color:#1a1a2e}}
.main{{margin-left:260px;flex:1;padding:30px 40px;max-width:900px}}
.ep-card{{background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px;overflow:hidden}}
.ep-header{{padding:16px 24px;border-bottom:1px solid #eee;display:flex;align-items:center;gap:12px}}
.ep-header .badge{{padding:4px 10px;border-radius:4px;font-size:12px;font-weight:700;color:#fff}}
.ep-header .badge.get{{background:#61affe}}.ep-header .badge.post{{background:#49cc90}}
.ep-header .badge.put{{background:#fca130}}.ep-header .badge.delete{{background:#f93e3e}}
.ep-header .badge.patch{{background:#50e3c2;color:#1a1a2e}}
.ep-header .path{{font-family:monospace;font-size:15px;font-weight:600}}
.ep-header .ep-title{{color:#666;font-size:14px;margin-left:auto}}
.ep-body{{padding:20px 24px}}
.ep-desc{{color:#666;font-size:13px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}}
th{{background:#f8f9fb;text-align:left;padding:8px 12px;font-weight:600;border-bottom:2px solid #eee}}
td{{padding:8px 12px;border-bottom:1px solid #f0f0f0}}
td code{{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:12px}}
.debug-section{{margin-top:16px;padding:16px;background:#f8f9fb;border-radius:6px}}
.debug-section h4{{margin-bottom:10px;font-size:14px;color:#333}}
.debug-row{{display:flex;gap:8px;margin-bottom:8px;align-items:center}}
.debug-row label{{min-width:100px;font-size:12px;font-weight:600;color:#555}}
.debug-row input,.debug-row select,.debug-row textarea{{flex:1;padding:6px 10px;border:1px solid #ddd;border-radius:4px;font-size:13px;font-family:inherit}}
.debug-row textarea{{min-height:60px;font-family:monospace}}
.btn{{padding:8px 20px;border:none;border-radius:4px;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s}}
.btn-send{{background:#49cc90;color:#fff}}.btn-send:hover{{background:#3baa7b}}
.btn-send:disabled{{background:#ccc;cursor:not-allowed}}
.response-box{{margin-top:12px;background:#1a1a2e;border-radius:6px;padding:16px;max-height:400px;overflow:auto}}
.response-box pre{{color:#a5d6ff;font-size:13px;font-family:'Fira Code',monospace;white-space:pre-wrap}}
.response-box .status{{font-weight:700;margin-bottom:8px}}
.response-box .status.ok{{color:#49cc90}}.response-box .status.err{{color:#f93e3e}}
.inferred{{color:#e67e22;font-size:11px}}.inferred::before{{content:'⚙ '}}
.example-box{{background:#2d2d2d;color:#a5d6ff;padding:12px 16px;border-radius:6px;margin:8px 0;font-size:12px;font-family:monospace;white-space:pre-wrap;overflow-x:auto}}
.section-title{{font-size:14px;font-weight:700;margin:16px 0 8px;color:#333}}
</style>
</head>
<body>
<div class="sidebar">
<h2>{_esc(title)}</h2>
{_render_sidebar(grouped)}
</div>
<div class="main" id="main">
{_render_ep_cards(endpoints)}
</div>
<script>
const ENDPOINTS = {endpoints_json};
const BASE_URL = '{_esc(base_url)}';
function sendRequest(epIdx){{
  const ep = ENDPOINTS[epIdx];
  const card = document.getElementById('debug-' + epIdx);
  const urlInput = card.querySelector('.debug-url');
  const bodyInput = card.querySelector('.debug-body');
  const respBox = card.querySelector('.response-box');
  const btn = card.querySelector('.btn-send');
  btn.disabled = true;
  btn.textContent = '发送中...';
  let url = urlInput.value;
  const options = {{method: ep.method, headers: {{}}}};
  if (['POST','PUT','PATCH'].includes(ep.method) && bodyInput && bodyInput.value.trim()){{
    try{{ options.body = JSON.stringify(JSON.parse(bodyInput.value)); options.headers['Content-Type']='application/json'; }}
    catch(e){{ options.body = bodyInput.value; }}
  }}
  const startTime = performance.now();
  fetch(url, options).then(r=>{{const elapsed=(performance.now()-startTime).toFixed(0);const ct=r.headers.get('content-type')||'';return r.text().then(t=>({{status:r.status,statusText:r.statusText,ok:r.ok,body:t,isJson:ct.includes('json'),elapsed}}))}})
  .then(d=>{{let body=d.isJson?JSON.stringify(JSON.parse(d.body),null,2):d.body;const cls=d.ok?'ok':'err';respBox.style.display='block';respBox.innerHTML='<div class="status '+cls+'">HTTP '+d.status+' '+d.statusText+' ('+d.elapsed+'ms)</div><pre>'+_escHtml(body)+'</pre>';}})
  .catch(e=>{{respBox.style.display='block';respBox.innerHTML='<div class="status err">Error: '+_escHtml(e.message)+'</div>';}})
  .finally(()=>{{btn.disabled=false;btn.textContent='发送请求';}});
}}
function _escHtml(s){{const d=document.createElement('div');d.textContent=s;return d.innerHTML;}}
document.querySelectorAll('.sidebar a').forEach(a=>a.addEventListener('click',function(e){{e.preventDefault();document.querySelectorAll('.sidebar a').forEach(x=>x.classList.remove('active'));this.classList.add('active');const id=this.getAttribute('href').slice(1);document.getElementById(id)?.scrollIntoView({{behavior:'smooth',block:'start'}});}}));
</script>
</body>
</html>"""
    return html


def _endpoint_to_debug_dict(ep: ApiEndpoint) -> dict:
    return {
        "method": ep.method,
        "path": ep.path,
        "title": ep.title,
        "description": ep.description,
        "params": [
            {"name": p.name, "type": p.type_, "required": p.required,
             "description": p.description, "default": p.default, "in": p.in_,
             "type_inferred": p.type_inferred, "original_type": p.original_type}
            for p in ep.params
        ],
        "success_example": ep.success_example,
        "error_example": ep.error_example,
        "tags": ep.tags,
    }


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_sidebar(grouped: dict) -> str:
    parts = []
    for tag, eps in grouped.items():
        parts.append(f'<div class="group"><div class="group-title">{_esc(tag)}</div>')
        for ep in eps:
            anchor = _make_anchor(ep.method, ep.path)
            mcls = ep.method.lower()
            parts.append(f'<a href="#{anchor}"><span class="method {mcls}">{ep.method}</span>{_esc(ep.path)}</a>')
        parts.append("</div>")
    return "\n".join(parts)


BASE_URL_DEFAULT = "http://localhost:8000"


def _render_ep_cards(endpoints: list) -> str:
    parts = []
    for idx, ep in enumerate(endpoints):
        anchor = _make_anchor(ep.method, ep.path)
        mcls = ep.method.lower()
        card = f'<div class="ep-card" id="{anchor}">\n'
        card += f'<div class="ep-header"><span class="badge {mcls}">{ep.method}</span><span class="path">{_esc(ep.path)}</span>'
        if ep.title:
            card += f'<span class="ep-title">{_esc(ep.title)}</span>'
        card += '</div>\n<div class="ep-body">\n'
        if ep.description:
            card += f'<div class="ep-desc">{_esc(ep.description)}</div>\n'
        if ep.params:
            card += '<div class="section-title">参数</div>\n<table><tr><th>参数名</th><th>类型</th><th>必填</th><th>位置</th><th>默认值</th><th>说明</th></tr>\n'
            for p in ep.params:
                req = "是" if p.required else "否"
                default = p.default or "-"
                type_display = _esc(p.type_)
                if p.type_inferred:
                    orig = f" (原: {_esc(p.original_type)})" if p.original_type else ""
                    type_display = f'<span class="inferred">{_esc(p.type_)}</span>{_esc(orig)}'
                card += f'<tr><td><code>{_esc(p.name)}</code></td><td>{type_display}</td><td>{req}</td><td>{p.in_}</td><td>{_esc(default)}</td><td>{_esc(p.description)}</td></tr>\n'
            card += '</table>\n'
        if ep.responses:
            card += '<div class="section-title">返回字段</div>\n<table><tr><th>字段</th><th>类型</th><th>说明</th></tr>\n'
            for r in ep.responses:
                type_display = _esc(r.type_)
                if r.type_inferred:
                    orig = f" (原: {_esc(r.original_type)})" if r.original_type else ""
                    type_display = f'<span class="inferred">{_esc(r.type_)}</span>{_esc(orig)}'
                card += f'<tr><td><code>{_esc(r.name)}</code></td><td>{type_display}</td><td>{_esc(r.description)}</td></tr>\n'
            card += '</table>\n'
        if ep.success_example:
            card += '<div class="section-title">成功响应示例</div>\n<div class="example-box">'
            card += _esc(_extract_json(ep.success_example))
            card += '</div>\n'
        if ep.error_example:
            card += '<div class="section-title">错误响应示例</div>\n<div class="example-box">'
            card += _esc(_extract_json(ep.error_example))
            card += '</div>\n'
        card += _render_debug_section(ep, idx)
        card += '</div>\n</div>\n'
        parts.append(card)
    return "\n".join(parts)


def _render_debug_section(ep: ApiEndpoint, idx: int) -> str:
    path_params = [p for p in ep.params if p.in_ == "path"]
    query_params = [p for p in ep.params if p.in_ == "query"]
    body_params = [p for p in ep.params if p.in_ == "body"]
    has_body = ep.method in ("POST", "PUT", "PATCH") or bool(body_params)

    url = ep.path
    for p in path_params:
        val = p.default or f"{{{p.name}}}"
        url = url.replace("{" + p.name + "}", val)
    if query_params:
        qs = "&".join(f"{p.name}={p.default or ''}" for p in query_params)
        url += f"?{qs}"

    s = f'<div class="debug-section" id="debug-{idx}"><h4>🚀 在线调试</h4>\n'
    s += f'<div class="debug-row"><label>请求 URL</label><input class="debug-url" value="{_esc(BASE_URL_DEFAULT)}{_esc(url)}" style="flex:1"></div>\n'
    if has_body:
        body_json = "{}"
        if body_params:
            body_obj = {}
            for p in body_params:
                if p.default:
                    body_obj[p.name] = p.default
                elif p.type_ in ("integer", "number"):
                    body_obj[p.name] = 0
                elif p.type_ == "boolean":
                    body_obj[p.name] = True
                else:
                    body_obj[p.name] = ""
            body_json = json.dumps(body_obj, ensure_ascii=False, indent=2)
        s += f'<div class="debug-row"><label>请求体</label><textarea class="debug-body">{_esc(body_json)}</textarea></div>\n'
    s += f'<button class="btn btn-send" onclick="sendRequest({idx})">发送请求</button>\n'
    s += f'<div class="response-box" style="display:none"></div>\n'
    s += '</div>\n'
    return s


def version_save(endpoints: list, version_dir: str, version_name: str, title: str = "API Documentation") -> str:
    vdir = Path(version_dir)
    vdir.mkdir(parents=True, exist_ok=True)

    if not version_name:
        from datetime import datetime
        version_name = datetime.now().strftime("v%Y%m%d_%H%M%S")

    snapshot = {
        "version": version_name,
        "title": title,
        "timestamp": _now_iso(),
        "endpoints": [_endpoint_to_debug_dict(ep) for ep in endpoints],
    }

    snapshot_file = vdir / f"{version_name}.json"
    snapshot_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(snapshot_file)


def version_list(version_dir: str) -> list:
    vdir = Path(version_dir)
    if not vdir.exists():
        return []

    versions = []
    for f in sorted(vdir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            versions.append({
                "version": data.get("version", f.stem),
                "title": data.get("title", ""),
                "timestamp": data.get("timestamp", ""),
                "file": str(f),
                "endpoint_count": len(data.get("endpoints", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return versions


def version_diff(version_dir: str, v1_name: str, v2_name: str) -> str:
    vdir = Path(version_dir)
    f1 = vdir / f"{v1_name}.json"
    f2 = vdir / f"{v2_name}.json"

    if not f1.exists():
        return f"[错误] 版本不存在: {v1_name}"
    if not f2.exists():
        return f"[错误] 版本不存在: {v2_name}"

    d1 = json.loads(f1.read_text(encoding="utf-8"))
    d2 = json.loads(f2.read_text(encoding="utf-8"))

    eps1 = {f"{e['method']} {e['path']}": e for e in d1.get("endpoints", [])}
    eps2 = {f"{e['method']} {e['path']}": e for e in d2.get("endpoints", [])}

    keys1 = set(eps1.keys())
    keys2 = set(eps2.keys())

    added = sorted(keys2 - keys1)
    removed = sorted(keys1 - keys2)
    common = sorted(keys1 & keys2)

    lines = []
    lines.append(f"# API 版本对比: {v1_name} → {v2_name}\n")

    if added:
        lines.append(f"## 新增接口 ({len(added)})\n")
        for k in added:
            ep = eps2[k]
            lines.append(f"- **{k}** — {ep.get('title', '')}")
            for p in ep.get("params", []):
                lines.append(f"  - 参数 `{p['name']}` ({p.get('type', '?')}, {p.get('in', '?')})")
        lines.append("")

    if removed:
        lines.append(f"## 删除接口 ({len(removed)})\n")
        for k in removed:
            ep = eps1[k]
            lines.append(f"- **{k}** — {ep.get('title', '')}")
        lines.append("")

    changed = []
    for k in common:
        changes = _diff_endpoint(eps1[k], eps2[k])
        if changes:
            changed.append((k, changes))

    if changed:
        lines.append(f"## 变更接口 ({len(changed)})\n")
        for k, changes in changed:
            lines.append(f"### {k}\n")
            for c in changes:
                lines.append(f"- {c}")
            lines.append("")

    if not added and not removed and not changed:
        lines.append("*两个版本无差异*")

    return "\n".join(lines)


def _diff_endpoint(ep1: dict, ep2: dict) -> list:
    changes = []

    if ep1.get("title") != ep2.get("title"):
        changes.append(f"标题: `{ep1.get('title', '')}` → `{ep2.get('title', '')}`")
    if ep1.get("description") != ep2.get("description"):
        changes.append("描述已变更")

    params1 = {p["name"]: p for p in ep1.get("params", [])}
    params2 = {p["name"]: p for p in ep2.get("params", [])}

    p_added = set(params2.keys()) - set(params1.keys())
    p_removed = set(params1.keys()) - set(params2.keys())

    for name in sorted(p_added):
        p = params2[name]
        changes.append(f"新增参数 `{name}` ({p.get('type', '?')}, {p.get('in', '?')})")
    for name in sorted(p_removed):
        changes.append(f"删除参数 `{name}`")

    for name in sorted(set(params1.keys()) & set(params2.keys())):
        p1, p2 = params1[name], params2[name]
        if p1.get("type") != p2.get("type"):
            changes.append(f"参数 `{name}` 类型: `{p1.get('type', '')}` → `{p2.get('type', '')}`")
        if p1.get("required") != p2.get("required"):
            r1 = "必填" if p1.get("required") else "可选"
            r2 = "必填" if p2.get("required") else "可选"
            changes.append(f"参数 `{name}` 必填: {r1} → {r2}")
        if p1.get("description") != p2.get("description"):
            changes.append(f"参数 `{name}` 说明已变更")

    return changes


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def export_openapi(endpoints: list, title: str = "API Documentation", version: str = "1.0.0", base_url: str = "") -> dict:
    spec = {
        "openapi": "3.0.3",
        "info": {"title": title, "version": version},
        "paths": {},
    }
    if base_url:
        spec["servers"] = [{"url": base_url}]

    tags_set = set()
    for ep in endpoints:
        if ep.tags:
            tags_set.update(ep.tags)

    if tags_set:
        spec["tags"] = [{"name": t} for t in sorted(tags_set)]

    for ep in endpoints:
        path_obj = spec["paths"].setdefault(ep.path, {})
        method_obj = {
            "summary": ep.title,
            "description": ep.description,
            "tags": ep.tags if ep.tags else [],
            "parameters": [],
            "responses": {},
        }

        body_params = [p for p in ep.params if p.in_ == "body"]
        non_body_params = [p for p in ep.params if p.in_ != "body"]

        for p in non_body_params:
            param_obj = {
                "name": p.name,
                "in": p.in_,
                "required": p.required,
                "description": p.description,
                "schema": _type_to_openapi_schema(p.type_),
            }
            if p.default is not None:
                param_obj["schema"]["default"] = _coerce_default(p.default, p.type_)
            method_obj["parameters"].append(param_obj)

        if body_params:
            properties = {}
            required_list = []
            for p in body_params:
                prop = _type_to_openapi_schema(p.type_)
                prop["description"] = p.description
                if p.default is not None:
                    prop["default"] = _coerce_default(p.default, p.type_)
                properties[p.name] = prop
                if p.required:
                    required_list.append(p.name)
            body_schema = {"type": "object", "properties": properties}
            if required_list:
                body_schema["required"] = required_list
            content_obj = {"application/json": {"schema": body_schema}}
            if ep.success_example:
                example_data = _try_parse_json(ep.success_example)
                if example_data:
                    content_obj["application/json"]["example"] = example_data
            method_obj["requestBody"] = {
                "required": True,
                "content": content_obj,
            }

        if not method_obj["parameters"]:
            del method_obj["parameters"]

        status_responses = {}
        field_responses = []
        for r in ep.responses:
            if r.name.isdigit() or (len(r.name) == 3 and r.name[0] in "2345"):
                status_responses[r.name] = r
            else:
                field_responses.append(r)

        if status_responses:
            for code, r in status_responses.items():
                resp_obj = {"description": r.description}
                resp_schema = _build_response_schema(r.name, field_responses, ep)
                if resp_schema:
                    resp_obj["content"] = {"application/json": {"schema": resp_schema}}
                    if code.startswith("2") and ep.success_example:
                        example_data = _try_parse_json(ep.success_example)
                        if example_data:
                            resp_obj["content"]["application/json"]["example"] = example_data
                    elif not code.startswith("2") and ep.error_example:
                        example_data = _try_parse_json(ep.error_example)
                        if example_data:
                            resp_obj["content"]["application/json"]["example"] = example_data
                method_obj["responses"][code] = resp_obj
        else:
            success_schema = _build_response_schema("200", field_responses, ep)
            resp_200 = {"description": "成功响应"}
            if success_schema:
                resp_200["content"] = {"application/json": {"schema": success_schema}}
                if ep.success_example:
                    example_data = _try_parse_json(ep.success_example)
                    if example_data:
                        resp_200["content"]["application/json"]["example"] = example_data
            method_obj["responses"]["200"] = resp_200
            if ep.error_example:
                example_data = _try_parse_json(ep.error_example)
                err_resp = {"description": "错误响应"}
                if example_data:
                    err_resp["content"] = {"application/json": {"schema": {"type": "object"}, "example": example_data}}
                method_obj["responses"]["400"] = err_resp

        path_obj[ep.method.lower()] = method_obj

    return spec


def _type_to_openapi_schema(type_: str) -> dict:
    type_map = {
        "string": "string", "integer": "integer", "number": "number",
        "boolean": "boolean", "array": "array", "object": "object",
        "String": "string", "Number": "number", "Array": "array",
        "Boolean": "boolean", "Object": "object",
    }
    oas_type = type_map.get(type_, "string")
    schema = {"type": oas_type}
    if type_ == "array":
        schema["items"] = {"type": "object"}
    return schema


def _coerce_default(value: str, type_: str) -> object:
    if type_ in ("integer", "int", "Number"):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if type_ in ("number", "float"):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if type_ in ("boolean", "bool"):
        return value.lower() in ("true", "1", "yes")
    return value


def _build_response_schema(status_code: str, field_responses: list, ep: ApiEndpoint) -> dict:
    if not field_responses:
        return {}
    prefix = f"{status_code}."
    relevant = [r for r in field_responses if r.name.startswith(prefix)]
    if not relevant:
        return {}
    properties = {}
    required = []
    for r in relevant:
        field_name = r.name[len(prefix):]
        if "." in field_name:
            parent = field_name.split(".")[0]
            if parent not in properties:
                properties[parent] = {"type": "object", "properties": {}}
            child_name = ".".join(field_name.split(".")[1:])
            properties[parent]["properties"][child_name] = _type_to_openapi_schema(r.type_)
        else:
            properties[field_name] = _type_to_openapi_schema(r.type_)
            if r.type_:
                properties[field_name]["description"] = r.description
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def detect_and_parse(filepath: str, source: str = None) -> list:
    p = Path(filepath)
    content = source if source is not None else p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    endpoints = []

    if suffix == ".json":
        data = json.loads(content)
        if "openapi" in data or "swagger" in data:
            endpoints = parse_openapi_schema(data)
        elif "paths" in data:
            endpoints = parse_openapi_schema(data)
        else:
            endpoints = _parse_simple_json_api(data)
    elif suffix == ".py":
        comment_eps = parse_comment_doc(content)
        docstring_eps = parse_python_docstrings(content)
        endpoints = comment_eps + docstring_eps
        endpoints = infer_types_from_ast(content, endpoints)
    else:
        comment_eps = parse_comment_doc(content)
        if comment_eps:
            endpoints = comment_eps

    endpoints = infer_types_from_examples(endpoints)

    return endpoints


def _parse_simple_json_api(data: dict) -> list:
    endpoints = []
    apis = data if isinstance(data, list) else data.get("apis", data.get("endpoints", []))
    if isinstance(apis, dict):
        apis = [apis]

    for api in apis:
        ep = ApiEndpoint(
            method=api.get("method", "GET").upper(),
            path=api.get("path", api.get("url", "")),
            title=api.get("title", api.get("name", "")),
            description=api.get("description", ""),
            tags=api.get("tags", api.get("group", [])),
        )

        if isinstance(ep.tags, str):
            ep.tags = [ep.tags]

        for param in api.get("params", api.get("parameters", [])):
            p = ApiParam(
                name=param.get("name", ""),
                type_=param.get("type", param.get("type_", "")),
                required=param.get("required", False),
                description=param.get("description", ""),
                default=str(param["default"]) if "default" in param else None,
                in_=param.get("in", param.get("location", "query")),
            )
            ep.params.append(p)

        for resp in api.get("responses", api.get("returns", api.get("return", []))):
            if isinstance(resp, dict):
                ep.responses.append(
                    ApiResponse(
                        name=resp.get("name", resp.get("field", "")),
                        type_=resp.get("type", resp.get("type_", "")),
                        description=resp.get("description", ""),
                    )
                )
            elif isinstance(resp, str):
                ep.responses.append(ApiResponse(name=resp))

        if "successExample" in api or "success_example" in api or "example" in api:
            ex = api.get("successExample", api.get("success_example", api.get("example", "")))
            if isinstance(ex, (dict, list)):
                ep.success_example = json.dumps(ex, ensure_ascii=False, indent=2)
            else:
                ep.success_example = str(ex)

        if "errorExample" in api or "error_example" in api:
            ex = api.get("errorExample", api.get("error_example", ""))
            if isinstance(ex, (dict, list)):
                ep.error_example = json.dumps(ex, ensure_ascii=False, indent=2)
            else:
                ep.error_example = str(ex)

        endpoints.append(ep)

    return endpoints


def main():
    parser = argparse.ArgumentParser(
        description="API 文档生成器 — 从代码注释或 JSON Schema 生成 Markdown/HTML/OpenAPI 格式的 API 文档"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    gen_parser = subparsers.add_parser("gen", help="生成 Markdown 文档（默认）")
    gen_parser.add_argument("inputs", nargs="+", help="输入文件路径")
    gen_parser.add_argument("-o", "--output", default="api_doc.md", help="输出路径")
    gen_parser.add_argument("-t", "--title", default="API Documentation", help="文档标题")

    html_parser = subparsers.add_parser("html", help="生成带在线调试的交互式 HTML 文档")
    html_parser.add_argument("inputs", nargs="+", help="输入文件路径")
    html_parser.add_argument("-o", "--output", default="api_doc.html", help="输出路径")
    html_parser.add_argument("-t", "--title", default="API Documentation", help="文档标题")
    html_parser.add_argument("--base-url", default="http://localhost:8000", help="API 基础 URL")

    openapi_parser = subparsers.add_parser("openapi", help="导出为 OpenAPI 3.0 JSON")
    openapi_parser.add_argument("inputs", nargs="+", help="输入文件路径")
    openapi_parser.add_argument("-o", "--output", default="openapi.json", help="输出路径")
    openapi_parser.add_argument("-t", "--title", default="API Documentation", help="API 标题")
    openapi_parser.add_argument("--api-version", default="1.0.0", help="API 版本号")
    openapi_parser.add_argument("--base-url", default="", help="服务器基础 URL")

    ver_parser = subparsers.add_parser("version", help="版本管理")
    ver_sub = ver_parser.add_subparsers(dest="version_cmd", help="版本操作")
    ver_save = ver_sub.add_parser("save", help="保存当前 API 快照")
    ver_save.add_argument("inputs", nargs="+", help="输入文件路径")
    ver_save.add_argument("-d", "--dir", default=".api_versions", help="版本存储目录")
    ver_save.add_argument("-n", "--name", default="", help="版本名称（默认自动生成）")
    ver_save.add_argument("-t", "--title", default="API Documentation", help="文档标题")
    ver_list = ver_sub.add_parser("list", help="列出所有已保存版本")
    ver_list.add_argument("-d", "--dir", default=".api_versions", help="版本存储目录")
    ver_diff = ver_sub.add_parser("diff", help="对比两个版本差异")
    ver_diff.add_argument("-d", "--dir", default=".api_versions", help="版本存储目录")
    ver_diff.add_argument("v1", help="旧版本名称")
    ver_diff.add_argument("v2", help="新版本名称")
    ver_diff.add_argument("-o", "--output", default="", help="输出到文件（默认打印到终端）")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "gen":
        _cmd_gen(args)
    elif args.command == "html":
        _cmd_html(args)
    elif args.command == "openapi":
        _cmd_openapi(args)
    elif args.command == "version":
        if not args.version_cmd:
            parser.parse_args(["version", "--help"])
        elif args.version_cmd == "save":
            _cmd_version_save(args)
        elif args.version_cmd == "list":
            _cmd_version_list(args)
        elif args.version_cmd == "diff":
            _cmd_version_diff(args)


def _parse_inputs(inputs: list) -> list:
    all_endpoints = []
    for filepath in inputs:
        p = Path(filepath)
        if not p.exists():
            print(f"[警告] 文件不存在，跳过: {filepath}", file=sys.stderr)
            continue
        print(f"[信息] 解析文件: {filepath}")
        eps = detect_and_parse(filepath)
        print(f"[信息]   发现 {len(eps)} 个接口")
        all_endpoints.extend(eps)
    return all_endpoints


def _cmd_gen(args):
    all_endpoints = _parse_inputs(args.inputs)
    if not all_endpoints:
        print("[错误] 未发现任何接口定义", file=sys.stderr)
        sys.exit(1)
    md = render_markdown(all_endpoints, title=args.title)
    out = Path(args.output)
    out.write_text(md, encoding="utf-8")
    print(f"[完成] 已生成 Markdown 文档: {out.absolute()} ({len(all_endpoints)} 个接口)")


def _cmd_html(args):
    all_endpoints = _parse_inputs(args.inputs)
    if not all_endpoints:
        print("[错误] 未发现任何接口定义", file=sys.stderr)
        sys.exit(1)
    html = render_html(all_endpoints, title=args.title, base_url=args.base_url)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"[完成] 已生成 HTML 文档（含在线调试）: {out.absolute()} ({len(all_endpoints)} 个接口)")


def _cmd_openapi(args):
    all_endpoints = _parse_inputs(args.inputs)
    if not all_endpoints:
        print("[错误] 未发现任何接口定义", file=sys.stderr)
        sys.exit(1)
    spec = export_openapi(all_endpoints, title=args.title, version=args.api_version, base_url=args.base_url)
    out = Path(args.output)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 已导出 OpenAPI 3.0 规范: {out.absolute()} ({len(all_endpoints)} 个接口)")


def _cmd_version_save(args):
    all_endpoints = _parse_inputs(args.inputs)
    if not all_endpoints:
        print("[错误] 未发现任何接口定义", file=sys.stderr)
        sys.exit(1)
    saved = version_save(all_endpoints, version_dir=args.dir, version_name=args.name, title=args.title)
    print(f"[完成] 已保存版本快照: {saved} ({len(all_endpoints)} 个接口)")


def _cmd_version_list(args):
    versions = version_list(args.dir)
    if not versions:
        print(f"[信息] 版本目录为空: {args.dir}")
        return
    print(f"已保存版本 (目录: {args.dir})\n")
    print(f"{'版本':<25} {'标题':<25} {'接口数':<8} {'时间'}")
    print("-" * 80)
    for v in versions:
        print(f"{v['version']:<25} {v['title']:<25} {v['endpoint_count']:<8} {v['timestamp']}")


def _cmd_version_diff(args):
    diff_md = version_diff(args.dir, args.v1, args.v2)
    if args.output:
        out = Path(args.output)
        out.write_text(diff_md, encoding="utf-8")
        print(f"[完成] 版本差异已输出: {out.absolute()}")
    else:
        print(diff_md)


if __name__ == "__main__":
    main()
