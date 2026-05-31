import warnings
import re
import math
from typing import List, Dict, Any, Optional, Tuple


CHINESE_COLOR_MAP = {
    "红色": "#FF0000", "浅红": "#FF6666", "深红": "#8B0000", "玫红": "#FF007F",
    "橙色": "#FFA500", "深橙": "#FF8C00", "浅橙": "#FFCC80",
    "黄色": "#FFFF00", "金黄": "#FFD700", "浅黄": "#FFFFE0", "深黄": "#FFB300",
    "绿色": "#008000", "浅绿": "#90EE90", "深绿": "#006400", "翠绿": "#00CC99",
    "青色": "#00FFFF", "深青": "#008B8B",
    "蓝色": "#0000FF", "浅蓝": "#ADD8E6", "深蓝": "#00008B", "天蓝": "#87CEEB",
    "紫色": "#800080", "浅紫": "#DDA0DD", "深紫": "#4B0082",
    "粉色": "#FFC0CB", "深粉": "#FF1493", "浅粉": "#FFE4E1",
    "棕色": "#A52A2A", "浅棕": "#D2B48C", "深棕": "#654321",
    "灰色": "#808080", "浅灰": "#D3D3D3", "深灰": "#A9A9A9",
    "黑色": "#000000", "白色": "#FFFFFF",
    "金色": "#FFD700", "银色": "#C0C0C0",
    "藏青": "#000080", "墨绿": "#003300", "酒红": "#722F37",
    "米色": "#F5F5DC", "象牙白": "#FFFFF0", "珊瑚色": "#FF7F50",
    "橄榄色": "#808000", "青柠": "#00FF00", "品红": "#FF00FF",
    "靛蓝": "#4B0082", "紫罗兰": "#EE82EE", "赭色": "#A0522D",
}

ENGLISH_COLOR_MAP = {
    "red": "#FF0000", "lightred": "#FF6666", "darkred": "#8B0000",
    "orange": "#FFA500", "yellow": "#FFFF00", "gold": "#FFD700",
    "green": "#008000", "lightgreen": "#90EE90", "darkgreen": "#006400",
    "cyan": "#00FFFF", "blue": "#0000FF", "lightblue": "#ADD8E6", "darkblue": "#00008B",
    "purple": "#800080", "pink": "#FFC0CB", "brown": "#A52A2A",
    "gray": "#808080", "grey": "#808080", "lightgray": "#D3D3D3", "darkgray": "#A9A9A9",
    "black": "#000000", "white": "#FFFFFF", "silver": "#C0C0C0",
    "navy": "#000080", "teal": "#008080", "maroon": "#800000",
    "olive": "#808000", "lime": "#00FF00", "aqua": "#00FFFF", "fuchsia": "#FF00FF",
    "coral": "#FF7F50", "salmon": "#FA8072", "khaki": "#F0E68C",
    "violet": "#EE82EE", "indigo": "#4B0082", "beige": "#F5F5DC",
    "ivory": "#FFFFF0", "tan": "#D2B48C", "sienna": "#A0522D",
    "plum": "#DDA0DD", "orchid": "#DA70D6", "peru": "#CD853F",
    "tomato": "#FF6347", "turquoise": "#40E0D0", "wheat": "#F5DEB3",
    "crimson": "#DC143C", "chocolate": "#D2691E", "firebrick": "#B22222",
}

CHART_PALETTE = [
    "#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#2980B9", "#C0392B", "#27AE60",
    "#8E44AD", "#16A085", "#D35400", "#2C3E50", "#F1C40F",
]


def normalize_color(color: str) -> str:
    if not color or color == "none":
        return color

    stripped = color.strip()

    if stripped in CHINESE_COLOR_MAP:
        return CHINESE_COLOR_MAP[stripped]

    lower = stripped.lower()
    if lower in ENGLISH_COLOR_MAP:
        return ENGLISH_COLOR_MAP[lower]

    if re.match(r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$', stripped):
        return stripped.upper()

    if re.match(r'^#([0-9a-fA-F]{3})([0-9a-fA-F]{3})$', stripped):
        r, g, b = stripped[1], stripped[2], stripped[3], stripped[4], stripped[5], stripped[6]
        return f"#{r}{g}{b}".upper()

    if lower.startswith('rgb'):
        match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', stripped)
        if match:
            r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return f"#{r:02X}{g:02X}{b:02X}"

    warnings.warn(f"Unrecognized color format: '{color}', using as-is. Consider using HEX format (#RRGGBB).", UserWarning)
    return stripped


def _check_bounds(shape_type: str, shape: Dict[str, Any], width: int, height: int) -> List[str]:
    msgs = []

    def _add(msg: str):
        msgs.append(f"[Boundary] {shape_type}: {msg} (canvas: {width}x{height})")

    if shape_type == "circle":
        cx, cy, r = shape.get("cx", 0), shape.get("cy", 0), shape.get("r", 0)
        if cx - r > width or cy - r > height or cx + r < 0 or cy + r < 0:
            _add(f"Circle at ({cx},{cy}) r={r} is completely outside canvas")
        elif cx - r < 0 or cy - r < 0 or cx + r > width or cy + r > height:
            _add(f"Circle at ({cx},{cy}) r={r} partially exceeds canvas")

    elif shape_type == "rect":
        x, y = shape.get("x", 0), shape.get("y", 0)
        w, h = shape.get("width", 0), shape.get("height", 0)
        if x > width or y > height or x + w < 0 or y + h < 0:
            _add(f"Rect at ({x},{y}) size ({w}x{h}) is completely outside canvas")
        elif x < 0 or y < 0 or x + w > width or y + h > height:
            _add(f"Rect at ({x},{y}) size ({w}x{h}) partially exceeds canvas")

    elif shape_type == "line":
        x1, y1 = shape.get("x1", 0), shape.get("y1", 0)
        x2, y2 = shape.get("x2", 0), shape.get("y2", 0)
        sw = shape.get("stroke_width", 1)
        if (min(x1, x2) - sw > width or max(x1, x2) + sw < 0 or
                min(y1, y2) - sw > height or max(y1, y2) + sw < 0):
            _add(f"Line ({x1},{y1})->({x2},{y2}) is completely outside canvas")
        elif (x1 < 0 or y1 < 0 or x2 > width or y2 > height or
              x2 < 0 or y2 < 0 or x1 > width or y1 > height):
            _add(f"Line ({x1},{y1})->({x2},{y2}) partially exceeds canvas")

    elif shape_type == "ellipse":
        cx, cy = shape.get("cx", 0), shape.get("cy", 0)
        rx, ry = shape.get("rx", 0), shape.get("ry", 0)
        if cx - rx > width or cy - ry > height or cx + rx < 0 or cy + ry < 0:
            _add(f"Ellipse at ({cx},{cy}) rx={rx} ry={ry} is completely outside canvas")
        elif cx - rx < 0 or cy - ry < 0 or cx + rx > width or cy + ry > height:
            _add(f"Ellipse at ({cx},{cy}) rx={rx} ry={ry} partially exceeds canvas")

    elif shape_type == "text":
        x, y = shape.get("x", 0), shape.get("y", 0)
        if x > width or y > height or x < 0 or y < 0:
            _add(f"Text at ({x},{y}) may be outside canvas (text bounds cannot be precisely calculated)")

    elif shape_type == "polygon":
        points_str = shape.get("points", "")
        if points_str:
            try:
                coords = [float(v) for v in re.split(r'[,\s]+', points_str.strip()) if v]
                xs = coords[0::2]
                ys = coords[1::2]
                if xs and ys:
                    if min(xs) > width or max(xs) < 0 or min(ys) > height or max(ys) < 0:
                        _add(f"Polygon is completely outside canvas")
                    elif min(xs) < 0 or max(xs) > width or min(ys) < 0 or max(ys) > height:
                        _add(f"Polygon partially exceeds canvas")
            except (ValueError, IndexError):
                pass

    elif shape_type == "path":
        d = shape.get("d", "")
        if d:
            nums = [float(v) for v in re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)]
            xs = nums[0::2]
            ys = nums[1::2]
            if xs and ys:
                if min(xs) > width or max(xs) < 0 or min(ys) > height or max(ys) < 0:
                    _add(f"Path may be completely outside canvas")
                elif min(xs) < 0 or max(xs) > width or min(ys) < 0 or max(ys) > height:
                    _add(f"Path may partially exceed canvas")

    return msgs


def generate_svg(
    shapes: List[Dict[str, Any]],
    width: int = 400,
    height: int = 300,
    background: str = "white",
    clip: bool = False,
    warn_bounds: bool = True,
) -> str:
    bg_hex = normalize_color(background) if background else None

    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']

    if clip:
        svg_parts.append(f'  <defs>')
        svg_parts.append(f'    <clipPath id="canvas-clip">')
        svg_parts.append(f'      <rect width="{width}" height="{height}"/>')
        svg_parts.append(f'    </clipPath>')
        svg_parts.append(f'  </defs>')

    if bg_hex:
        svg_parts.append(f'  <rect width="100%" height="100%" fill="{bg_hex}"/>')

    for idx, shape in enumerate(shapes):
        shape_type = shape.get("type", "").lower()

        if warn_bounds:
            boundary_warnings = _check_bounds(shape_type, shape, width, height)
            for msg in boundary_warnings:
                warnings.warn(msg, UserWarning)

        if shape_type == "circle":
            cx = shape.get("cx", 0)
            cy = shape.get("cy", 0)
            r = shape.get("r", 0)
            fill = normalize_color(shape.get("fill", "black"))
            stroke = normalize_color(shape.get("stroke", "none"))
            stroke_width = shape.get("stroke_width", 1)
            opacity = shape.get("opacity", 1.0)
            clip_attr = ' clip-path="url(#canvas-clip)"' if clip else ""
            svg_parts.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"{clip_attr}/>')

        elif shape_type == "rect":
            x = shape.get("x", 0)
            y = shape.get("y", 0)
            w = shape.get("width", 0)
            h = shape.get("height", 0)
            fill = normalize_color(shape.get("fill", "black"))
            stroke = normalize_color(shape.get("stroke", "none"))
            stroke_width = shape.get("stroke_width", 1)
            rx = shape.get("rx", 0)
            ry = shape.get("ry", 0)
            opacity = shape.get("opacity", 1.0)
            clip_attr = ' clip-path="url(#canvas-clip)"' if clip else ""
            svg_parts.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}" ry="{ry}" opacity="{opacity}"{clip_attr}/>')

        elif shape_type == "line":
            x1 = shape.get("x1", 0)
            y1 = shape.get("y1", 0)
            x2 = shape.get("x2", 0)
            y2 = shape.get("y2", 0)
            stroke = normalize_color(shape.get("stroke", "black"))
            stroke_width = shape.get("stroke_width", 1)
            opacity = shape.get("opacity", 1.0)
            clip_attr = ' clip-path="url(#canvas-clip)"' if clip else ""
            svg_parts.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"{clip_attr}/>')

        elif shape_type == "path":
            d = shape.get("d", "")
            fill = normalize_color(shape.get("fill", "none"))
            stroke = normalize_color(shape.get("stroke", "black"))
            stroke_width = shape.get("stroke_width", 1)
            opacity = shape.get("opacity", 1.0)
            clip_attr = ' clip-path="url(#canvas-clip)"' if clip else ""
            svg_parts.append(f'  <path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"{clip_attr}/>')

        elif shape_type == "polygon":
            points = shape.get("points", "")
            fill = normalize_color(shape.get("fill", "black"))
            stroke = normalize_color(shape.get("stroke", "none"))
            stroke_width = shape.get("stroke_width", 1)
            opacity = shape.get("opacity", 1.0)
            clip_attr = ' clip-path="url(#canvas-clip)"' if clip else ""
            svg_parts.append(f'  <polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"{clip_attr}/>')

        elif shape_type == "ellipse":
            cx = shape.get("cx", 0)
            cy = shape.get("cy", 0)
            rx = shape.get("rx", 0)
            ry = shape.get("ry", 0)
            fill = normalize_color(shape.get("fill", "black"))
            stroke = normalize_color(shape.get("stroke", "none"))
            stroke_width = shape.get("stroke_width", 1)
            opacity = shape.get("opacity", 1.0)
            clip_attr = ' clip-path="url(#canvas-clip)"' if clip else ""
            svg_parts.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"{clip_attr}/>')

        elif shape_type == "text":
            x = shape.get("x", 0)
            y = shape.get("y", 0)
            text = shape.get("text", "")
            font_size = shape.get("font_size", 16)
            font_family = shape.get("font_family", "Arial")
            fill = normalize_color(shape.get("fill", "black"))
            opacity = shape.get("opacity", 1.0)
            clip_attr = ' clip-path="url(#canvas-clip)"' if clip else ""
            svg_parts.append(f'  <text x="{x}" y="{y}" font-size="{font_size}" font-family="{font_family}" fill="{fill}" opacity="{opacity}"{clip_attr}>{text}</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def save_svg(svg_content: str, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_content)


def svg_to_png(svg_content: str, output_path: str, scale: float = 1.0) -> None:
    import cairosvg
    cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), write_to=output_path, scale=scale)


def svg_to_png_bytes(svg_content: str, scale: float = 1.0) -> bytes:
    import cairosvg
    return cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), scale=scale)


def generate_line_chart(
    data: Dict[str, List[float]],
    labels: List[str],
    width: int = 600,
    height: int = 400,
    title: str = "",
    line_colors: Optional[List[str]] = None,
    background: str = "white",
    show_grid: bool = True,
    show_dots: bool = True,
    line_width: int = 2,
    font_size: int = 12,
    padding: Optional[Dict[str, int]] = None,
) -> str:
    if padding is None:
        padding = {"top": 50, "right": 30, "bottom": 60, "left": 60}

    plot_x = padding["left"]
    plot_y = padding["top"]
    plot_w = width - padding["left"] - padding["right"]
    plot_h = height - padding["top"] - padding["bottom"]

    all_values = []
    for values in data.values():
        all_values.extend(values)
    if not all_values:
        all_values = [0]

    y_min = 0
    y_max = math.ceil(max(all_values) * 1.1) if max(all_values) > 0 else 10

    shapes = []
    bg_hex = normalize_color(background) if background else None
    if bg_hex:
        shapes.append({"type": "rect", "x": 0, "y": 0, "width": width, "height": height, "fill": bg_hex})

    if title:
        shapes.append({"type": "text", "x": width // 2, "y": padding["top"] // 2 + 5, "text": title, "font_size": font_size + 6, "fill": "#2C3E50"})

    if show_grid:
        grid_count = 5
        for i in range(grid_count + 1):
            gy = plot_y + plot_h - (plot_h * i / grid_count)
            val = y_min + (y_max - y_min) * i / grid_count
            shapes.append({"type": "line", "x1": plot_x, "y1": gy, "x2": plot_x + plot_w, "y2": gy, "stroke": "#E0E0E0", "stroke_width": 1})
            shapes.append({"type": "text", "x": plot_x - 8, "y": gy + 4, "text": f"{val:.0f}", "font_size": font_size - 2, "fill": "#7F8C8D"})

    shapes.append({"type": "line", "x1": plot_x, "y1": plot_y, "x2": plot_x, "y2": plot_y + plot_h, "stroke": "#BDC3C7", "stroke_width": 1})
    shapes.append({"type": "line", "x1": plot_x, "y1": plot_y + plot_h, "x2": plot_x + plot_w, "y2": plot_y + plot_h, "stroke": "#BDC3C7", "stroke_width": 1})

    n = len(labels)
    if n > 1:
        step = plot_w / (n - 1)
    else:
        step = plot_w / 2

    for i, label in enumerate(labels):
        if n > 1:
            lx = plot_x + step * i
        else:
            lx = plot_x + step
        shapes.append({"type": "text", "x": lx, "y": plot_y + plot_h + 20, "text": label, "font_size": font_size - 2, "fill": "#7F8C8D"})

    colors = line_colors or CHART_PALETTE
    for sidx, (series_name, values) in enumerate(data.items()):
        color = normalize_color(colors[sidx % len(colors)])
        points = []
        for i, v in enumerate(values):
            if n > 1:
                px = plot_x + step * i
            else:
                px = plot_x + step
            py = plot_y + plot_h - (v - y_min) / (y_max - y_min) * plot_h if y_max != y_min else plot_y + plot_h
            points.append(f"{px:.1f},{py:.1f}")

        if len(points) >= 2:
            shapes.append({"type": "polyline" if False else "path",
                           "d": "M" + " L".join(points),
                           "fill": "none", "stroke": color, "stroke_width": line_width})

        if show_dots:
            for i, v in enumerate(values):
                if n > 1:
                    px = plot_x + step * i
                else:
                    px = plot_x + step
                py = plot_y + plot_h - (v - y_min) / (y_max - y_min) * plot_h if y_max != y_min else plot_y + plot_h
                shapes.append({"type": "circle", "cx": px, "cy": py, "r": 4, "fill": color, "stroke": "#FFFFFF", "stroke_width": 2})

    legend_x = plot_x + plot_w - len(data) * 90
    legend_y = plot_y + 15
    for sidx, series_name in enumerate(data.keys()):
        color = normalize_color(colors[sidx % len(colors)])
        lx = legend_x + sidx * 90
        shapes.append({"type": "line", "x1": lx, "y1": legend_y, "x2": lx + 20, "y2": legend_y, "stroke": color, "stroke_width": 3})
        shapes.append({"type": "text", "x": lx + 25, "y": legend_y + 4, "text": series_name, "font_size": font_size - 2, "fill": "#2C3E50"})

    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for shape in shapes:
        shape_type = shape.get("type", "").lower()
        if shape_type == "rect":
            svg_parts.append(f'  <rect x="{shape.get("x",0)}" y="{shape.get("y",0)}" width="{shape.get("width",0)}" height="{shape.get("height",0)}" fill="{normalize_color(shape.get("fill","black"))}"/>')
        elif shape_type == "line":
            svg_parts.append(f'  <line x1="{shape.get("x1",0)}" y1="{shape.get("y1",0)}" x2="{shape.get("x2",0)}" y2="{shape.get("y2",0)}" stroke="{normalize_color(shape.get("stroke","black"))}" stroke-width="{shape.get("stroke_width",1)}"/>')
        elif shape_type == "text":
            svg_parts.append(f'  <text x="{shape.get("x",0)}" y="{shape.get("y",0)}" font-size="{shape.get("font_size",12)}" fill="{normalize_color(shape.get("fill","black"))}" text-anchor="middle">{shape.get("text","")}</text>')
        elif shape_type == "circle":
            svg_parts.append(f'  <circle cx="{shape.get("cx",0)}" cy="{shape.get("cy",0)}" r="{shape.get("r",4)}" fill="{normalize_color(shape.get("fill","black"))}" stroke="{normalize_color(shape.get("stroke","none"))}" stroke-width="{shape.get("stroke_width",1)}"/>')
        elif shape_type == "path":
            svg_parts.append(f'  <path d="{shape.get("d","")}" fill="{normalize_color(shape.get("fill","none"))}" stroke="{normalize_color(shape.get("stroke","black"))}" stroke-width="{shape.get("stroke_width",1)}"/>')
    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def generate_bar_chart(
    data: Dict[str, float],
    width: int = 600,
    height: int = 400,
    title: str = "",
    bar_colors: Optional[List[str]] = None,
    background: str = "white",
    show_grid: bool = True,
    show_values: bool = True,
    bar_width: Optional[int] = None,
    font_size: int = 12,
    padding: Optional[Dict[str, int]] = None,
) -> str:
    if padding is None:
        padding = {"top": 50, "right": 30, "bottom": 80, "left": 60}

    plot_x = padding["left"]
    plot_y = padding["top"]
    plot_w = width - padding["left"] - padding["right"]
    plot_h = height - padding["top"] - padding["bottom"]

    labels = list(data.keys())
    values = list(data.values())
    n = len(labels)

    if not values:
        values = [0]

    y_max = math.ceil(max(values) * 1.15) if max(values) > 0 else 10

    shapes = []
    bg_hex = normalize_color(background) if background else None
    if bg_hex:
        shapes.append({"type": "rect", "x": 0, "y": 0, "width": width, "height": height, "fill": bg_hex})

    if title:
        shapes.append({"type": "text", "x": width // 2, "y": padding["top"] // 2 + 5, "text": title, "font_size": font_size + 6, "fill": "#2C3E50"})

    if show_grid:
        grid_count = 5
        for i in range(grid_count + 1):
            gy = plot_y + plot_h - (plot_h * i / grid_count)
            val = y_max * i / grid_count
            shapes.append({"type": "line", "x1": plot_x, "y1": gy, "x2": plot_x + plot_w, "y2": gy, "stroke": "#E0E0E0", "stroke_width": 1})
            shapes.append({"type": "text", "x": plot_x - 8, "y": gy + 4, "text": f"{val:.0f}", "font_size": font_size - 2, "fill": "#7F8C8D"})

    shapes.append({"type": "line", "x1": plot_x, "y1": plot_y, "x2": plot_x, "y2": plot_y + plot_h, "stroke": "#BDC3C7", "stroke_width": 1})
    shapes.append({"type": "line", "x1": plot_x, "y1": plot_y + plot_h, "x2": plot_x + plot_w, "y2": plot_y + plot_h, "stroke": "#BDC3C7", "stroke_width": 1})

    group_width = plot_w / n
    if bar_width is None:
        bar_width = min(int(group_width * 0.6), 60)

    colors = bar_colors or CHART_PALETTE

    for i, (label, value) in enumerate(zip(labels, values)):
        color = normalize_color(colors[i % len(colors)])
        bx = plot_x + group_width * i + (group_width - bar_width) / 2
        bh = (value / y_max) * plot_h if y_max > 0 else 0
        by = plot_y + plot_h - bh
        shapes.append({"type": "rect", "x": int(bx), "y": int(by), "width": bar_width, "height": int(bh), "fill": color, "rx": 3})

        if show_values:
            shapes.append({"type": "text", "x": int(bx + bar_width / 2), "y": int(by - 8), "text": f"{value}", "font_size": font_size - 2, "fill": "#2C3E50"})

        shapes.append({"type": "text", "x": int(bx + bar_width / 2), "y": plot_y + plot_h + 20, "text": label, "font_size": font_size - 2, "fill": "#7F8C8D"})

    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for shape in shapes:
        shape_type = shape.get("type", "").lower()
        if shape_type == "rect":
            svg_parts.append(f'  <rect x="{shape.get("x",0)}" y="{shape.get("y",0)}" width="{shape.get("width",0)}" height="{shape.get("height",0)}" fill="{normalize_color(shape.get("fill","black"))}" rx="{shape.get("rx",0)}"/>')
        elif shape_type == "line":
            svg_parts.append(f'  <line x1="{shape.get("x1",0)}" y1="{shape.get("y1",0)}" x2="{shape.get("x2",0)}" y2="{shape.get("y2",0)}" stroke="{normalize_color(shape.get("stroke","black"))}" stroke-width="{shape.get("stroke_width",1)}"/>')
        elif shape_type == "text":
            svg_parts.append(f'  <text x="{shape.get("x",0)}" y="{shape.get("y",0)}" font-size="{shape.get("font_size",12)}" fill="{normalize_color(shape.get("fill","black"))}" text-anchor="middle">{shape.get("text","")}</text>')
    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def generate_pie_chart(
    data: Dict[str, float],
    width: int = 500,
    height: int = 400,
    title: str = "",
    slice_colors: Optional[List[str]] = None,
    background: str = "white",
    show_labels: bool = True,
    show_percent: bool = True,
    font_size: int = 12,
    donut: bool = False,
    donut_radius: Optional[float] = None,
) -> str:
    cx = width // 2
    cy = height // 2 + 15
    radius = min(width, height) // 2 - 60
    if radius < 50:
        radius = 50

    total = sum(data.values())
    if total == 0:
        total = 1

    labels = list(data.keys())
    values = list(data.values())
    n = len(labels)

    shapes = []
    bg_hex = normalize_color(background) if background else None
    if bg_hex:
        shapes.append({"type": "rect", "x": 0, "y": 0, "width": width, "height": height, "fill": bg_hex})

    if title:
        shapes.append({"type": "text", "x": width // 2, "y": 25, "text": title, "font_size": font_size + 6, "fill": "#2C3E50"})

    colors = slice_colors or CHART_PALETTE
    start_angle = -math.pi / 2

    for i, (label, value) in enumerate(zip(labels, values)):
        color = normalize_color(colors[i % len(colors)])
        sweep = (value / total) * 2 * math.pi
        end_angle = start_angle + sweep

        large_arc = 1 if sweep > math.pi else 0

        x1 = cx + radius * math.cos(start_angle)
        y1 = cy + radius * math.sin(start_angle)
        x2 = cx + radius * math.cos(end_angle)
        y2 = cy + radius * math.sin(end_angle)

        if abs(sweep - 2 * math.pi) < 0.001:
            shapes.append({"type": "circle", "cx": cx, "cy": cy, "r": radius, "fill": color, "stroke": "#FFFFFF", "stroke_width": 2})
        else:
            if donut and donut_radius is not None:
                ir = donut_radius
                ix1 = cx + ir * math.cos(end_angle)
                iy1 = cy + ir * math.sin(end_angle)
                ix2 = cx + ir * math.cos(start_angle)
                iy2 = cy + ir * math.sin(start_angle)
                large_arc_inner = large_arc
                d = (f"M{x1:.1f},{y1:.1f} "
                     f"A{radius},{radius} 0 {large_arc},1 {x2:.1f},{y2:.1f} "
                     f"L{ix1:.1f},{iy1:.1f} "
                     f"A{ir},{ir} 0 {large_arc_inner},0 {ix2:.1f},{iy2:.1f} Z")
                shapes.append({"type": "path", "d": d, "fill": color, "stroke": "#FFFFFF", "stroke_width": 2})
            else:
                d = (f"M{cx},{cy} "
                     f"L{x1:.1f},{y1:.1f} "
                     f"A{radius},{radius} 0 {large_arc},1 {x2:.1f},{y2:.1f} Z")
                shapes.append({"type": "path", "d": d, "fill": color, "stroke": "#FFFFFF", "stroke_width": 2})

        if show_labels or show_percent:
            mid_angle = start_angle + sweep / 2
            label_r = radius * 0.65 if not donut else (radius + (donut_radius or 0)) / 2
            if donut and donut_radius:
                label_r = (radius + donut_radius) / 2
            lx = cx + label_r * math.cos(mid_angle)
            ly = cy + label_r * math.sin(mid_angle)

            text_parts = []
            if show_labels:
                text_parts.append(label)
            if show_percent:
                pct = (value / total) * 100
                text_parts.append(f"{pct:.1f}%")
            text_str = " ".join(text_parts)

            if sweep > 0.15:
                shapes.append({"type": "text", "x": lx, "y": ly + 4, "text": text_str, "font_size": font_size - 2, "fill": "#FFFFFF"})

        start_angle = end_angle

    legend_y = height - 25
    legend_x_start = width // 2 - n * 40
    for i, label in enumerate(labels):
        color = normalize_color(colors[i % len(colors)])
        lx = legend_x_start + i * 80
        if lx + 70 > width:
            break
        shapes.append({"type": "rect", "x": lx, "y": legend_y - 8, "width": 12, "height": 12, "fill": color})
        shapes.append({"type": "text", "x": lx + 16, "y": legend_y + 2, "text": label, "font_size": font_size - 2, "fill": "#2C3E50"})

    svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for shape in shapes:
        shape_type = shape.get("type", "").lower()
        if shape_type == "rect":
            svg_parts.append(f'  <rect x="{shape.get("x",0)}" y="{shape.get("y",0)}" width="{shape.get("width",0)}" height="{shape.get("height",0)}" fill="{normalize_color(shape.get("fill","black"))}"/>')
        elif shape_type == "circle":
            svg_parts.append(f'  <circle cx="{shape.get("cx",0)}" cy="{shape.get("cy",0)}" r="{shape.get("r",0)}" fill="{normalize_color(shape.get("fill","black"))}" stroke="{normalize_color(shape.get("stroke","none"))}" stroke-width="{shape.get("stroke_width",1)}"/>')
        elif shape_type == "text":
            svg_parts.append(f'  <text x="{shape.get("x",0)}" y="{shape.get("y",0)}" font-size="{shape.get("font_size",12)}" fill="{normalize_color(shape.get("fill","black"))}" text-anchor="middle">{shape.get("text","")}</text>')
        elif shape_type == "path":
            svg_parts.append(f'  <path d="{shape.get("d","")}" fill="{normalize_color(shape.get("fill","black"))}" stroke="{normalize_color(shape.get("stroke","none"))}" stroke-width="{shape.get("stroke_width",1)}"/>')
    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def optimize_svg(svg_content: str, remove_metadata: bool = True, remove_comments: bool = True, minify: bool = True) -> str:
    optimized = svg_content

    if remove_comments:
        optimized = re.sub(r'<!--.*?-->', '', optimized, flags=re.DOTALL)

    if remove_metadata:
        optimized = re.sub(r'\s*<metadata[^>]*>.*?</metadata>', '', optimized, flags=re.DOTALL)
        optimized = re.sub(r'\s*<desc[^>]*>.*?</desc>', '', optimized, flags=re.DOTALL)
        optimized = re.sub(r'\s*<title[^>]*>.*?</title>', '', optimized, flags=re.DOTALL)

    redundant_defaults = [
        (r'\s+stroke="none"', ''),
        (r'\s+fill="black"', ''),
        (r'\s+opacity="1(\.0)?"', ''),
        (r'\s+stroke-width="1"', ''),
        (r'\s+rx="0"', ''),
        (r'\s+ry="0"', ''),
    ]

    for pattern, replacement in redundant_defaults:
        optimized = re.sub(pattern, replacement, optimized)

    optimized = re.sub(r'\s+>', '>', optimized)
    optimized = re.sub(r'\s+/>', '/>', optimized)
    optimized = re.sub(r'(\w)  +(\w)', r'\1 \2', optimized)

    if minify:
        lines = [line.strip() for line in optimized.split('\n') if line.strip()]
        optimized = ''.join(lines)

    optimized = re.sub(r'</svg>\s*$', '</svg>', optimized)

    return optimized


if __name__ == "__main__":
    print("=== Test 1: Line Chart ===")
    line_svg = generate_line_chart(
        data={"Sales": [120, 200, 150, 280, 230, 310], "Costs": [80, 120, 100, 150, 130, 180]},
        labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        title="Monthly Sales vs Costs",
        width=600, height=400,
    )
    save_svg(line_svg, "chart_line.svg")
    print(f"Line chart saved ({len(line_svg)} chars)")

    print("\n=== Test 2: Bar Chart ===")
    bar_svg = generate_bar_chart(
        data={"Product A": 450, "Product B": 320, "Product C": 580, "Product D": 220, "Product E": 390},
        title="Product Sales Comparison",
        width=600, height=400,
    )
    save_svg(bar_svg, "chart_bar.svg")
    print(f"Bar chart saved ({len(bar_svg)} chars)")

    print("\n=== Test 3: Pie Chart ===")
    pie_svg = generate_pie_chart(
        data={"Desktop": 45, "Mobile": 30, "Tablet": 15, "Other": 10},
        title="Device Usage Distribution",
        width=500, height=400,
    )
    save_svg(pie_svg, "chart_pie.svg")
    print(f"Pie chart saved ({len(pie_svg)} chars)")

    print("\n=== Test 4: Donut Chart ===")
    donut_svg = generate_pie_chart(
        data={"Direct": 35, "Organic": 28, "Referral": 20, "Social": 12, "Email": 5},
        title="Traffic Sources",
        donut=True,
        donut_radius=60,
        width=500, height=400,
    )
    save_svg(donut_svg, "chart_donut.svg")
    print(f"Donut chart saved ({len(donut_svg)} chars)")

    print("\n=== Test 5: SVG Optimization ===")
    test_svg = generate_svg(
        [
            {"type": "rect", "x": 10, "y": 10, "width": 100, "height": 80, "fill": "#3498DB", "rx": 5},
            {"type": "circle", "cx": 200, "cy": 80, "r": 40, "fill": "#E74C3C", "stroke": "none", "stroke_width": 1, "opacity": 1.0},
            {"type": "line", "x1": 10, "y1": 150, "x2": 300, "y2": 150, "stroke": "#2ECC71", "stroke_width": 2},
        ],
        width=400, height=300,
    )
    optimized = optimize_svg(test_svg)
    save_svg(optimized, "optimized.svg")
    print(f"Original: {len(test_svg)} chars -> Optimized: {len(optimized)} chars ({100 - len(optimized)/len(test_svg)*100:.1f}% reduction)")

    print("\n=== Test 6: SVG to PNG ===")
    try:
        svg_to_png(line_svg, "chart_line.png", scale=2.0)
        svg_to_png(bar_svg, "chart_bar.png", scale=2.0)
        svg_to_png(pie_svg, "chart_pie.png", scale=2.0)
        print("All charts converted to PNG successfully (2x scale)")
    except Exception as e:
        print(f"PNG conversion requires Cairo system library: {e}")

    print("\nAll tests completed.")
