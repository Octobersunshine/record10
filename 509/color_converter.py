def rgb_to_hex(r, g, b):
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in range 0-255")
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        raise ValueError("HEX color must be 6 characters long")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def _clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def rgb_to_hsl(r, g, b):
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in range 0-255")
    
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    max_val = max(r_norm, g_norm, b_norm)
    min_val = min(r_norm, g_norm, b_norm)
    delta = max_val - min_val
    
    l = (max_val + min_val) / 2.0
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / (1 - abs(2 * l - 1)) if l != 0 and l != 1 else 0
        
        if max_val == r_norm:
            h = ((g_norm - b_norm) / delta) % 6
        elif max_val == g_norm:
            h = (b_norm - r_norm) / delta + 2
        else:
            h = (r_norm - g_norm) / delta + 4
        
        h *= 60
        if h < 0:
            h += 360
    
    h = round(h % 360, 2)
    s = round(_clamp(s * 100, 0, 100), 2)
    l = round(_clamp(l * 100, 0, 100), 2)
    return (h, s, l)


def hsl_to_rgb(h, s, l):
    if not (0 <= h <= 360):
        raise ValueError("Hue must be in range 0-360")
    if not (0 <= s <= 100):
        raise ValueError("Saturation must be in range 0-100")
    if not (0 <= l <= 100):
        raise ValueError("Lightness must be in range 0-100")
    
    s_norm = s / 100.0
    l_norm = l / 100.0
    
    c = (1 - abs(2 * l_norm - 1)) * s_norm
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l_norm - c / 2
    
    if 0 <= h < 60:
        r_norm, g_norm, b_norm = c, x, 0
    elif 60 <= h < 120:
        r_norm, g_norm, b_norm = x, c, 0
    elif 120 <= h < 180:
        r_norm, g_norm, b_norm = 0, c, x
    elif 180 <= h < 240:
        r_norm, g_norm, b_norm = 0, x, c
    elif 240 <= h < 300:
        r_norm, g_norm, b_norm = x, 0, c
    else:
        r_norm, g_norm, b_norm = c, 0, x
    
    r = round((r_norm + m) * 255)
    g = round((g_norm + m) * 255)
    b = round((b_norm + m) * 255)
    
    return (r, g, b)


def rgb_to_hsv(r, g, b):
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in range 0-255")
    
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    max_val = max(r_norm, g_norm, b_norm)
    min_val = min(r_norm, g_norm, b_norm)
    delta = max_val - min_val
    
    v = max_val
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / max_val if max_val != 0 else 0
        
        if max_val == r_norm:
            h = ((g_norm - b_norm) / delta) % 6
        elif max_val == g_norm:
            h = (b_norm - r_norm) / delta + 2
        else:
            h = (r_norm - g_norm) / delta + 4
        
        h *= 60
        if h < 0:
            h += 360
    
    h = round(h % 360, 2)
    s = round(_clamp(s * 100, 0, 100), 2)
    v = round(_clamp(v * 100, 0, 100), 2)
    return (h, s, v)


def hsv_to_rgb(h, s, v):
    if not (0 <= h <= 360):
        raise ValueError("Hue must be in range 0-360")
    if not (0 <= s <= 100):
        raise ValueError("Saturation must be in range 0-100")
    if not (0 <= v <= 100):
        raise ValueError("Value must be in range 0-100")
    
    s_norm = s / 100.0
    v_norm = v / 100.0
    
    c = v_norm * s_norm
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v_norm - c
    
    if 0 <= h < 60:
        r_norm, g_norm, b_norm = c, x, 0
    elif 60 <= h < 120:
        r_norm, g_norm, b_norm = x, c, 0
    elif 120 <= h < 180:
        r_norm, g_norm, b_norm = 0, c, x
    elif 180 <= h < 240:
        r_norm, g_norm, b_norm = 0, x, c
    elif 240 <= h < 300:
        r_norm, g_norm, b_norm = x, 0, c
    else:
        r_norm, g_norm, b_norm = c, 0, x
    
    r = round((r_norm + m) * 255)
    g = round((g_norm + m) * 255)
    b = round((b_norm + m) * 255)
    
    return (r, g, b)


def rgb_to_cmyk(r, g, b):
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in range 0-255")
    
    if r == 0 and g == 0 and b == 0:
        return (0, 0, 0, 100)
    
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    k = 1 - max(r_norm, g_norm, b_norm)
    if k == 1:
        c = m = y = 0
    else:
        c = (1 - r_norm - k) / (1 - k)
        m = (1 - g_norm - k) / (1 - k)
        y = (1 - b_norm - k) / (1 - k)
    
    return (
        round(_clamp(c * 100, 0, 100), 2),
        round(_clamp(m * 100, 0, 100), 2),
        round(_clamp(y * 100, 0, 100), 2),
        round(_clamp(k * 100, 0, 100), 2)
    )


def cmyk_to_rgb(c, m, y, k):
    if not (0 <= c <= 100 and 0 <= m <= 100 and 0 <= y <= 100 and 0 <= k <= 100):
        raise ValueError("CMYK values must be in range 0-100")
    
    c_norm = c / 100.0
    m_norm = m / 100.0
    y_norm = y / 100.0
    k_norm = k / 100.0
    
    r = round(255 * (1 - c_norm) * (1 - k_norm))
    g = round(255 * (1 - m_norm) * (1 - k_norm))
    b = round(255 * (1 - y_norm) * (1 - k_norm))
    
    return (_clamp(r, 0, 255), _clamp(g, 0, 255), _clamp(b, 0, 255))


def rgb_to_xyz(r, g, b):
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in range 0-255")
    
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    def f(v):
        if v > 0.04045:
            return ((v + 0.055) / 1.055) ** 2.4
        else:
            return v / 12.92
    
    r_lin = f(r_norm)
    g_lin = f(g_norm)
    b_lin = f(b_norm)
    
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    
    return (round(x * 100, 4), round(y * 100, 4), round(z * 100, 4))


def xyz_to_rgb(x, y, z):
    x_norm = x / 100.0
    y_norm = y / 100.0
    z_norm = z / 100.0
    
    r_lin = x_norm * 3.2404542 + y_norm * -1.5371385 + z_norm * -0.4985314
    g_lin = x_norm * -0.9692660 + y_norm * 1.8760108 + z_norm * 0.0415560
    b_lin = x_norm * 0.0556434 + y_norm * -0.2040259 + z_norm * 1.0572252
    
    def f(v):
        if v > 0.0031308:
            return 1.055 * (v ** (1 / 2.4)) - 0.055
        else:
            return 12.92 * v
    
    r = round(f(r_lin) * 255)
    g = round(f(g_lin) * 255)
    b = round(f(b_lin) * 255)
    
    return (_clamp(r, 0, 255), _clamp(g, 0, 255), _clamp(b, 0, 255))


def xyz_to_lab(x, y, z):
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    
    x_norm = x / ref_x
    y_norm = y / ref_y
    z_norm = z / ref_z
    
    def f(t):
        if t > (6 / 29) ** 3:
            return t ** (1 / 3)
        else:
            return (t / (3 * (6 / 29) ** 2)) + (4 / 29)
    
    fx = f(x_norm)
    fy = f(y_norm)
    fz = f(z_norm)
    
    l = round(116 * fy - 16, 4)
    a = round(500 * (fx - fy), 4)
    b = round(200 * (fy - fz), 4)
    
    return (l, a, b)


def lab_to_xyz(l, a, b):
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    
    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inv(t):
        if t > 6 / 29:
            return t ** 3
        else:
            return 3 * (6 / 29) ** 2 * (t - 4 / 29)
    
    x = f_inv(fx) * ref_x
    y = f_inv(fy) * ref_y
    z = f_inv(fz) * ref_z
    
    return (round(x, 4), round(y, 4), round(z, 4))


def rgb_to_lab(r, g, b):
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)


def lab_to_rgb(l, a, b):
    x, y, z = lab_to_xyz(l, a, b)
    return xyz_to_rgb(x, y, z)


def delta_e76(lab1, lab2):
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    return ((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2) ** 0.5


def delta_e94(lab1, lab2, k_L=1, k_C=1, k_H=1):
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    
    C1 = (a1 ** 2 + b1 ** 2) ** 0.5
    C2 = (a2 ** 2 + b2 ** 2) ** 0.5
    
    dL = l2 - l1
    dC = C2 - C1
    
    da = a2 - a1
    db = b2 - b1
    dH_sq = da ** 2 + db ** 2 - dC ** 2
    dH = (dH_sq if dH_sq > 0 else 0) ** 0.5
    
    s_L = 1
    s_C = 1 + 0.045 * C1
    s_H = 1 + 0.015 * C1
    
    return (((dL / (k_L * s_L)) ** 2) + 
            ((dC / (k_C * s_C)) ** 2) + 
            ((dH / (k_H * s_H)) ** 2)) ** 0.5


def delta_e00(lab1, lab2, k_L=1, k_C=1, k_H=1):
    import math
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    
    C1 = (a1 ** 2 + b1 ** 2) ** 0.5
    C2 = (a2 ** 2 + b2 ** 2) ** 0.5
    C_bar = (C1 + C2) / 2
    
    G = 0.5 * (1 - (C_bar ** 7 / (C_bar ** 7 + 25 ** 7)) ** 0.5)
    
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    C1_prime = (a1_prime ** 2 + b1 ** 2) ** 0.5
    C2_prime = (a2_prime ** 2 + b2 ** 2) ** 0.5
    
    h1_prime = math.atan2(b1, a1_prime) if (a1_prime != 0 or b1 != 0) else 0
    h2_prime = math.atan2(b2, a2_prime) if (a2_prime != 0 or b2 != 0) else 0
    if h1_prime < 0: h1_prime += 2 * math.pi
    if h2_prime < 0: h2_prime += 2 * math.pi
    
    dL_prime = l2 - l1
    dC_prime = C2_prime - C1_prime
    
    dh_prime = h2_prime - h1_prime
    if abs(dh_prime) > math.pi:
        if h2_prime <= h1_prime:
            dh_prime += 2 * math.pi
        else:
            dh_prime -= 2 * math.pi
    dH_prime = 2 * (C1_prime * C2_prime) ** 0.5 * math.sin(dh_prime / 2)
    
    L_bar_prime = (l1 + l2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    
    h_bar_prime = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > math.pi:
        h_bar_prime -= math.pi
    if h_bar_prime < 0:
        h_bar_prime += 2 * math.pi
    
    T = (1 - 0.17 * math.cos(h_bar_prime - math.radians(30)) +
         0.24 * math.cos(2 * h_bar_prime) +
         0.32 * math.cos(3 * h_bar_prime + math.radians(6)) -
         0.20 * math.cos(4 * h_bar_prime - math.radians(63)))
    
    s_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / (20 + (L_bar_prime - 50) ** 2) ** 0.5
    s_C = 1 + 0.045 * C_bar_prime
    s_H = 1 + 0.015 * C_bar_prime * T
    
    delta_theta = math.radians(30) * math.exp(-((math.degrees(h_bar_prime) - 275) / 25) ** 2)
    R_C = 2 * (C_bar_prime ** 7 / (C_bar_prime ** 7 + 25 ** 7)) ** 0.5
    R_T = -R_C * math.sin(2 * delta_theta)
    
    term1 = (dL_prime / (k_L * s_L)) ** 2
    term2 = (dC_prime / (k_C * s_C)) ** 2
    term3 = (dH_prime / (k_H * s_H)) ** 2
    term4 = R_T * (dC_prime / (k_C * s_C)) * (dH_prime / (k_H * s_H))
    
    return (term1 + term2 + term3 + term4) ** 0.5


COLOR_NAMES = {
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255),
    'RED': (255, 0, 0),
    'LIME': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'YELLOW': (255, 255, 0),
    'CYAN': (0, 255, 255),
    'MAGENTA': (255, 0, 255),
    'SILVER': (192, 192, 192),
    'GRAY': (128, 128, 128),
    'MAROON': (128, 0, 0),
    'OLIVE': (128, 128, 0),
    'GREEN': (0, 128, 0),
    'PURPLE': (128, 0, 128),
    'TEAL': (0, 128, 128),
    'NAVY': (0, 0, 128),
    'ORANGE': (255, 165, 0),
    'PINK': (255, 192, 203),
    'BROWN': (165, 42, 42),
    'TOMATO': (255, 99, 71),
    'CORAL': (255, 127, 80),
    'GOLD': (255, 215, 0),
    'KHAKI': (240, 230, 140),
    'BEIGE': (245, 245, 220),
    'IVORY': (255, 255, 240),
    'SNOW': (255, 250, 250),
    'MINT': (189, 252, 201),
    'LAVENDER': (230, 230, 250),
    'VIOLET': (238, 130, 238),
    'INDIGO': (75, 0, 130),
    'SKYBLUE': (135, 206, 235),
    'TURQUOISE': (64, 224, 208),
    'CHARTREUSE': (127, 255, 0),
    'SALMON': (250, 128, 114),
    'CRIMSON': (220, 20, 60),
    'FIREBRICK': (178, 34, 34),
    'DARKRED': (139, 0, 0),
    'FORESTGREEN': (34, 139, 34),
    'SEAGREEN': (46, 139, 87),
    'ROYALBLUE': (65, 105, 225),
    'DEEPSKYBLUE': (0, 191, 255),
}


def get_color_name(color, threshold=30):
    if isinstance(color, str):
        rgb = hex_to_rgb(color)
    else:
        rgb = tuple(color[:3])
    
    best_name = None
    best_distance = float('inf')
    
    for name, name_rgb in COLOR_NAMES.items():
        dist = sum((rgb[i] - name_rgb[i]) ** 2 for i in range(3)) ** 0.5
        if dist < best_distance:
            best_distance = dist
            best_name = name
    
    if best_distance <= threshold:
        return best_name.lower()
    return None


def get_color_by_name(name):
    name_upper = name.strip().upper().replace(' ', '')
    if name_upper in COLOR_NAMES:
        rgb = COLOR_NAMES[name_upper]
        return {'name': name_upper.lower(), 'rgb': rgb, 'hex': rgb_to_hex(*rgb)}
    return None


def list_color_names():
    return [name.lower() for name in sorted(COLOR_NAMES.keys())]


def batch_convert(colors, conversion_type):
    results = []
    for color in colors:
        try:
            if conversion_type == 'rgb_to_hex':
                result = rgb_to_hex(*color)
            elif conversion_type == 'hex_to_rgb':
                result = hex_to_rgb(color)
            elif conversion_type == 'rgb_to_hsl':
                result = rgb_to_hsl(*color)
            elif conversion_type == 'hsl_to_rgb':
                result = hsl_to_rgb(*color)
            elif conversion_type == 'rgb_to_hsv':
                result = rgb_to_hsv(*color)
            elif conversion_type == 'hsv_to_rgb':
                result = hsv_to_rgb(*color)
            elif conversion_type == 'rgb_to_cmyk':
                result = rgb_to_cmyk(*color)
            elif conversion_type == 'cmyk_to_rgb':
                result = cmyk_to_rgb(*color)
            elif conversion_type == 'rgb_to_lab':
                result = rgb_to_lab(*color)
            elif conversion_type == 'lab_to_rgb':
                result = lab_to_rgb(*color)
            else:
                raise ValueError(f"Unknown conversion type: {conversion_type}")
            results.append({'input': color, 'output': result, 'success': True})
        except Exception as e:
            results.append({'input': color, 'error': str(e), 'success': False})
    return results


if __name__ == "__main__":
    print("=== 颜色空间转换测试 ===")
    
    rgb_color = (255, 128, 64)
    hex_color = "#FF8040"
    hsl_color = (20, 100, 62.55)
    hsv_color = (20, 74.9, 100)
    cmyk_color = (0, 49.8, 74.9, 0)
    
    print(f"\nRGB: {rgb_color}")
    print(f"  → HEX: {rgb_to_hex(*rgb_color)}")
    print(f"  → HSL: {rgb_to_hsl(*rgb_color)}")
    print(f"  → HSV: {rgb_to_hsv(*rgb_color)}")
    print(f"  → CMYK: {rgb_to_cmyk(*rgb_color)}")
    print(f"  → LAB: {rgb_to_lab(*rgb_color)}")
    
    print(f"\nHEX: {hex_color}")
    print(f"  → RGB: {hex_to_rgb(hex_color)}")
    
    print(f"\nHSL: {hsl_color}")
    print(f"  → RGB: {hsl_to_rgb(*hsl_color)}")
    
    print(f"\nHSV: {hsv_color}")
    print(f"  → RGB: {hsv_to_rgb(*hsv_color)}")
    
    print(f"\nCMYK: {cmyk_color}")
    print(f"  → RGB: {cmyk_to_rgb(*cmyk_color)}")
    
    print("\n=== 色差计算测试 ===")
    lab1 = rgb_to_lab(255, 0, 0)
    lab2 = rgb_to_lab(200, 30, 30)
    print(f"红色 LAB: {lab1}")
    print(f"深红 LAB: {lab2}")
    print(f"ΔE76: {delta_e76(lab1, lab2):.4f}")
    print(f"ΔE94: {delta_e94(lab1, lab2):.4f}")
    print(f"ΔE00: {delta_e00(lab1, lab2):.4f}")
    
    print("\n=== 颜色名称查询测试 ===")
    print(f"#FF0000 → {get_color_name('#FF0000')}")
    print(f"RGB(255,165,0) → {get_color_name((255, 165, 0))}")
    print(f"按名称'red' → {get_color_by_name('red')}")
    print(f"可用颜色数: {len(list_color_names())}")
    
    print("\n=== 批量转换测试 ===")
    rgb_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    results = batch_convert(rgb_colors, 'rgb_to_cmyk')
    for r in results:
        if r['success']:
            print(f"{r['input']} → {r['output']}")
        else:
            print(f"{r['input']} → 错误: {r['error']}")
