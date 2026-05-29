import re
from typing import Dict, Union, List, Optional


CSS_COLORS = {
    'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4', 'azure': '#F0FFFF', 'beige': '#F5F5DC',
    'bisque': '#FFE4C4', 'black': '#000000', 'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF', 'blueviolet': '#8A2BE2', 'brown': '#A52A2A',
    'burlywood': '#DEB887', 'cadetblue': '#5F9EA0', 'chartreuse': '#7FFF00',
    'chocolate': '#D2691E', 'coral': '#FF7F50', 'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC', 'crimson': '#DC143C', 'cyan': '#00FFFF',
    'darkblue': '#00008B', 'darkcyan': '#008B8B', 'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9', 'darkgrey': '#A9A9A9', 'darkgreen': '#006400',
    'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00', 'darkorchid': '#9932CC', 'darkred': '#8B0000',
    'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F', 'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F', 'darkslategrey': '#2F4F4F', 'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3', 'deeppink': '#FF1493', 'deepskyblue': '#00BFFF',
    'dimgray': '#696969', 'dimgrey': '#696969', 'dodgerblue': '#1E90FF',
    'firebrick': '#B22222', 'floralwhite': '#FFFAF0', 'forestgreen': '#228B22',
    'fuchsia': '#FF00FF', 'gainsboro': '#DCDCDC', 'ghostwhite': '#F8F8FF',
    'gold': '#FFD700', 'goldenrod': '#DAA520', 'gray': '#808080',
    'grey': '#808080', 'green': '#008000', 'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0', 'hotpink': '#FF69B4', 'indianred': '#CD5C5C',
    'indigo': '#4B0082', 'ivory': '#FFFFF0', 'khaki': '#F0E68C',
    'lavender': '#E6E6FA', 'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6', 'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2', 'lightgray': '#D3D3D3',
    'lightgrey': '#D3D3D3', 'lightgreen': '#90EE90', 'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A', 'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA',
    'lightslategray': '#778899', 'lightslategrey': '#778899', 'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0', 'lime': '#00FF00', 'limegreen': '#32CD32',
    'linen': '#FAF0E6', 'magenta': '#FF00FF', 'maroon': '#800000',
    'mediumaquamarine': '#66CDAA', 'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB', 'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A', 'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585',
    'midnightblue': '#191970', 'mintcream': '#F5FFFA', 'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5', 'navajowhite': '#FFDEAD', 'navy': '#000080',
    'oldlace': '#FDF5E6', 'olive': '#808000', 'olivedrab': '#6B8E23',
    'orange': '#FFA500', 'orangered': '#FF4500', 'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98', 'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9',
    'peru': '#CD853F', 'pink': '#FFC0CB', 'plum': '#DDA0DD',
    'powderblue': '#B0E0E6', 'purple': '#800080', 'rebeccapurple': '#663399',
    'red': '#FF0000', 'rosybrown': '#BC8F8F', 'royalblue': '#4169E1',
    'saddlebrown': '#8B4513', 'salmon': '#FA8072', 'sandybrown': '#F4A460',
    'seagreen': '#2E8B57', 'seashell': '#FFF5EE', 'sienna': '#A0522D',
    'silver': '#C0C0C0', 'skyblue': '#87CEEB', 'slateblue': '#6A5ACD',
    'slategray': '#708090', 'slategrey': '#708090', 'snow': '#FFFAFA',
    'springgreen': '#00FF7F', 'steelblue': '#4682B4', 'tan': '#D2B48C',
    'teal': '#008080', 'thistle': '#D8BFD8', 'tomato': '#FF6347',
    'turquoise': '#40E0D0', 'violet': '#EE82EE', 'wheat': '#F5DEB3',
    'white': '#FFFFFF', 'whitesmoke': '#F5F5F5', 'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'
}


class ColorConverter:
    @staticmethod
    def get_css_color_names() -> List[str]:
        return sorted(CSS_COLORS.keys())

    @staticmethod
    def name_to_hex(color_name: str) -> Dict[str, str]:
        normalized_name = color_name.strip().lower().replace(' ', '')
        if normalized_name not in CSS_COLORS:
            raise ValueError(f"Unknown color name: {color_name}. Use get_css_color_names() for valid names.")
        return {'hex': CSS_COLORS[normalized_name], 'name': normalized_name}

    @staticmethod
    def name_to_rgb(color_name: str) -> Dict:
        hex_result = ColorConverter.name_to_hex(color_name)
        rgb_result = ColorConverter.hex_to_rgb(hex_result['hex'])
        return {**rgb_result, 'name': hex_result['name'], 'hex': hex_result['hex']}

    @staticmethod
    def hex_to_name(hex_color: str) -> Dict[str, str]:
        normalized = ColorConverter.normalize_hex(hex_color)
        for name, hex_val in CSS_COLORS.items():
            if hex_val == normalized:
                return {'name': name, 'hex': normalized}
        return {'name': None, 'hex': normalized, 'message': 'No matching CSS color name found'}

    @staticmethod
    def normalize_hex(hex_color: str) -> str:
        hex_color = hex_color.strip().lstrip('#')
        if not re.match(r'^[0-9A-Fa-f]{3}$|^[0-9A-Fa-f]{6}$', hex_color):
            raise ValueError("Invalid HEX color format: must be 3 or 6 hex digits")
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        return '#' + hex_color.upper()

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Dict[str, int]:
        normalized = ColorConverter.normalize_hex(hex_color)
        hex_digits = normalized.lstrip('#')
        
        r = int(hex_digits[0:2], 16)
        g = int(hex_digits[2:4], 16)
        b = int(hex_digits[4:6], 16)
        
        return {'r': r, 'g': g, 'b': b}

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> Dict[str, str]:
        if not (isinstance(r, int) and isinstance(g, int) and isinstance(b, int)):
            raise ValueError("RGB values must be integers")
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("RGB values must be between 0 and 255")
        
        hex_color = '#{:02X}{:02X}{:02X}'.format(r, g, b)
        return {'hex': hex_color}

    @staticmethod
    def rgb_to_hsl(r: int, g: int, b: int) -> Dict[str, float]:
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("RGB values must be between 0 and 255")
        
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        delta = max_val - min_val
        
        l = (max_val + min_val) / 2.0
        
        if delta == 0:
            h = 0.0
            s = 0.0
        else:
            s = delta / (1 - abs(2 * l - 1))
            
            if max_val == r_norm:
                h = ((g_norm - b_norm) / delta) % 6
            elif max_val == g_norm:
                h = (b_norm - r_norm) / delta + 2
            else:
                h = (r_norm - g_norm) / delta + 4
            
            h *= 60
            if h < 0:
                h += 360
        
        return {
            'h': round(h, 2),
            's': round(s * 100, 2),
            'l': round(l * 100, 2)
        }

    @staticmethod
    def hsl_to_rgb(h: float, s: float, l: float) -> Dict[str, int]:
        if not (0 <= h <= 360):
            raise ValueError("Hue must be between 0 and 360")
        if not (0 <= s <= 100):
            raise ValueError("Saturation must be between 0 and 100")
        if not (0 <= l <= 100):
            raise ValueError("Lightness must be between 0 and 100")
        
        s_norm = s / 100.0
        l_norm = l / 100.0
        
        c = (1 - abs(2 * l_norm - 1)) * s_norm
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l_norm - c / 2
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return {
            'r': round((r + m) * 255),
            'g': round((g + m) * 255),
            'b': round((b + m) * 255)
        }

    @staticmethod
    def rgb_to_cmyk(r: int, g: int, b: int) -> Dict[str, float]:
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("RGB values must be between 0 and 255")
        
        if r == 0 and g == 0 and b == 0:
            return {'c': 0, 'm': 0, 'y': 0, 'k': 100}
        
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        k = 1 - max(r_norm, g_norm, b_norm)
        c = (1 - r_norm - k) / (1 - k) if (1 - k) != 0 else 0
        m = (1 - g_norm - k) / (1 - k) if (1 - k) != 0 else 0
        y = (1 - b_norm - k) / (1 - k) if (1 - k) != 0 else 0
        
        return {
            'c': round(c * 100, 2),
            'm': round(m * 100, 2),
            'y': round(y * 100, 2),
            'k': round(k * 100, 2)
        }

    @staticmethod
    def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> Dict[str, int]:
        if not (0 <= c <= 100 and 0 <= m <= 100 and 0 <= y <= 100 and 0 <= k <= 100):
            raise ValueError("CMYK values must be between 0 and 100")
        
        c_norm = c / 100.0
        m_norm = m / 100.0
        y_norm = y / 100.0
        k_norm = k / 100.0
        
        r = 255 * (1 - c_norm) * (1 - k_norm)
        g = 255 * (1 - m_norm) * (1 - k_norm)
        b = 255 * (1 - y_norm) * (1 - k_norm)
        
        return {
            'r': round(r),
            'g': round(g),
            'b': round(b)
        }

    @staticmethod
    def hex_to_hsl(hex_color: str) -> Dict[str, float]:
        rgb = ColorConverter.hex_to_rgb(hex_color)
        return ColorConverter.rgb_to_hsl(rgb['r'], rgb['g'], rgb['b'])

    @staticmethod
    def hex_to_cmyk(hex_color: str) -> Dict[str, float]:
        rgb = ColorConverter.hex_to_rgb(hex_color)
        return ColorConverter.rgb_to_cmyk(rgb['r'], rgb['g'], rgb['b'])

    @staticmethod
    def hsl_to_hex(h: float, s: float, l: float) -> Dict[str, str]:
        rgb = ColorConverter.hsl_to_rgb(h, s, l)
        return ColorConverter.rgb_to_hex(rgb['r'], rgb['g'], rgb['b'])

    @staticmethod
    def hsl_to_cmyk(h: float, s: float, l: float) -> Dict[str, float]:
        rgb = ColorConverter.hsl_to_rgb(h, s, l)
        return ColorConverter.rgb_to_cmyk(rgb['r'], rgb['g'], rgb['b'])

    @staticmethod
    def cmyk_to_hex(c: float, m: float, y: float, k: float) -> Dict[str, str]:
        rgb = ColorConverter.cmyk_to_rgb(c, m, y, k)
        return ColorConverter.rgb_to_hex(rgb['r'], rgb['g'], rgb['b'])

    @staticmethod
    def cmyk_to_hsl(c: float, m: float, y: float, k: float) -> Dict[str, float]:
        rgb = ColorConverter.cmyk_to_rgb(c, m, y, k)
        return ColorConverter.rgb_to_hsl(rgb['r'], rgb['g'], rgb['b'])

    @staticmethod
    def convert_all(hex_color: str = None, r: int = None, g: int = None, b: int = None,
                    h: float = None, s: float = None, l: float = None,
                    c: float = None, m: float = None, y: float = None, k: float = None) -> Dict[str, Union[str, float, int]]:
        if hex_color is not None:
            rgb = ColorConverter.hex_to_rgb(hex_color)
            r, g, b = rgb['r'], rgb['g'], rgb['b']
        elif r is not None and g is not None and b is not None:
            pass
        elif h is not None and s is not None and l is not None:
            rgb = ColorConverter.hsl_to_rgb(h, s, l)
            r, g, b = rgb['r'], rgb['g'], rgb['b']
        elif c is not None and m is not None and y is not None and k is not None:
            rgb = ColorConverter.cmyk_to_rgb(c, m, y, k)
            r, g, b = rgb['r'], rgb['g'], rgb['b']
        else:
            raise ValueError("Must provide at least one color format")
        
        hex_result = ColorConverter.rgb_to_hex(r, g, b)
        hsl_result = ColorConverter.rgb_to_hsl(r, g, b)
        cmyk_result = ColorConverter.rgb_to_cmyk(r, g, b)
        
        return {
            'hex': hex_result['hex'],
            'rgb': {'r': r, 'g': g, 'b': b},
            'hsl': hsl_result,
            'cmyk': cmyk_result
        }

    @staticmethod
    def get_color_palette(base_color: Union[str, Dict], palette_type: str = 'complementary') -> Dict:
        if isinstance(base_color, str):
            if base_color.lower().replace(' ', '') in CSS_COLORS:
                base_hex = ColorConverter.name_to_hex(base_color)['hex']
            else:
                base_hex = ColorConverter.normalize_hex(base_color)
        elif isinstance(base_color, dict):
            if 'hex' in base_color:
                base_hex = ColorConverter.normalize_hex(base_color['hex'])
            elif 'rgb' in base_color:
                rgb = base_color['rgb']
                base_hex = ColorConverter.rgb_to_hex(rgb['r'], rgb['g'], rgb['b'])['hex']
            elif 'hsl' in base_color:
                hsl = base_color['hsl']
                rgb = ColorConverter.hsl_to_rgb(hsl['h'], hsl['s'], hsl['l'])
                base_hex = ColorConverter.rgb_to_hex(rgb['r'], rgb['g'], rgb['b'])['hex']
            else:
                raise ValueError("Invalid base color format")
        else:
            raise ValueError("Base color must be a string or dict")

        base_hsl = ColorConverter.hex_to_hsl(base_hex)
        h, s, l = base_hsl['h'], base_hsl['s'], base_hsl['l']

        palette = {'base': base_hex, 'type': palette_type, 'colors': []}

        if palette_type == 'complementary':
            palette['colors'] = [
                base_hex,
                ColorConverter.hsl_to_hex((h + 180) % 360, s, l)['hex']
            ]
        elif palette_type == 'analogous':
            palette['colors'] = [
                ColorConverter.hsl_to_hex((h - 30) % 360, s, l)['hex'],
                base_hex,
                ColorConverter.hsl_to_hex((h + 30) % 360, s, l)['hex']
            ]
        elif palette_type == 'triadic':
            palette['colors'] = [
                base_hex,
                ColorConverter.hsl_to_hex((h + 120) % 360, s, l)['hex'],
                ColorConverter.hsl_to_hex((h + 240) % 360, s, l)['hex']
            ]
        elif palette_type == 'split_complementary':
            palette['colors'] = [
                base_hex,
                ColorConverter.hsl_to_hex((h + 150) % 360, s, l)['hex'],
                ColorConverter.hsl_to_hex((h + 210) % 360, s, l)['hex']
            ]
        elif palette_type == 'tetradic':
            palette['colors'] = [
                base_hex,
                ColorConverter.hsl_to_hex((h + 90) % 360, s, l)['hex'],
                ColorConverter.hsl_to_hex((h + 180) % 360, s, l)['hex'],
                ColorConverter.hsl_to_hex((h + 270) % 360, s, l)['hex']
            ]
        elif palette_type == 'square':
            palette['colors'] = [
                base_hex,
                ColorConverter.hsl_to_hex((h + 90) % 360, s, l)['hex'],
                ColorConverter.hsl_to_hex((h + 180) % 360, s, l)['hex'],
                ColorConverter.hsl_to_hex((h + 270) % 360, s, l)['hex']
            ]
        elif palette_type == 'monochromatic':
            palette['colors'] = [
                ColorConverter.hsl_to_hex(h, s, max(10, l - 40))['hex'],
                ColorConverter.hsl_to_hex(h, s, max(10, l - 20))['hex'],
                base_hex,
                ColorConverter.hsl_to_hex(h, s, min(90, l + 20))['hex'],
                ColorConverter.hsl_to_hex(h, s, min(90, l + 40))['hex']
            ]
        elif palette_type == 'shades':
            palette['colors'] = [
                ColorConverter.hsl_to_hex(h, s, 90)['hex'],
                ColorConverter.hsl_to_hex(h, s, 70)['hex'],
                ColorConverter.hsl_to_hex(h, s, 50)['hex'],
                ColorConverter.hsl_to_hex(h, s, 30)['hex'],
                ColorConverter.hsl_to_hex(h, s, 10)['hex']
            ]
        else:
            raise ValueError(f"Unknown palette type: {palette_type}. "
                           f"Valid types: complementary, analogous, triadic, split_complementary, "
                           f"tetradic, square, monochromatic, shades")

        return palette


def color_conversion_api(color_input: Union[Dict, List]) -> Dict:
    try:
        if isinstance(color_input, list):
            results = []
            for item in color_input:
                result = color_conversion_api(item)
                results.append({'input': item, 'result': result})
            return {'success': True, 'batch': True, 'results': results}

        if isinstance(color_input, str):
            color_name = color_input.lower().replace(' ', '')
            if color_name in CSS_COLORS:
                result = ColorConverter.convert_all(hex_color=CSS_COLORS[color_name])
                result['name'] = color_name
                return {'success': True, 'result': result}
            else:
                result = ColorConverter.convert_all(hex_color=color_input)
                return {'success': True, 'result': result}

        if 'name' in color_input:
            result = ColorConverter.convert_all(
                hex_color=ColorConverter.name_to_hex(color_input['name'])['hex']
            )
            result['name'] = color_input['name'].lower().replace(' ', '')
            return {'success': True, 'result': result}

        if 'hex' in color_input:
            result = ColorConverter.convert_all(hex_color=color_input['hex'])
            name_result = ColorConverter.hex_to_name(color_input['hex'])
            if name_result['name']:
                result['name'] = name_result['name']
        elif 'rgb' in color_input:
            rgb = color_input['rgb']
            result = ColorConverter.convert_all(r=rgb['r'], g=rgb['g'], b=rgb['b'])
        elif 'hsl' in color_input:
            hsl = color_input['hsl']
            result = ColorConverter.convert_all(h=hsl['h'], s=hsl['s'], l=hsl['l'])
        elif 'cmyk' in color_input:
            cmyk = color_input['cmyk']
            result = ColorConverter.convert_all(c=cmyk['c'], m=cmyk['m'], y=cmyk['y'], k=cmyk['k'])
        elif 'palette' in color_input:
            palette = ColorConverter.get_color_palette(
                color_input['palette'].get('base', '#FF0000'),
                color_input['palette'].get('type', 'complementary')
            )
            return {'success': True, 'palette': palette}
        else:
            return {'success': False, 'error': 'Invalid input format. Must provide hex, rgb, hsl, cmyk, name, or palette.'}

        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}


if __name__ == '__main__':
    print("=== 颜色格式转换测试 ===")
    
    print("\n--- 基本转换测试 ---")
    test_hex = '#FF5733'
    print(f"\n1. HEX 输入: {test_hex}")
    result = color_conversion_api({'hex': test_hex})
    print(f"结果: {result}")
    
    test_rgb = {'r': 255, 'g': 87, 'b': 51}
    print(f"\n2. RGB 输入: {test_rgb}")
    result = color_conversion_api({'rgb': test_rgb})
    print(f"结果: {result}")
    
    test_hsl = {'h': 11, 's': 100, 'l': 60}
    print(f"\n3. HSL 输入: {test_hsl}")
    result = color_conversion_api({'hsl': test_hsl})
    print(f"结果: {result}")
    
    test_cmyk = {'c': 0, 'm': 66, 'y': 80, 'k': 0}
    print(f"\n4. CMYK 输入: {test_cmyk}")
    result = color_conversion_api({'cmyk': test_cmyk})
    print(f"结果: {result}")
    
    print("\n--- 前导零测试 ---")
    low_rgb = {'r': 10, 'g': 47, 'b': 0}
    print(f"\n5. RGB(10, 47, 0) → HEX (应为 #0A2F00)")
    result = color_conversion_api({'rgb': low_rgb})
    print(f"结果: {result}")
    
    print(f"\n6. RGB(0, 0, 0) → HEX (应为 #000000)")
    result = color_conversion_api({'rgb': {'r': 0, 'g': 0, 'b': 0}})
    print(f"结果: {result}")
    
    print(f"\n7. RGB(1, 2, 3) → HEX (应为 #010203)")
    result = color_conversion_api({'rgb': {'r': 1, 'g': 2, 'b': 3}})
    print(f"结果: {result}")
    
    print("\n--- 简写HEX转全写测试 ---")
    print(f"\n8. #FFF → normalize → (应为 #FFFFFF)")
    print(f"   normalize_hex: {ColorConverter.normalize_hex('#FFF')}")
    
    print(f"\n9. #0af → normalize → (应为 #00AAFF)")
    print(f"   normalize_hex: {ColorConverter.normalize_hex('#0af')}")
    
    print(f"\n10. #000 → RGB → HEX (应为 #000000)")
    result = color_conversion_api({'hex': '#000'})
    print(f"结果: {result}")
    
    print(f"\n11. #F00 → 全格式转换 (应为 RGB 255,0,0)")
    result = color_conversion_api({'hex': '#F00'})
    print(f"结果: {result}")
    
    print("\n--- 颜色名称查询测试 ---")
    print(f"\n12. CSS颜色总数: {len(CSS_COLORS)} 个")
    print(f"   前10个颜色名: {sorted(CSS_COLORS.keys())[:10]}")
    
    print(f"\n13. 颜色名 'red' → HEX (应为 #FF0000)")
    result = ColorConverter.name_to_hex('red')
    print(f"结果: {result}")
    
    print(f"\n14. 颜色名 'blue' → 全格式转换")
    result = color_conversion_api({'name': 'blue'})
    print(f"结果: {result}")
    
    print(f"\n15. 颜色名 'sky blue' (带空格) → 全格式转换")
    result = color_conversion_api({'name': 'sky blue'})
    print(f"结果: {result}")
    
    print(f"\n16. 直接字符串输入 'gold'")
    result = color_conversion_api('gold')
    print(f"结果: {result}")
    
    print(f"\n17. HEX #FF0000 → 颜色名 (应为 red)")
    result = ColorConverter.hex_to_name('#FF0000')
    print(f"结果: {result}")
    
    print(f"\n--- 批量转换测试 ---")
    batch_input = ['red', 'blue', {'hex': '#00FF00'}, {'rgb': {'r': 255, 'g': 255, 'b': 0}}]
    print(f"\n18. 批量输入: {batch_input}")
    result = color_conversion_api(batch_input)
    print(f"批量结果数: {len(result['results'])}")
    for i, item in enumerate(result['results']):
        print(f"  [{i}] 输入: {item['input']} → 成功: {item['result']['success']}")
    
    print(f"\n--- 调色板推荐测试 ---")
    print(f"\n19. 基于 'red' 的互补色调色板")
    result = color_conversion_api({'palette': {'base': 'red', 'type': 'complementary'}})
    print(f"结果: {result['palette']}")
    
    print(f"\n20. 基于 #3498DB 的类似色调色板")
    result = color_conversion_api({'palette': {'base': '#3498DB', 'type': 'analogous'}})
    print(f"结果: {result['palette']}")
    
    print(f"\n21. 基于 'gold' 的三角色调色板")
    result = color_conversion_api({'palette': {'base': 'gold', 'type': 'triadic'}})
    print(f"结果: {result['palette']}")
    
    print(f"\n22. 基于 #2ECC71 的单色调色板")
    result = color_conversion_api({'palette': {'base': '#2ECC71', 'type': 'monochromatic'}})
    print(f"结果: {result['palette']}")
    
    print(f"\n23. 基于 'purple' 的深浅渐变调色板")
    result = color_conversion_api({'palette': {'base': 'purple', 'type': 'shades'}})
    print(f"结果: {result['palette']}")
    
    print(f"\n24. 可用调色板类型: complementary, analogous, triadic, split_complementary,")
    print(f"    tetradic, square, monochromatic, shades")
