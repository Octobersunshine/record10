import random
import string
import io
import math
import base64
from PIL import Image, ImageDraw, ImageFont


CHARS = string.ascii_letters + string.digits


def _random_color(start=0, end=255):
    return tuple(random.randint(start, end) for _ in range(3))


def _load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        return ImageFont.load_default(size=size)


def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_text_captcha(length=None, width=160, height=60):
    length = length or random.randint(4, 6)
    code = ''.join(random.choices(CHARS, k=length))

    img = Image.new('RGB', (width, height), _random_color(200, 255))
    draw = ImageDraw.Draw(img)

    font = _load_font(random.randint(28, 38))
    char_width = width // (length + 1)
    for i, ch in enumerate(code):
        x = char_width * i + random.randint(8, 16)
        y = random.randint(4, 14)
        draw.text((x, y), ch, font=font, fill=_random_color(10, 120))

    _add_noise(draw, width, height)

    return code, _img_to_b64(img)


def generate_arithmetic_captcha(width=200, height=60):
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    op = random.choice(['+', '-', '×'])
    if op == '+':
        answer = a + b
    elif op == '-':
        a, b = max(a, b), min(a, b)
        answer = a - b
    else:
        a = random.randint(2, 12)
        b = random.randint(2, 9)
        answer = a * b

    expr = '%d %s %d = ?' % (a, op, b)

    img = Image.new('RGB', (width, height), _random_color(200, 255))
    draw = ImageDraw.Draw(img)

    font = _load_font(random.randint(28, 36))
    bbox = draw.textbbox((0, 0), expr, font=font)
    tw = bbox[2] - bbox[0]
    x = (width - tw) // 2
    y = random.randint(6, 14)
    draw.text((x, y), expr, font=font, fill=_random_color(10, 120))

    _add_noise(draw, width, height)

    return str(answer), _img_to_b64(img)


def generate_slide_captcha(width=300, height=160, block_size=50):
    bg = _generate_background(width, height)
    gap_x = random.randint(block_size + 20, width - block_size - 20)
    gap_y = random.randint(20, height - block_size - 20)

    mask = _make_puzzle_mask(block_size)

    bg_with_hole = bg.copy()
    bg_draw = ImageDraw.Draw(bg_with_hole)
    for py in range(block_size):
        for px in range(block_size):
            if mask[py][px]:
                cx, cy = gap_x + px, gap_y + py
                if 0 <= cx < width and 0 <= cy < height:
                    bg_draw.point((cx, cy), fill=(60, 60, 60))

    block_img = Image.new('RGBA', (block_size + 10, block_size + 10), (0, 0, 0, 0))
    block_draw = ImageDraw.Draw(block_img)
    for py in range(block_size):
        for px in range(block_size):
            if mask[py][px]:
                cx, cy = gap_x + px, gap_y + py
                if 0 <= cx < width and 0 <= cy < height:
                    block_draw.point((px + 5, py + 5), fill=(255, 255, 255, 200))

    return str(gap_x), {
        'background': 'data:image/png;base64,' + _img_to_b64(bg_with_hole),
        'slider': 'data:image/png;base64,' + _img_to_b64(block_img),
        'block_size': block_size,
        'height': height
    }


def _add_noise(draw, width, height):
    for _ in range(random.randint(4, 8)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=_random_color(60, 180), width=1)
    for _ in range(random.randint(40, 80)):
        x, y = random.randint(0, width), random.randint(0, height)
        draw.point((x, y), fill=_random_color(30, 160))


def _generate_background(width, height):
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        for x in range(width):
            r = random.randint(120, 200)
            g = random.randint(120, 200)
            b = random.randint(120, 200)
            draw.point((x, y), fill=(r, g, b))
    for _ in range(30):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=_random_color(80, 160), width=1)
    for _ in range(5):
        cx, cy = random.randint(20, width - 20), random.randint(20, height - 20)
        r = random.randint(10, 30)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_random_color(100, 180))
    return img


def _make_puzzle_mask(size):
    tab_size = size // 4
    tab_r = tab_size // 2
    mask = [[False] * size for _ in range(size)]

    for y in range(size):
        for x in range(size):
            if x == 0 or x == size - 1 or y == 0 or y == size - 1:
                mask[y][x] = True
                continue
            dx = x - size // 2
            dy = y - size // 2
            if abs(dx) < size // 2 and abs(dy) < size // 2:
                mask[y][x] = True

    tab_cx = size
    tab_cy = size // 2
    for y in range(size):
        for x in range(size):
            dist = math.sqrt((x - tab_cx) ** 2 + (y - tab_cy) ** 2)
            if dist <= tab_r:
                if 0 <= y < size and 0 <= x < size:
                    mask[y][x] = True

    tab_cx2 = size // 2
    tab_cy2 = size
    for y in range(size):
        for x in range(size):
            dist = math.sqrt((x - tab_cx2) ** 2 + (y - tab_cy2) ** 2)
            if dist <= tab_r:
                if 0 <= y < size and 0 <= x < size:
                    mask[y][x] = True

    return mask


CAPTCHA_TYPES = {
    'text': generate_text_captcha,
    'arithmetic': generate_arithmetic_captcha,
    'slide': generate_slide_captcha,
}
