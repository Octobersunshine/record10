import io
import base64
import qrcode
import cv2
import numpy as np
from PIL import Image

MAX_QR_CONTENT_LENGTHS = {
    'L': 2953,
    'M': 2331,
    'Q': 1663,
    'H': 1273,
}

MAX_QR_CONTENT_LENGTH = 2953
DEFAULT_FILL_COLOR = '#000000'
DEFAULT_BACK_COLOR = '#FFFFFF'
LOGO_SIZE_RATIO = 0.22


def _get_max_length(error_correction: str) -> int:
    return MAX_QR_CONTENT_LENGTHS.get(error_correction.upper(), MAX_QR_CONTENT_LENGTHS['M'])


def _parse_color(color: str) -> tuple:
    if not color:
        return None
    color = color.strip()
    if color.startswith('#'):
        color = color[1:]
        if len(color) == 3:
            color = ''.join(c * 2 for c in color)
        if len(color) == 6:
            return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    return color


def _embed_logo(qr_img: Image.Image, logo_bytes: bytes, logo_ratio: float = LOGO_SIZE_RATIO) -> Image.Image:
    qr_width, qr_height = qr_img.size
    logo_max_size = int(min(qr_width, qr_height) * logo_ratio)

    logo_img = Image.open(io.BytesIO(logo_bytes))
    if logo_img.mode != 'RGBA':
        logo_img = logo_img.convert('RGBA')

    logo_width, logo_height = logo_img.size
    logo_scale = min(logo_max_size / logo_width, logo_max_size / logo_height)
    new_width = int(logo_width * logo_scale)
    new_height = int(logo_height * logo_scale)
    logo_img = logo_img.resize((new_width, new_height), Image.LANCZOS)

    pos_x = (qr_width - new_width) // 2
    pos_y = (qr_height - new_height) // 2

    if qr_img.mode != 'RGBA':
        qr_img = qr_img.convert('RGBA')

    qr_img.paste(logo_img, (pos_x, pos_y), logo_img)
    return qr_img


def generate_qrcode_base64(
    text: str,
    error_correction: str = 'M',
    box_size: int = 10,
    border: int = 4,
    fill_color: str = DEFAULT_FILL_COLOR,
    back_color: str = DEFAULT_BACK_COLOR,
    logo_bytes: bytes = None,
    logo_ratio: float = LOGO_SIZE_RATIO,
) -> str:
    if not text:
        raise ValueError('文本内容不能为空')

    ec_key = error_correction.upper()
    use_logo = logo_bytes is not None

    if use_logo and ec_key not in ('H', 'Q'):
        ec_key = 'H'
    else:
        ec_key = ec_key

    max_len = _get_max_length(ec_key)
    text_len = len(text)

    if text_len > max_len:
        raise ValueError(
            f'文本内容过长（{text_len} 字符），超出当前纠错级别 {ec_key} 的最大容量（{max_len} 字符）。'
            f'请精简文本内容，或降低纠错级别后重试。'
        )

    error_levels = {
        'L': qrcode.constants.ERROR_CORRECT_L,
        'M': qrcode.constants.ERROR_CORRECT_M,
        'Q': qrcode.constants.ERROR_CORRECT_Q,
        'H': qrcode.constants.ERROR_CORRECT_H,
    }
    error_level = error_levels.get(ec_key, qrcode.constants.ERROR_CORRECT_M)

    fill_rgb = _parse_color(fill_color)
    back_rgb = _parse_color(back_color)

    if fill_rgb is None:
        fill_rgb = DEFAULT_FILL_COLOR
    if back_rgb is None:
        back_rgb = DEFAULT_BACK_COLOR

    qr = qrcode.QRCode(
        version=None,
        error_correction=error_level,
        box_size=box_size,
        border=border,
    )
    qr.add_data(text)

    try:
        qr.make(fit=True)
    except ValueError as e:
        if 'version' in str(e).lower():
            raise ValueError(
                f'文本内容过长（{text_len} 字符），超出二维码最大容量。'
                f'当前纠错级别 {ec_key} 最多支持 {max_len} 字符，请精简文本或降低纠错级别。'
            ) from e
        raise

    img = qr.make_image(fill_color=fill_rgb, back_color=back_rgb)

    if use_logo:
        img = _embed_logo(img, logo_bytes, logo_ratio)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_base64


def decode_qrcode_image(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)

    qr_detector = cv2.QRCodeDetector()
    decoded_text, _, _ = qr_detector.detectAndDecode(gray)

    if not decoded_text:
        raise ValueError('未能识别二维码，请确保图片包含清晰的二维码')

    return decoded_text


def save_qrcode_image(
    text: str,
    file_path: str,
    error_correction: str = 'M',
    box_size: int = 10,
    border: int = 4,
    fill_color: str = DEFAULT_FILL_COLOR,
    back_color: str = DEFAULT_BACK_COLOR,
    logo_bytes: bytes = None,
    logo_ratio: float = LOGO_SIZE_RATIO,
) -> None:
    base64_str = generate_qrcode_base64(
        text=text,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
        fill_color=fill_color,
        back_color=back_color,
        logo_bytes=logo_bytes,
        logo_ratio=logo_ratio,
    )
    img_bytes = base64.b64decode(base64_str)
    with open(file_path, 'wb') as f:
        f.write(img_bytes)
