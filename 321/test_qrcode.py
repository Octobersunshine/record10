import base64
import os
from qrcode_utils import generate_qrcode_base64, decode_qrcode_image, save_qrcode_image


def test_generate():
    print('=== 测试二维码生成 ===')
    test_text = 'https://www.example.com'
    base64_str = generate_qrcode_base64(test_text)
    print(f'输入文本: {test_text}')
    print(f'Base64 长度: {len(base64_str)} 字符')
    print(f'Base64 前缀: {base64_str[:50]}...')
    assert len(base64_str) > 500, 'Base64 字符串长度过短'
    try:
        img_bytes = base64.b64decode(base64_str)
        assert len(img_bytes) > 300, '解码后的图片数据过短'
        print('Base64 解码成功，图片数据有效')
    except Exception as e:
        print(f'Base64 解码失败: {e}')
        raise
    print('生成测试通过!\n')
    return base64_str


def test_save_and_decode():
    print('=== 测试二维码保存和解码 ===')
    test_text = 'Hello, 二维码测试! 12345'
    test_file = 'test_qr.png'

    save_qrcode_image(test_text, test_file)
    assert os.path.exists(test_file), '二维码图片未保存成功'
    print(f'二维码已保存到: {test_file}')

    with open(test_file, 'rb') as f:
        img_bytes = f.read()

    decoded_text = decode_qrcode_image(img_bytes)
    print(f'原始文本: {test_text}')
    print(f'解码文本: {decoded_text}')
    assert decoded_text == test_text, f'解码文本不匹配! 期望: {test_text}, 实际: {decoded_text}'
    print('解码测试通过!\n')

    os.remove(test_file)
    print('测试文件已清理\n')


def test_generate_with_options():
    print('=== 测试自定义参数生成 ===')
    test_text = '自定义参数测试'

    for ec in ['L', 'M', 'Q', 'H']:
        base64_str = generate_qrcode_base64(test_text, error_correction=ec, box_size=12, border=2)
        print(f'纠错级别 {ec}: Base64 长度 {len(base64_str)}')

    print('自定义参数测试通过!\n')


if __name__ == '__main__':
    try:
        test_generate()
        test_save_and_decode()
        test_generate_with_options()
        print('=' * 50)
        print('所有测试通过! ✓')
        print('=' * 50)
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()
