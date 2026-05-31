import io
import base64
import requests
from PIL import Image


def create_test_image(format='PNG', size=(100, 100), color=(255, 0, 0), with_alpha=False):
    if with_alpha and format in ('PNG', 'WEBP'):
        img = Image.new('RGBA', size, color + (128,))
    else:
        img = Image.new('RGB', size, color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer


def test_rgba_to_jpeg_warning():
    print('\n测试: RGBA透明图片 -> JPEG (警告验证)')
    
    url = 'http://127.0.0.1:5000/convert'
    image_buffer = create_test_image(format='PNG', with_alpha=True)
    
    files = {'image': ('test_rgba.png', image_buffer, 'image/png')}
    data = {'format': 'JPEG'}
    
    try:
        response = requests.post(url, files=files, data=data)
        result = response.json()
        
        if response.status_code == 200 and result.get('success'):
            print(f'  ✓ 成功! 状态码: {response.status_code}')
            
            if 'warnings' in result and len(result['warnings']) > 0:
                print(f'  ✓ 警告信息存在:')
                for warning in result['warnings']:
                    print(f'    - {warning}')
                if '透明' in result['warnings'][0] or 'alpha' in result['warnings'][0].lower():
                    print(f'  ✓ 透明通道警告内容正确!')
                    return True
                else:
                    print(f'  ✗ 警告内容不正确!')
                    return False
            else:
                print(f'  ✗ 缺少警告信息!')
                return False
        else:
            print(f'  ✗ 失败! 状态码: {response.status_code}')
            print(f'    错误: {result.get("error", "未知错误")}')
            return False
    except Exception as e:
        print(f'  ✗ 异常: {str(e)}')
        return False


def test_conversion(source_format, target_format):
    print(f'\n测试: {source_format} -> {target_format}')
    
    url = 'http://127.0.0.1:5000/convert'
    image_buffer = create_test_image(format=source_format)
    
    files = {'image': (f'test.{source_format.lower()}', image_buffer, f'image/{source_format.lower()}')}
    data = {'format': target_format}
    
    try:
        response = requests.post(url, files=files, data=data)
        result = response.json()
        
        if response.status_code == 200 and result.get('success'):
            print(f'  ✓ 成功! 状态码: {response.status_code}')
            print(f'    目标格式: {result["format"]}')
            print(f'    MIME类型: {result["mime_type"]}')
            print(f'    Base64长度: {len(result["data"])} 字符')
            
            decoded = base64.b64decode(result['data'])
            img = Image.open(io.BytesIO(decoded))
            print(f'    验证格式: {img.format}')
            print(f'    图像尺寸: {img.size}')
            
            if img.format == target_format or (target_format == 'JPEG' and img.format == 'JPEG'):
                print(f'    ✓ 格式验证通过!')
                return True
            else:
                print(f'    ✗ 格式不匹配!')
                return False
        else:
            print(f'  ✗ 失败! 状态码: {response.status_code}')
            print(f'    错误: {result.get("error", "未知错误")}')
            return False
    except Exception as e:
        print(f'  ✗ 异常: {str(e)}')
        return False


def test_supported_formats():
    print('\n测试: 获取支持的格式列表')
    url = 'http://127.0.0.1:5000/formats'
    
    try:
        response = requests.get(url)
        result = response.json()
        print(f'  ✓ 成功! 支持的格式: {result["supported_formats"]}')
        return True
    except Exception as e:
        print(f'  ✗ 异常: {str(e)}')
        return False


def test_error_cases():
    print('\n测试: 错误处理')
    url = 'http://127.0.0.1:5000/convert'
    
    print('  1. 缺少图片文件')
    response = requests.post(url, data={'format': 'PNG'})
    print(f'     状态码: {response.status_code}, 错误: {response.json()["error"]}')
    
    print('  2. 缺少目标格式')
    image_buffer = create_test_image()
    files = {'image': ('test.png', image_buffer, 'image/png')}
    response = requests.post(url, files=files)
    print(f'     状态码: {response.status_code}, 错误: {response.json()["error"]}')
    
    print('  3. 不支持的格式')
    image_buffer = create_test_image()
    files = {'image': ('test.png', image_buffer, 'image/png')}
    response = requests.post(url, files=files, data={'format': 'GIF'})
    print(f'     状态码: {response.status_code}, 错误: {response.json()["error"]}')
    
    return True


def test_quality_parameter():
    print('\n测试: 质量参数调节 (JPEG/WebP)')
    
    url = 'http://127.0.0.1:5000/convert'
    results = []
    
    for quality in [10, 50, 95]:
        image_buffer = create_test_image(format='PNG')
        files = {'image': ('test.png', image_buffer, 'image/png')}
        data = {'format': 'JPEG', 'quality': str(quality)}
        
        try:
            response = requests.post(url, files=files, data=data)
            result = response.json()
            
            if response.status_code == 200 and result.get('success'):
                data_size = len(result['data'])
                print(f'  ✓ 质量={quality}: Base64长度={data_size} 字符')
                results.append(True)
            else:
                print(f'  ✗ 质量={quality}: 失败')
                results.append(False)
        except Exception as e:
            print(f'  ✗ 质量={quality}: 异常 - {str(e)}')
            results.append(False)
    
    return all(results)


def test_batch_convert():
    print('\n测试: 批量转换 (3张图片)')
    
    url = 'http://127.0.0.1:5000/batch-convert'
    
    images = [
        ('img1.png', create_test_image(format='PNG', color=(255, 0, 0))),
        ('img2.png', create_test_image(format='PNG', color=(0, 255, 0))),
        ('img3.png', create_test_image(format='PNG', color=(0, 0, 255))),
    ]
    
    files = []
    for filename, buffer in images:
        files.append(('images', (filename, buffer, 'image/png')))
    
    data = {'format': 'JPEG', 'quality': '80'}
    
    try:
        response = requests.post(url, files=files, data=data)
        result = response.json()
        
        if response.status_code == 200:
            print(f'  ✓ 成功! 总数: {result["total"]}, 成功: {result["success_count"]}')
            for idx, item in enumerate(result['results']):
                if item['success']:
                    print(f'    - {item["filename"]}: {item["format"]}, {len(item["data"])} 字符')
                else:
                    print(f'    - {item["filename"]}: 失败 - {item["error"]}')
            
            return result['success_count'] == len(images)
        else:
            print(f'  ✗ 失败! 状态码: {response.status_code}')
            return False
    except Exception as e:
        print(f'  ✗ 异常: {str(e)}')
        return False


def test_detect_format_single():
    print('\n测试: 单张图片格式检测')
    
    url = 'http://127.0.0.1:5000/detect-format'
    image_buffer = create_test_image(format='PNG', size=(200, 150))
    
    files = {'image': ('test.png', image_buffer, 'image/png')}
    
    try:
        response = requests.post(url, files=files)
        result = response.json()
        
        if response.status_code == 200 and result.get('success'):
            print(f'  ✓ 成功!')
            print(f'    格式: {result["format"]}')
            print(f'    模式: {result["mode"]}')
            print(f'    尺寸: {result["size"]["width"]}x{result["size"]["height"]}')
            print(f'    透明通道: {result["has_alpha"]}')
            return True
        else:
            print(f'  ✗ 失败! 状态码: {response.status_code}')
            return False
    except Exception as e:
        print(f'  ✗ 异常: {str(e)}')
        return False


def test_detect_format_batch():
    print('\n测试: 批量格式检测 (3张图片)')
    
    url = 'http://127.0.0.1:5000/detect-format'
    
    images = [
        ('img1.png', create_test_image(format='PNG', size=(100, 100))),
        ('img2.jpg', create_test_image(format='JPEG', size=(200, 150))),
        ('img3.webp', create_test_image(format='WEBP', size=(300, 200))),
    ]
    
    files = []
    for filename, buffer in images:
        files.append(('images', (filename, buffer, 'image/'+filename.split('.')[-1])))
    
    try:
        response = requests.post(url, files=files)
        result = response.json()
        
        if response.status_code == 200:
            print(f'  ✓ 成功! 总数: {result["total"]}')
            for item in result['results']:
                if item['success']:
                    print(f'    - {item["filename"]}: {item["format"]}, {item["size"]["width"]}x{item["size"]["height"]}')
                else:
                    print(f'    - {item["filename"]}: 失败 - {item["error"]}')
            return result['total'] == len(images)
        else:
            print(f'  ✗ 失败! 状态码: {response.status_code}')
            return False
    except Exception as e:
        print(f'  ✗ 异常: {str(e)}')
        return False


if __name__ == '__main__':
    print('=' * 50)
    print('图像格式转换API 测试脚本')
    print('=' * 50)
    
    test_supported_formats()
    
    conversions = [
        ('PNG', 'JPEG'),
        ('PNG', 'WEBP'),
        ('PNG', 'BMP'),
        ('JPEG', 'PNG'),
        ('JPEG', 'WEBP'),
        ('BMP', 'PNG'),
        ('BMP', 'JPEG'),
        ('WEBP', 'PNG'),
    ]
    
    results = []
    for source, target in conversions:
        results.append(test_conversion(source, target))
    
    results.append(test_rgba_to_jpeg_warning())
    results.append(test_quality_parameter())
    results.append(test_batch_convert())
    results.append(test_detect_format_single())
    results.append(test_detect_format_batch())
    
    test_error_cases()
    
    print('\n' + '=' * 50)
    passed = sum(results)
    total = len(results)
    print(f'测试完成: {passed}/{total} 通过')
    print('=' * 50)
