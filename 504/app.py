import io
import base64
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

ALLOWED_FORMATS = {
    'JPEG': ['JPEG', 'JPG'],
    'PNG': ['PNG'],
    'WEBP': ['WEBP'],
    'BMP': ['BMP']
}

MIME_TYPES = {
    'JPEG': 'image/jpeg',
    'PNG': 'image/png',
    'WEBP': 'image/webp',
    'BMP': 'image/bmp'
}


def normalize_format(target_format):
    target_format_upper = target_format.upper()
    for canonical, aliases in ALLOWED_FORMATS.items():
        if target_format_upper in aliases:
            return canonical
    return None


def convert_image(image_file, target_format, quality=85):
    img = Image.open(image_file)
    warnings = []
    original_format = img.format
    original_mode = img.mode
    
    has_alpha = False
    if img.mode in ('RGBA', 'LA'):
        has_alpha = True
    elif img.mode == 'P':
        has_alpha = 'transparency' in img.info
    
    if target_format == 'JPEG':
        if has_alpha or img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            if img.mode in ('RGBA', 'LA'):
                alpha = img.split()[-1]
                background.paste(img, mask=alpha)
            else:
                background.paste(img)
            img = background
            warnings.append('JPEG格式不支持透明通道，已自动将透明背景替换为白色')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
    elif target_format in ('PNG', 'WEBP'):
        if img.mode == 'P':
            img = img.convert('RGBA')
        elif img.mode not in ('RGBA', 'RGB', 'LA', 'L'):
            img = img.convert('RGBA')
    elif target_format == 'BMP':
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
    
    output_buffer = io.BytesIO()
    save_format = 'JPEG' if target_format == 'JPEG' else target_format
    
    save_kwargs = {'format': save_format}
    if target_format in ('JPEG', 'WEBP'):
        quality = max(1, min(100, quality))
        save_kwargs['quality'] = quality
    
    img.save(output_buffer, **save_kwargs)
    output_buffer.seek(0)
    
    return output_buffer, warnings, {
        'original_format': original_format,
        'original_mode': original_mode,
        'size': img.size
    }


@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return jsonify({'error': '未找到上传的图片文件'}), 400
    
    target_format = request.form.get('format', '').strip()
    if not target_format:
        return jsonify({'error': '请指定目标格式'}), 400
    
    canonical_format = normalize_format(target_format)
    if not canonical_format:
        return jsonify({
            'error': f'不支持的格式: {target_format}',
            'supported_formats': ['JPEG', 'PNG', 'WebP', 'BMP']
        }), 400
    
    quality = int(request.form.get('quality', 85))
    quality = max(1, min(100, quality))
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    try:
        output_buffer, warnings, info = convert_image(image_file, canonical_format, quality)
        image_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        response = {
            'success': True,
            'format': canonical_format,
            'mime_type': MIME_TYPES[canonical_format],
            'data': image_base64,
            'quality': quality if canonical_format in ('JPEG', 'WEBP') else None,
            'info': info
        }
        if warnings:
            response['warnings'] = warnings
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'图片转换失败: {str(e)}'}), 500


@app.route('/batch-convert', methods=['POST'])
def batch_convert():
    images = request.files.getlist('images')
    if not images:
        return jsonify({'error': '未找到上传的图片文件'}), 400
    
    target_format = request.form.get('format', '').strip()
    if not target_format:
        return jsonify({'error': '请指定目标格式'}), 400
    
    canonical_format = normalize_format(target_format)
    if not canonical_format:
        return jsonify({
            'error': f'不支持的格式: {target_format}',
            'supported_formats': ['JPEG', 'PNG', 'WebP', 'BMP']
        }), 400
    
    quality = int(request.form.get('quality', 85))
    quality = max(1, min(100, quality))
    
    results = []
    success_count = 0
    
    for idx, image_file in enumerate(images):
        if image_file.filename == '':
            continue
        
        try:
            output_buffer, warnings, info = convert_image(image_file, canonical_format, quality)
            image_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            result = {
                'index': idx,
                'filename': image_file.filename,
                'success': True,
                'format': canonical_format,
                'mime_type': MIME_TYPES[canonical_format],
                'data': image_base64,
                'quality': quality if canonical_format in ('JPEG', 'WEBP') else None,
                'info': info
            }
            if warnings:
                result['warnings'] = warnings
            success_count += 1
        except Exception as e:
            result = {
                'index': idx,
                'filename': image_file.filename,
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    return jsonify({
        'success': success_count == len(results),
        'total': len(results),
        'success_count': success_count,
        'results': results
    })


@app.route('/detect-format', methods=['POST'])
def detect_format():
    images = request.files.getlist('images')
    single_image = 'image' in request.files
    
    if not images and not single_image:
        return jsonify({'error': '未找到上传的图片文件'}), 400
    
    if single_image:
        images = [request.files['image']]
    
    results = []
    
    for idx, image_file in enumerate(images):
        if image_file.filename == '':
            continue
        
        try:
            img = Image.open(image_file)
            result = {
                'index': idx,
                'filename': image_file.filename,
                'success': True,
                'format': img.format,
                'mode': img.mode,
                'size': {
                    'width': img.size[0],
                    'height': img.size[1]
                },
                'has_alpha': img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
            }
        except Exception as e:
            result = {
                'index': idx,
                'filename': image_file.filename,
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    if single_image and len(results) == 1:
        return jsonify(results[0])
    
    return jsonify({
        'total': len(results),
        'results': results
    })


@app.route('/formats', methods=['GET'])
def supported_formats():
    return jsonify({
        'supported_formats': ['JPEG', 'PNG', 'WebP', 'BMP'],
        'description': '支持JPEG、PNG、WebP、BMP四种格式之间的互转',
        'features': [
            '单张图片转换',
            '批量图片转换',
            '质量参数调节 (JPEG/WebP: 1-100)',
            '图片格式检测',
            '透明通道自动处理'
        ]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
