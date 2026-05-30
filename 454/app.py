import os
import tempfile
from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from waveform_api import generate_waveform, waveform_to_dict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'wav'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/waveform', methods=['POST'])
def api_waveform():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '只支持WAV格式文件'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        num_points_param = request.form.get('num_points')
        num_points = int(num_points_param) if num_points_param is not None else None
        width = int(request.form.get('width', 800))
        height = int(request.form.get('height', 200))
        downsample_method = request.form.get('method', 'peak')
        channel_strategy = request.form.get('channel_strategy', 'average')
        use_adaptive = request.form.get('use_adaptive', 'true').lower() == 'true'
        max_points_per_pixel = float(request.form.get('max_points_per_pixel', 2.0))
        min_points = int(request.form.get('min_points', 100))
        max_points = int(request.form.get('max_points', 2000))
        include_image = request.form.get('include_image', 'true').lower() == 'true'
        
        svg_stroke_color = request.form.get('svg_stroke_color', '#4a90d9')
        svg_stroke_width = int(request.form.get('svg_stroke_width', 1))
        svg_fill_color = request.form.get('svg_fill_color', 'rgba(74, 144, 217, 0.3)')
        if svg_fill_color.lower() == 'none':
            svg_fill_color = None
        
        result = generate_waveform(
            filepath,
            num_points=num_points,
            width=width,
            height=height,
            downsample_method=downsample_method,
            channel_strategy=channel_strategy,
            use_adaptive_downsample=use_adaptive,
            max_points_per_pixel=max_points_per_pixel,
            min_points=min_points,
            max_points=max_points,
            svg_stroke_color=svg_stroke_color,
            svg_stroke_width=svg_stroke_width,
            svg_fill_color=svg_fill_color,
            include_image=include_image
        )
        
        return jsonify(waveform_to_dict(result))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/waveform/svg', methods=['POST'])
def api_waveform_svg():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '只支持WAV格式文件'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        num_points_param = request.form.get('num_points')
        num_points = int(num_points_param) if num_points_param is not None else None
        width = int(request.form.get('width', 800))
        height = int(request.form.get('height', 200))
        downsample_method = request.form.get('method', 'peak')
        channel_strategy = request.form.get('channel_strategy', 'average')
        use_adaptive = request.form.get('use_adaptive', 'true').lower() == 'true'
        max_points_per_pixel = float(request.form.get('max_points_per_pixel', 2.0))
        min_points = int(request.form.get('min_points', 100))
        max_points = int(request.form.get('max_points', 2000))
        
        svg_stroke_color = request.form.get('svg_stroke_color', '#4a90d9')
        svg_stroke_width = int(request.form.get('svg_stroke_width', 1))
        svg_fill_color = request.form.get('svg_fill_color', 'rgba(74, 144, 217, 0.3)')
        if svg_fill_color.lower() == 'none':
            svg_fill_color = None
        
        result = generate_waveform(
            filepath,
            num_points=num_points,
            width=width,
            height=height,
            downsample_method=downsample_method,
            channel_strategy=channel_strategy,
            use_adaptive_downsample=use_adaptive,
            max_points_per_pixel=max_points_per_pixel,
            min_points=min_points,
            max_points=max_points,
            svg_stroke_color=svg_stroke_color,
            svg_stroke_width=svg_stroke_width,
            svg_fill_color=svg_fill_color,
            include_image=False
        )
        
        svg_content = result.svg_path
        return Response(svg_content, mimetype='image/svg+xml')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': '音频波形图API服务正常'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
