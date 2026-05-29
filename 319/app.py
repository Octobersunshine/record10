import os
from flask import Flask, request, jsonify
from captcha_gen import CAPTCHA_TYPES
from captcha_cache import captcha_cache

app = Flask(__name__)


@app.route('/captcha', methods=['GET'])
def get_captcha():
    captcha_type = request.args.get('type', 'text')
    if captcha_type not in CAPTCHA_TYPES:
        return jsonify({
            'error': 'invalid type, must be one of: %s' % ', '.join(CAPTCHA_TYPES.keys())
        }), 400

    length = request.args.get('length', type=int)
    if captcha_type == 'text' and length is not None and not (4 <= length <= 6):
        return jsonify({'error': 'length must be between 4 and 6'}), 400

    generator = CAPTCHA_TYPES[captcha_type]

    if captcha_type == 'text':
        code, img_b64 = generator(length=length)
        token = captcha_cache.put(code, captcha_type='text')
        ttl = int(os.getenv('CAPTCHA_TTL', '300'))
        return jsonify({
            'type': 'text',
            'token': token,
            'captcha_image': 'data:image/png;base64,' + img_b64,
            'expires_in': ttl
        })

    elif captcha_type == 'arithmetic':
        answer, img_b64 = generator()
        token = captcha_cache.put(answer, captcha_type='arithmetic')
        ttl = int(os.getenv('CAPTCHA_TTL', '300'))
        return jsonify({
            'type': 'arithmetic',
            'token': token,
            'captcha_image': 'data:image/png;base64,' + img_b64,
            'expires_in': ttl
        })

    elif captcha_type == 'slide':
        gap_x, slide_data = generator()
        token = captcha_cache.put(gap_x, captcha_type='slide')
        ttl = int(os.getenv('CAPTCHA_TTL', '300'))
        return jsonify({
            'type': 'slide',
            'token': token,
            'background_image': slide_data['background'],
            'slider_image': slide_data['slider'],
            'block_size': slide_data['block_size'],
            'height': slide_data['height'],
            'expires_in': ttl
        })


@app.route('/captcha/verify', methods=['POST'])
def verify_captcha():
    data = request.get_json(silent=True) or {}
    token = data.get('token', '')
    code = data.get('code', '')
    if not token or code == '':
        return jsonify({'error': 'token and code are required'}), 400

    tolerance = data.get('tolerance', 5)
    ok, msg = captcha_cache.verify(token, str(code), tolerance=tolerance)
    status = 200 if ok else 400
    return jsonify({'success': ok, 'message': msg}), status


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=debug, port=port, use_reloader=not debug)
