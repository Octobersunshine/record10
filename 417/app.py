from flask import Flask, request, jsonify
from mask_utils import mask_json_data, unmask_json_data

app = Flask(__name__)


@app.route('/api/mask', methods=['POST'])
def mask_data():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        data = request_data.get('data')
        field_mappings = request_data.get('field_mappings', {})

        if data is None:
            return jsonify({'error': 'Missing required field: data'}), 400

        masked_data = mask_json_data(data, field_mappings)

        return jsonify({
            'success': True,
            'data': masked_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/decrypt', methods=['POST'])
def decrypt_data():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        data = request_data.get('data')
        field_mappings = request_data.get('field_mappings', {})

        if data is None:
            return jsonify({'error': 'Missing required field: data'}), 400

        decrypted_data = unmask_json_data(data, field_mappings)

        return jsonify({
            'success': True,
            'data': decrypted_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
