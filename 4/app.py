from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)


class OutlierDetector:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.original_data = self.data.copy()
        self.outliers_mask = None
        self.outlier_indices = []
        self._EPS = 1e-10

    def detect_iqr(self, k=1.5):
        if len(self.data) == 0:
            raise ValueError("Cannot detect outliers in empty data")
        
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        if iqr < self._EPS:
            self.outliers_mask = np.zeros_like(self.data, dtype=bool)
            self.outlier_indices = []
            return {
                'method': 'IQR',
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_count': 0,
                'outlier_indices': [],
                'outlier_values': [],
                'note': 'IQR is near zero, data distribution is approximately constant'
            }
        
        self.outliers_mask = (self.data < lower_bound) | (self.data > upper_bound)
        self.outlier_indices = np.where(self.outliers_mask)[0].tolist()
        return {
            'method': 'IQR',
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': int(np.sum(self.outliers_mask)),
            'outlier_indices': self.outlier_indices,
            'outlier_values': self.data[self.outliers_mask].tolist()
        }

    def detect_zscore(self, threshold=3.0):
        if len(self.data) == 0:
            raise ValueError("Cannot detect outliers in empty data")
        
        mean = np.mean(self.data)
        std = np.std(self.data)
        
        if std < self._EPS:
            self.outliers_mask = np.zeros_like(self.data, dtype=bool)
            self.outlier_indices = []
            return {
                'method': 'Z-score',
                'mean': float(mean),
                'std': float(std),
                'threshold': float(threshold),
                'outlier_count': 0,
                'outlier_indices': [],
                'outlier_values': [],
                'note': 'Standard deviation is near zero, all data values are approximately constant'
            }
        
        z_scores = (self.data - mean) / std
        self.outliers_mask = np.abs(z_scores) > threshold
        self.outlier_indices = np.where(self.outliers_mask)[0].tolist()
        
        return {
            'method': 'Z-score',
            'mean': float(mean),
            'std': float(std),
            'threshold': float(threshold),
            'outlier_count': int(np.sum(self.outliers_mask)),
            'outlier_indices': self.outlier_indices,
            'outlier_values': self.data[self.outliers_mask].tolist(),
            'z_scores': z_scores.tolist()
        }

    def detect_sliding_window(self, window_size=5, k=2.0, center=False):
        if len(self.data) == 0:
            raise ValueError("Cannot detect outliers in empty data")
        
        if window_size < 2:
            raise ValueError("Window size must be at least 2")
        
        n = len(self.data)
        self.outliers_mask = np.zeros(n, dtype=bool)
        window_means = np.full(n, np.nan)
        window_stds = np.full(n, np.nan)
        deviations = np.full(n, np.nan)
        
        for i in range(n):
            if center:
                start = max(0, i - window_size // 2)
                end = min(n, i + window_size // 2 + 1)
            else:
                start = max(0, i - window_size)
                end = i
            
            window = self.data[start:end]
            
            if len(window) >= 2:
                window_mean = np.mean(window)
                window_std = np.std(window)
                
                window_means[i] = window_mean
                window_stds[i] = window_std
                
                if window_std < self._EPS:
                    deviation = 0.0
                else:
                    deviation = abs(self.data[i] - window_mean) / window_std
                
                deviations[i] = deviation
                
                if window_std >= self._EPS and deviation > k:
                    self.outliers_mask[i] = True
            else:
                window_means[i] = self.data[i]
                window_stds[i] = 0.0
                deviations[i] = 0.0
        
        self.outlier_indices = np.where(self.outliers_mask)[0].tolist()
        
        return {
            'method': 'Sliding Window',
            'window_size': window_size,
            'k_threshold': float(k),
            'center': center,
            'outlier_count': int(np.sum(self.outliers_mask)),
            'outlier_indices': self.outlier_indices,
            'outlier_values': self.data[self.outliers_mask].tolist(),
            'window_means': window_means.tolist(),
            'window_stds': window_stds.tolist(),
            'deviations': deviations.tolist()
        }

    def handle_outliers(self, method='remove'):
        if self.outliers_mask is None:
            raise ValueError("Please detect outliers first")

        cleaned_data = self.original_data.copy()
        outlier_count = int(np.sum(self.outliers_mask))
        
        if outlier_count == 0:
            return {
                'handling_method': method,
                'original_length': int(len(self.original_data)),
                'cleaned_length': int(len(cleaned_data)),
                'cleaned_data': cleaned_data.tolist(),
                'note': 'No outliers detected, data unchanged',
                'statistics': {
                    'original_mean': float(np.mean(self.original_data)),
                    'original_median': float(np.median(self.original_data)),
                    'original_std': float(np.std(self.original_data)),
                    'cleaned_mean': float(np.mean(cleaned_data)),
                    'cleaned_median': float(np.median(cleaned_data)),
                    'cleaned_std': float(np.std(cleaned_data))
                }
            }

        non_outlier_mask = ~self.outliers_mask
        
        if method == 'remove':
            cleaned_data = cleaned_data[non_outlier_mask]
        elif method == 'mean':
            if np.sum(non_outlier_mask) > 0:
                mean_val = np.mean(cleaned_data[non_outlier_mask])
            else:
                mean_val = np.mean(self.original_data)
            cleaned_data[self.outliers_mask] = mean_val
        elif method == 'median':
            if np.sum(non_outlier_mask) > 0:
                median_val = np.median(cleaned_data[non_outlier_mask])
            else:
                median_val = np.median(self.original_data)
            cleaned_data[self.outliers_mask] = median_val
        else:
            raise ValueError(f"Unknown handling method: {method}")

        return {
            'handling_method': method,
            'original_length': int(len(self.original_data)),
            'cleaned_length': int(len(cleaned_data)),
            'cleaned_data': cleaned_data.tolist(),
            'statistics': {
                'original_mean': float(np.mean(self.original_data)),
                'original_median': float(np.median(self.original_data)),
                'original_std': float(np.std(self.original_data)),
                'cleaned_mean': float(np.mean(cleaned_data)),
                'cleaned_median': float(np.median(cleaned_data)),
                'cleaned_std': float(np.std(cleaned_data))
            }
        }


def parse_input_data(request):
    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in CSV file")
            column = request.form.get('column', numeric_cols[0])
            if column not in numeric_cols:
                raise ValueError(f"Column '{column}' not found or not numeric")
            return df[column].dropna().tolist(), column
        else:
            raise ValueError("Unsupported file format. Please use CSV")
    
    elif request.is_json:
        data = request.get_json()
        if 'data' not in data:
            raise ValueError("JSON must contain 'data' field with numeric list")
        return data['data'], None
    
    else:
        raise ValueError("Please upload a CSV file or send JSON data")


@app.route('/detect', methods=['POST'])
def detect_outliers():
    try:
        data_list, column_name = parse_input_data(request)
        
        if len(data_list) < 2:
            return jsonify({'error': 'At least 2 data points required'}), 400

        detector = OutlierDetector(data_list)
        
        detection_method = request.form.get('method', 'iqr') if 'file' in request.files else \
                          (request.get_json().get('method', 'iqr') if request.is_json else 'iqr')
        
        if detection_method == 'iqr':
            if len(data_list) < 4:
                return jsonify({'error': 'IQR method requires at least 4 data points'}), 400
            k = float(request.form.get('k', 1.5)) if 'file' in request.files else \
                float(request.get_json().get('k', 1.5))
            detection_report = detector.detect_iqr(k)
        elif detection_method == 'zscore':
            threshold = float(request.form.get('threshold', 3.0)) if 'file' in request.files else \
                       float(request.get_json().get('threshold', 3.0))
            detection_report = detector.detect_zscore(threshold)
        elif detection_method == 'sliding_window':
            window_size = int(request.form.get('window_size', 5)) if 'file' in request.files else \
                         int(request.get_json().get('window_size', 5))
            k = float(request.form.get('k', 2.0)) if 'file' in request.files else \
                float(request.get_json().get('k', 2.0))
            center = request.form.get('center', 'false').lower() == 'true' if 'file' in request.files else \
                    (request.get_json().get('center', False) if request.is_json else False)
            detection_report = detector.detect_sliding_window(window_size, k, center)
        else:
            return jsonify({'error': 'Method must be "iqr", "zscore", or "sliding_window"'}), 400

        handling_method = request.form.get('handle', 'remove') if 'file' in request.files else \
                         (request.get_json().get('handle', 'remove') if request.is_json else 'remove')
        
        handling_report = detector.handle_outliers(handling_method)

        response = {
            'success': True,
            'column_name': column_name,
            'detection_report': detection_report,
            'handling_report': handling_report
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Outlier detection service is running'})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Outlier Detection API',
        'version': '1.1',
        'endpoints': {
            '/health': 'GET - Health check',
            '/detect': 'POST - Detect and handle outliers',
        },
        'usage': {
            'method': 'POST /detect',
            'options': {
                'detection_methods': ['iqr', 'zscore', 'sliding_window'],
                'handling_methods': ['remove', 'mean', 'median'],
                'input_formats': ['CSV file upload', 'JSON']
            },
            'method_details': {
                'iqr': {
                    'params': {'k': 'IQR multiplier (default: 1.5)'},
                    'min_data_points': 4
                },
                'zscore': {
                    'params': {'threshold': 'Standard deviation threshold (default: 3.0)'},
                    'min_data_points': 2
                },
                'sliding_window': {
                    'params': {
                        'window_size': 'Size of the sliding window (default: 5)',
                        'k': 'Standard deviation threshold (default: 2.0)',
                        'center': 'Use centered window (default: false)'
                    },
                    'description': 'Time series anomaly detection - compares each point against the previous N points',
                    'min_data_points': 2
                }
            },
            'json_examples': {
                'iqr': {
                    'data': [1, 2, 3, 4, 5, 100],
                    'method': 'iqr',
                    'handle': 'mean',
                    'k': 1.5
                },
                'zscore': {
                    'data': [10, 12, 14, 16, 18, 100],
                    'method': 'zscore',
                    'handle': 'median',
                    'threshold': 2.5
                },
                'sliding_window': {
                    'data': [1, 2, 3, 4, 5, 6, 100, 7, 8, 9],
                    'method': 'sliding_window',
                    'handle': 'mean',
                    'window_size': 5,
                    'k': 2.0,
                    'center': False
                }
            }
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
