import io
import json
import time
import uuid
import numpy as np
from scipy.sparse import csr_matrix, issparse
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import redis


app = Flask(__name__)


REQUEST_COUNT = Counter('matrix_requests_total', 'Total number of matrix requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('matrix_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
COPY_COUNT = Counter('matrix_copy_total', 'Total number of matrix copies', ['matrix'])
COPY_TIME = Histogram('matrix_copy_seconds', 'Matrix copy time in seconds', ['matrix'])
SPARSE_CONVERSION_COUNT = Counter('sparse_conversion_total', 'Total number of sparse conversions', ['matrix'])
SPARSE_CONVERSION_TIME = Histogram('sparse_conversion_seconds', 'Sparse conversion time in seconds', ['matrix'])
SPARSE_USAGE_COUNT = Counter('sparse_usage_total', 'Total number of sparse multiplication usage')
DENSE_USAGE_COUNT = Counter('dense_usage_total', 'Total number of dense multiplication usage')
DISTRIBUTED_JOBS = Counter('distributed_jobs_total', 'Total number of distributed jobs')
DISTRIBUTED_BLOCKS = Counter('distributed_blocks_total', 'Total number of distributed blocks processed')
DISTRIBUTED_BLOCK_TIME = Histogram('distributed_block_seconds', 'Block processing time in seconds')
DISTRIBUTED_JOB_TIME = Histogram('distributed_job_seconds', 'Job processing time in seconds')


DENSITY_THRESHOLD = 0.1
DISTRIBUTED_THRESHOLD = 2000
BLOCK_SIZE = 1000


redis_client = None
TASK_STREAM = 'matrix_tasks'
RESULT_STREAM = 'matrix_results'
CONSUMER_GROUP = 'matrix_workers'


def init_redis():
    global redis_client
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()
        print('Redis connected successfully')
        try:
            redis_client.xgroup_create(TASK_STREAM, CONSUMER_GROUP, mkstream=True)
        except redis.exceptions.ResponseError:
            pass
        return True
    except Exception as e:
        print(f'Redis connection failed: {e}')
        return False


def ensure_contiguous(arr, name):
    if not arr.flags.c_contiguous:
        start = time.time()
        arr = np.ascontiguousarray(arr)
        COPY_TIME.labels(name).observe(time.time() - start)
        COPY_COUNT.labels(name).inc()
    return arr


def calculate_density(arr):
    non_zero = np.count_nonzero(arr)
    total = arr.size
    return non_zero / total if total > 0 else 0


def convert_to_csr(arr, name):
    start = time.time()
    csr = csr_matrix(arr)
    SPARSE_CONVERSION_TIME.labels(name).observe(time.time() - start)
    SPARSE_CONVERSION_COUNT.labels(name).inc()
    return csr


def matrix_multiply_and_norm(matrix_a, matrix_b, is_numpy=False):
    if is_numpy:
        a = matrix_a
        b = matrix_b
    else:
        a = np.asarray(matrix_a, order='C')
        b = np.asarray(matrix_b, order='C')
    
    a = ensure_contiguous(a, 'A')
    b = ensure_contiguous(b, 'B')
    
    if issparse(a) or issparse(b):
        use_sparse = True
    else:
        density_a = calculate_density(a)
        density_b = calculate_density(b)
        avg_density = (density_a + density_b) / 2
        use_sparse = avg_density < DENSITY_THRESHOLD
    
    if use_sparse:
        SPARSE_USAGE_COUNT.inc()
        if not issparse(a):
            a_sparse = convert_to_csr(a, 'A')
        else:
            a_sparse = a
        if not issparse(b):
            b_sparse = convert_to_csr(b, 'B')
        else:
            b_sparse = b
        
        result_sparse = a_sparse.dot(b_sparse)
        result = result_sparse.toarray()
        frobenius_norm = np.sqrt(result_sparse.power(2).sum())
    else:
        DENSE_USAGE_COUNT.inc()
        result = np.dot(a, b)
        frobenius_norm = np.linalg.norm(result, 'fro')
    
    return {
        'result': result.tolist(),
        'frobenius_norm': float(frobenius_norm),
        'used_sparse': use_sparse,
        'used_distributed': False
    }


def serialize_matrix(arr):
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return json.dumps({
        'type': 'numpy',
        'data': buffer.read().hex()
    })


def deserialize_matrix(data_str):
    data = json.loads(data_str)
    if data['type'] == 'numpy':
        buffer = io.BytesIO(bytes.fromhex(data['data']))
        return np.load(buffer)
    raise ValueError(f'Unknown matrix type: {data.get("type")}')


def split_matrix_blocks(a, b, block_size):
    m, k = a.shape
    k2, n = b.shape
    
    blocks_a = []
    blocks_b = []
    
    for i in range(0, m, block_size):
        row_blocks = []
        for l in range(0, k, block_size):
            row_blocks.append(a[i:i+block_size, l:l+block_size])
        blocks_a.append(row_blocks)
    
    for l in range(0, k, block_size):
        col_blocks = []
        for j in range(0, n, block_size):
            col_blocks.append(b[l:l+block_size, j:j+block_size])
        blocks_b.append(col_blocks)
    
    return blocks_a, blocks_b, m, n


def distributed_multiply(a, b, block_size=BLOCK_SIZE):
    if redis_client is None:
        raise RuntimeError('Redis not connected. Cannot use distributed mode.')
    
    job_id = str(uuid.uuid4())[:12]
    start_time = time.time()
    
    blocks_a, blocks_b, m, n = split_matrix_blocks(a, b, block_size)
    
    num_blocks_i = len(blocks_a)
    num_blocks_k = len(blocks_a[0])
    num_blocks_j = len(blocks_b[0])
    
    total_blocks = num_blocks_i * num_blocks_j
    
    for i in range(num_blocks_i):
        for j in range(num_blocks_j):
            block_a = blocks_a[i][0]
            for k_idx in range(1, num_blocks_k):
                block_a = np.hstack((block_a, blocks_a[i][k_idx]))
            
            block_b = blocks_b[0][j]
            for k_idx in range(1, num_blocks_k):
                block_b = np.vstack((block_b, blocks_b[k_idx][j]))
            
            task_msg = {
                'job_id': job_id,
                'block_i': str(i),
                'block_j': str(j),
                'block_a': serialize_matrix(block_a),
                'block_b': serialize_matrix(block_b),
                'timestamp': str(time.time())
            }
            redis_client.xadd(TASK_STREAM, task_msg)
    
    DISTRIBUTED_JOBS.inc()
    
    result_blocks = {}
    completed_blocks = 0
    last_id = '0-0'
    
    while completed_blocks < total_blocks:
        messages = redis_client.xread({RESULT_STREAM: last_id}, count=100, block=1000)
        
        for stream_name, stream_messages in messages:
            for message_id, message_data in stream_messages:
                last_id = message_id
                
                msg_job_id = message_data.get(b'job_id', b'').decode()
                if msg_job_id != job_id:
                    continue
                
                status = message_data.get(b'status', b'').decode()
                if status != 'completed':
                    raise RuntimeError(f'Block processing failed: {message_data.get(b"error", b"unknown")}')
                
                block_i = int(message_data[b'block_i'])
                block_j = int(message_data[b'block_j'])
                result_data = message_data[b'result'].decode()
                
                result_block = deserialize_matrix(result_data)
                result_blocks[(block_i, block_j)] = result_block
                
                processing_time = float(message_data[b'processing_time'])
                DISTRIBUTED_BLOCK_TIME.observe(processing_time)
                DISTRIBUTED_BLOCKS.inc()
                
                completed_blocks += 1
                print(f'Job {job_id}: completed {completed_blocks}/{total_blocks} blocks')
    
    result_rows = []
    for i in range(num_blocks_i):
        row_blocks = []
        for j in range(num_blocks_j):
            row_blocks.append(result_blocks[(i, j)])
        result_rows.append(np.hstack(row_blocks))
    
    result = np.vstack(result_rows)
    
    frobenius_norm = np.linalg.norm(result, 'fro')
    
    job_time = time.time() - start_time
    DISTRIBUTED_JOB_TIME.observe(job_time)
    
    return {
        'result': result.tolist(),
        'frobenius_norm': float(frobenius_norm),
        'used_sparse': False,
        'used_distributed': True,
        'job_id': job_id,
        'num_blocks': total_blocks,
        'processing_time': job_time
    }


def process_batch(batch_data):
    results = []
    for item in batch_data:
        matrix_a = item.get('matrix_a')
        matrix_b = item.get('matrix_b')
        if matrix_a is None or matrix_b is None:
            results.append({'error': 'Missing matrix_a or matrix_b'})
            continue
        try:
            results.append(matrix_multiply_and_norm(matrix_a, matrix_b))
        except Exception as e:
            results.append({'error': str(e)})
    return results


@app.route('/multiply', methods=['POST'])
def multiply():
    start_time = time.time()
    try:
        if request.content_type == 'application/octet-stream':
            data = np.load(io.BytesIO(request.data), allow_pickle=True)
            if isinstance(data, np.ndarray):
                if data.ndim == 3 and data.shape[0] == 2:
                    a, b = data[0], data[1]
                    if max(a.shape[0], a.shape[1], b.shape[0], b.shape[1]) >= DISTRIBUTED_THRESHOLD and redis_client:
                        result = distributed_multiply(a, b)
                    else:
                        result = matrix_multiply_and_norm(a, b, is_numpy=True)
                    REQUEST_COUNT.labels('/multiply', 'success').inc()
                    return jsonify(result)
                elif data.ndim == 4 and data.shape[1] == 2:
                    results = []
                    for i in range(data.shape[0]):
                        a, b = data[i, 0], data[i, 1]
                        if max(a.shape[0], a.shape[1], b.shape[0], b.shape[1]) >= DISTRIBUTED_THRESHOLD and redis_client:
                            results.append(distributed_multiply(a, b))
                        else:
                            results.append(matrix_multiply_and_norm(a, b, is_numpy=True))
                    REQUEST_COUNT.labels('/multiply', 'success').inc()
                    return jsonify({'results': results})
            REQUEST_COUNT.labels('/multiply', 'error').inc()
            return jsonify({'error': 'Invalid binary format'}), 400
        
        data = request.get_json()
        if not data:
            REQUEST_COUNT.labels('/multiply', 'error').inc()
            return jsonify({'error': 'No data provided'}), 400
        
        if 'batch' in data:
            results = process_batch(data['batch'])
            REQUEST_COUNT.labels('/multiply', 'success').inc()
            return jsonify({'results': results})
        
        matrix_a = data.get('matrix_a')
        matrix_b = data.get('matrix_b')
        if matrix_a is None or matrix_b is None:
            REQUEST_COUNT.labels('/multiply', 'error').inc()
            return jsonify({'error': 'Missing matrix_a or matrix_b'}), 400
        
        a = np.asarray(matrix_a, order='C')
        b = np.asarray(matrix_b, order='C')
        
        if max(a.shape[0], a.shape[1], b.shape[0], b.shape[1]) >= DISTRIBUTED_THRESHOLD and redis_client:
            result = distributed_multiply(a, b)
        else:
            result = matrix_multiply_and_norm(matrix_a, matrix_b)
        
        REQUEST_COUNT.labels('/multiply', 'success').inc()
        return jsonify(result)
    
    except Exception as e:
        REQUEST_COUNT.labels('/multiply', 'error').inc()
        return jsonify({'error': str(e)}), 500
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels('/multiply').observe(latency)


@app.route('/multiply/distributed', methods=['POST'])
def multiply_distributed():
    start_time = time.time()
    try:
        if redis_client is None:
            return jsonify({'error': 'Redis not connected. Start Redis server first.'}), 503
        
        if request.content_type == 'application/octet-stream':
            data = np.load(io.BytesIO(request.data), allow_pickle=True)
            if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] == 2:
                result = distributed_multiply(data[0], data[1])
                REQUEST_COUNT.labels('/multiply/distributed', 'success').inc()
                return jsonify(result)
            return jsonify({'error': 'Invalid binary format'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        matrix_a = data.get('matrix_a')
        matrix_b = data.get('matrix_b')
        if matrix_a is None or matrix_b is None:
            return jsonify({'error': 'Missing matrix_a or matrix_b'}), 400
        
        a = np.asarray(matrix_a, order='C')
        b = np.asarray(matrix_b, order='C')
        
        result = distributed_multiply(a, b)
        REQUEST_COUNT.labels('/multiply/distributed', 'success').inc()
        return jsonify(result)
    
    except Exception as e:
        REQUEST_COUNT.labels('/multiply/distributed', 'error').inc()
        return jsonify({'error': str(e)}), 500
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels('/multiply/distributed').observe(latency)


@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/health')
def health():
    redis_status = 'connected' if redis_client else 'disconnected'
    return jsonify({
        'status': 'healthy',
        'redis': redis_status
    })


@app.route('/distributed/status')
def distributed_status():
    if redis_client is None:
        return jsonify({
            'enabled': False,
            'redis_connected': False,
            'message': 'Redis not connected. Start Redis server first.'
        })
    
    try:
        pending = redis_client.xlen(TASK_STREAM)
        results = redis_client.xlen(RESULT_STREAM)
    except:
        pending = 0
        results = 0
    
    return jsonify({
        'enabled': True,
        'redis_connected': True,
        'threshold': DISTRIBUTED_THRESHOLD,
        'block_size': BLOCK_SIZE,
        'pending_tasks': pending,
        'total_results': results
    })


if __name__ == '__main__':
    init_redis()
    app.run(host='0.0.0.0', port=5000, debug=False)
