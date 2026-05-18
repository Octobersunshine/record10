import io
import time
import numpy as np
from scipy.sparse import csr_matrix, issparse
import redis
import json
import uuid


class MatrixWorker:
    def __init__(self, worker_id=None, redis_host='localhost', redis_port=6379):
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.task_stream = 'matrix_tasks'
        self.result_stream = 'matrix_results'
        self.consumer_group = 'matrix_workers'
        
        try:
            self.redis_client.xgroup_create(self.task_stream, self.consumer_group, mkstream=True)
        except redis.exceptions.ResponseError:
            pass
        
        print(f'Worker {self.worker_id} started. Listening for tasks...')
    
    def calculate_density(self, arr):
        non_zero = np.count_nonzero(arr)
        total = arr.size
        return non_zero / total if total > 0 else 0
    
    def process_block(self, block_a, block_b):
        density_a = self.calculate_density(block_a)
        density_b = self.calculate_density(block_b)
        avg_density = (density_a + density_b) / 2
        
        if avg_density < 0.1:
            if not issparse(block_a):
                block_a = csr_matrix(block_a)
            if not issparse(block_b):
                block_b = csr_matrix(block_b)
            result = block_a.dot(block_b)
            if issparse(result):
                result = result.toarray()
        else:
            result = np.dot(block_a, block_b)
        
        return result
    
    def deserialize_matrix(self, data):
        if isinstance(data, str):
            data = json.loads(data)
        
        if data.get('type') == 'numpy':
            buffer = io.BytesIO(bytes.fromhex(data['data']))
            return np.load(buffer)
        elif data.get('type') == 'list':
            return np.array(data['data'])
        else:
            raise ValueError(f'Unknown matrix type: {data.get("type")}')
    
    def serialize_matrix(self, arr):
        buffer = io.BytesIO()
        np.save(buffer, arr)
        buffer.seek(0)
        return {
            'type': 'numpy',
            'data': buffer.read().hex()
        }
    
    def process_task(self, task_id, task_data):
        start_time = time.time()
        
        try:
            block_a = self.deserialize_matrix(task_data[b'block_a'])
            block_b = self.deserialize_matrix(task_data[b'block_b'])
            block_i = int(task_data[b'block_i'])
            block_j = int(task_data[b'block_j'])
            job_id = task_data[b'job_id'].decode()
            
            print(f'Worker {self.worker_id} processing block ({block_i}, {block_j}) for job {job_id}...')
            
            result_block = self.process_block(block_a, block_b)
            
            processing_time = time.time() - start_time
            
            result_msg = {
                'job_id': job_id,
                'block_i': str(block_i),
                'block_j': str(block_j),
                'result': json.dumps(self.serialize_matrix(result_block)),
                'worker_id': self.worker_id,
                'processing_time': str(processing_time),
                'status': 'completed'
            }
            
            self.redis_client.xadd(self.result_stream, result_msg)
            print(f'Worker {self.worker_id} completed block ({block_i}, {block_j}) in {processing_time:.4f}s')
            
        except Exception as e:
            print(f'Worker {self.worker_id} error processing task: {e}')
            try:
                job_id = task_data.get(b'job_id', b'unknown').decode()
                block_i = int(task_data.get(b'block_i', 0))
                block_j = int(task_data.get(b'block_j', 0))
                
                result_msg = {
                    'job_id': job_id,
                    'block_i': str(block_i),
                    'block_j': str(block_j),
                    'worker_id': self.worker_id,
                    'status': 'error',
                    'error': str(e)
                }
                self.redis_client.xadd(self.result_stream, result_msg)
            except:
                pass
    
    def run(self):
        while True:
            try:
                messages = self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.worker_id,
                    {self.task_stream: '>'},
                    count=1,
                    block=1000
                )
                
                for stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        self.process_task(message_id, message_data)
                        self.redis_client.xack(self.task_stream, self.consumer_group, message_id)
                        
            except KeyboardInterrupt:
                print(f'Worker {self.worker_id} stopping...')
                break
            except Exception as e:
                print(f'Worker {self.worker_id} error: {e}')
                time.sleep(1)


if __name__ == '__main__':
    import sys
    
    worker_id = sys.argv[1] if len(sys.argv) > 1 else None
    worker = MatrixWorker(worker_id=worker_id)
    worker.run()
