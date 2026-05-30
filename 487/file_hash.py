import hashlib
import hmac
import os
import time
from typing import Optional, List, Dict, Union, Tuple


def get_file_hash(
    file_path: str,
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    计算大文件的哈希值，使用分块读取降低内存占用
    
    参数:
        file_path: 文件路径
        algorithm: 哈希算法名称，如 'md5', 'sha1', 'sha256', 'sha512'
        chunk_size: 每次读取的块大小（字节），默认8KB
    
    返回:
        十六进制格式的哈希字符串
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_file_hash_with_time(
    file_path: str,
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> Tuple[str, float]:
    """
    计算大文件的哈希值并返回耗时
    
    参数:
        file_path: 文件路径
        algorithm: 哈希算法名称
        chunk_size: 每次读取的块大小（字节）
    
    返回:
        (哈希字符串, 耗时秒数)
    """
    start_time = time.time()
    hash_value = get_file_hash(file_path, algorithm, chunk_size)
    elapsed = time.time() - start_time
    return hash_value, elapsed


def get_file_hash_with_progress(
    file_path: str,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
    progress_callback: Optional[callable] = None
) -> str:
    """
    计算大文件的哈希值（带进度回调）
    
    参数:
        file_path: 文件路径
        algorithm: 哈希算法名称
        chunk_size: 每次读取的块大小（字节）
        progress_callback: 进度回调函数，接收 (bytes_read, total_size) 参数
    
    返回:
        十六进制格式的哈希字符串
    """
    hash_obj = hashlib.new(algorithm)
    total_size = os.path.getsize(file_path)
    bytes_read = 0
    
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_obj.update(chunk)
            bytes_read += len(chunk)
            if progress_callback:
                progress_callback(bytes_read, total_size)
    
    return hash_obj.hexdigest()


def get_file_hmac(
    file_path: str,
    key: Union[str, bytes],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    计算大文件的HMAC值（带密钥的哈希），用于API签名验证
    
    参数:
        file_path: 文件路径
        key: 密钥，可以是字符串或字节
        algorithm: 哈希算法名称，如 'md5', 'sha1', 'sha256', 'sha512'
        chunk_size: 每次读取的块大小（字节），默认8KB
    
    返回:
        十六进制格式的HMAC字符串
    """
    if isinstance(key, str):
        key = key.encode("utf-8")
    
    hmac_obj = hmac.new(key, digestmod=algorithm)
    
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hmac_obj.update(chunk)
    
    return hmac_obj.hexdigest()


def get_file_hmac_with_time(
    file_path: str,
    key: Union[str, bytes],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> Tuple[str, float]:
    """
    计算大文件的HMAC值并返回耗时
    
    参数:
        file_path: 文件路径
        key: 密钥，可以是字符串或字节
        algorithm: 哈希算法名称
        chunk_size: 每次读取的块大小（字节）
    
    返回:
        (HMAC字符串, 耗时秒数)
    """
    start_time = time.time()
    hmac_value = get_file_hmac(file_path, key, algorithm, chunk_size)
    elapsed = time.time() - start_time
    return hmac_value, elapsed


def batch_file_hash(
    file_paths: List[str],
    algorithm: str = "sha256",
    chunk_size: int = 8192,
    include_metadata: bool = False
) -> List[Dict]:
    """
    批量计算多个文件的哈希值
    
    参数:
        file_paths: 文件路径列表
        algorithm: 哈希算法名称
        chunk_size: 每次读取的块大小（字节）
        include_metadata: 是否包含文件大小、耗时等元数据
    
    返回:
        结果列表，每个元素是包含文件信息的字典
    """
    results = []
    
    for file_path in file_paths:
        result = {"file_path": file_path}
        
        if not os.path.exists(file_path):
            result["success"] = False
            result["error"] = "文件不存在"
            results.append(result)
            continue
        
        if not os.path.isfile(file_path):
            result["success"] = False
            result["error"] = "不是文件"
            results.append(result)
            continue
        
        try:
            hash_value, elapsed = get_file_hash_with_time(file_path, algorithm, chunk_size)
            result["success"] = True
            result["hash"] = hash_value
            result["algorithm"] = algorithm
            
            if include_metadata:
                result["file_size"] = os.path.getsize(file_path)
                result["elapsed_time"] = elapsed
                result["speed"] = result["file_size"] / elapsed if elapsed > 0 else 0
            
            results.append(result)
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            results.append(result)
    
    return results


def batch_file_hmac(
    file_paths: List[str],
    key: Union[str, bytes],
    algorithm: str = "sha256",
    chunk_size: int = 8192,
    include_metadata: bool = False
) -> List[Dict]:
    """
    批量计算多个文件的HMAC值
    
    参数:
        file_paths: 文件路径列表
        key: 密钥，可以是字符串或字节
        algorithm: 哈希算法名称
        chunk_size: 每次读取的块大小（字节）
        include_metadata: 是否包含文件大小、耗时等元数据
    
    返回:
        结果列表，每个元素是包含文件信息的字典
    """
    results = []
    
    for file_path in file_paths:
        result = {"file_path": file_path}
        
        if not os.path.exists(file_path):
            result["success"] = False
            result["error"] = "文件不存在"
            results.append(result)
            continue
        
        if not os.path.isfile(file_path):
            result["success"] = False
            result["error"] = "不是文件"
            results.append(result)
            continue
        
        try:
            hmac_value, elapsed = get_file_hmac_with_time(file_path, key, algorithm, chunk_size)
            result["success"] = True
            result["hmac"] = hmac_value
            result["algorithm"] = algorithm
            
            if include_metadata:
                result["file_size"] = os.path.getsize(file_path)
                result["elapsed_time"] = elapsed
                result["speed"] = result["file_size"] / elapsed if elapsed > 0 else 0
            
            results.append(result)
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            results.append(result)
    
    return results


def verify_file_hmac(
    file_path: str,
    key: Union[str, bytes],
    expected_hmac: str,
    algorithm: str = "sha256"
) -> bool:
    """
    验证文件的HMAC签名（用于API签名验证）
    
    参数:
        file_path: 文件路径
        key: 密钥
        expected_hmac: 期望的HMAC值
        algorithm: 哈希算法名称
    
    返回:
        验证通过返回True，否则返回False
    """
    actual_hmac = get_file_hmac(file_path, key, algorithm)
    return hmac.compare_digest(actual_hmac, expected_hmac)


if __name__ == "__main__":
    import sys
    
    def print_usage():
        print(f"用法:")
        print(f"  单文件哈希: python {sys.argv[0]} hash <文件路径> [算法]")
        print(f"  单文件HMAC: python {sys.argv[0]} hmac <文件路径> <密钥> [算法]")
        print(f"  批量哈希:   python {sys.argv[0]} batch <文件1> <文件2> ... [算法]")
        print(f"  HMAC验证:   python {sys.argv[0]} verify <文件路径> <密钥> <期望HMAC> [算法]")
        print(f"\n支持的算法: md5, sha1, sha256, sha512")
        print(f"\n示例:")
        print(f"  py {sys.argv[0]} hash myfile.zip sha256")
        print(f"  py {sys.argv[0]} hmac myfile.zip my_secret_key")
        print(f"  py {sys.argv[0]} verify myfile.zip my_secret_key abc123...")
        print(f"  py {sys.argv[0]} batch file1.txt file2.bin file3.iso")
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "hash" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        algorithm = sys.argv[3] if len(sys.argv) > 3 else "sha256"
        
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
        
        file_size = os.path.getsize(file_path)
        print(f"文件: {file_path}")
        print(f"大小: {file_size / (1024 * 1024):.2f} MB")
        print(f"算法: {algorithm}")
        print(f"分块大小: 8 KB")
        print("正在计算...")
        
        file_hash, elapsed = get_file_hash_with_time(file_path, algorithm)
        speed = file_size / elapsed if elapsed > 0 else 0
        
        print(f"哈希值: {file_hash}")
        print(f"耗时: {elapsed:.4f} 秒")
        print(f"速度: {speed / (1024 * 1024):.2f} MB/s")
        print(f"内存峰值: ~8 KB (仅当前读取块)")
    
    elif mode == "hmac" and len(sys.argv) >= 4:
        file_path = sys.argv[2]
        key = sys.argv[3]
        algorithm = sys.argv[4] if len(sys.argv) > 4 else "sha256"
        
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
        
        file_size = os.path.getsize(file_path)
        print(f"文件: {file_path}")
        print(f"大小: {file_size / (1024 * 1024):.2f} MB")
        print(f"算法: HMAC-{algorithm.upper()}")
        print("正在计算...")
        
        hmac_value, elapsed = get_file_hmac_with_time(file_path, key, algorithm)
        speed = file_size / elapsed if elapsed > 0 else 0
        
        print(f"HMAC值: {hmac_value}")
        print(f"耗时: {elapsed:.4f} 秒")
        print(f"速度: {speed / (1024 * 1024):.2f} MB/s")
    
    elif mode == "verify" and len(sys.argv) >= 5:
        file_path = sys.argv[2]
        key = sys.argv[3]
        expected_hmac = sys.argv[4]
        algorithm = sys.argv[5] if len(sys.argv) > 5 else "sha256"
        
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
        
        print(f"正在验证 HMAC-{algorithm.upper()} 签名...")
        is_valid = verify_file_hmac(file_path, key, expected_hmac, algorithm)
        
        if is_valid:
            print("✓ HMAC验证通过！")
        else:
            print("✗ HMAC验证失败！")
            sys.exit(1)
    
    elif mode == "batch" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        algorithm = "sha256"
        
        if args[-1].lower() in ["md5", "sha1", "sha256", "sha512"]:
            algorithm = args[-1].lower()
            file_paths = args[:-1]
        else:
            file_paths = args
        
        print(f"批量哈希计算 ({len(file_paths)} 个文件)")
        print(f"算法: {algorithm}")
        print("-" * 80)
        
        results = batch_file_hash(file_paths, algorithm, include_metadata=True)
        
        total_size = 0
        total_time = 0
        success_count = 0
        
        for result in results:
            if result["success"]:
                success_count += 1
                total_size += result["file_size"]
                total_time += result["elapsed_time"]
                print(f"✓ {result['file_path']}")
                print(f"  哈希: {result['hash']}")
                print(f"  大小: {result['file_size'] / (1024 * 1024):.2f} MB, 耗时: {result['elapsed_time']:.4f}s")
            else:
                print(f"✗ {result['file_path']}: {result['error']}")
        
        print("-" * 80)
        print(f"成功: {success_count}/{len(results)}")
        if success_count > 0:
            print(f"总大小: {total_size / (1024 * 1024):.2f} MB")
            print(f"总耗时: {total_time:.4f} 秒")
            print(f"平均速度: {total_size / total_time / (1024 * 1024):.2f} MB/s")
    
    else:
        print_usage()
        sys.exit(1)
