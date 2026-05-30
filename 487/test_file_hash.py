import os
import tempfile
import hashlib
import hmac
from file_hash import (
    get_file_hash,
    get_file_hash_with_time,
    get_file_hash_with_progress,
    get_file_hmac,
    get_file_hmac_with_time,
    verify_file_hmac,
    batch_file_hash,
    batch_file_hmac
)


def test_correctness():
    """验证分块读取的哈希值与一次性读取一致"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(os.urandom(1024 * 1024))
        tmp_path = tmp.name
    
    try:
        with open(tmp_path, "rb") as f:
            data = f.read()
        expected_hash = hashlib.sha256(data).hexdigest()
        
        chunk_hash = get_file_hash(tmp_path, "sha256", 8192)
        
        assert chunk_hash == expected_hash, f"哈希不匹配: {chunk_hash} != {expected_hash}"
        print("✓ 正确性测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_different_chunk_sizes():
    """测试不同块大小得到相同哈希值"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(os.urandom(5 * 1024 * 1024))
        tmp_path = tmp.name
    
    try:
        hashes = []
        for chunk_size in [1024, 4096, 8192, 65536, 1048576]:
            h = get_file_hash(tmp_path, "sha256", chunk_size)
            hashes.append(h)
            print(f"  块大小 {chunk_size:>8} 字节: {h[:16]}...")
        
        assert len(set(hashes)) == 1, "不同块大小得到不同哈希值"
        print("✓ 不同块大小测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_different_algorithms():
    """测试不同哈希算法"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"Hello, World! This is a test file for hashing.")
        tmp_path = tmp.name
    
    try:
        for algo in ["md5", "sha1", "sha256", "sha512"]:
            chunk_hash = get_file_hash(tmp_path, algo)
            
            with open(tmp_path, "rb") as f:
                expected = hashlib.new(algo, f.read()).hexdigest()
            
            assert chunk_hash == expected, f"{algo} 哈希不匹配"
            print(f"  {algo:>6}: {chunk_hash[:16]}... ✓")
        
        print("✓ 多算法测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_progress_callback():
    """测试进度回调"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(os.urandom(100 * 1024))
        tmp_path = tmp.name
    
    try:
        progress_calls = []
        
        def callback(read, total):
            progress_calls.append((read, total))
        
        get_file_hash_with_progress(tmp_path, "sha256", 8192, callback)
        
        assert len(progress_calls) > 0, "进度回调未被调用"
        assert progress_calls[-1][0] == progress_calls[-1][1], "最终进度未完成"
        assert progress_calls[-1][0] == os.path.getsize(tmp_path), "读取字节数不匹配"
        
        print(f"✓ 进度回调测试通过 (调用 {len(progress_calls)} 次)")
        return True
    finally:
        os.unlink(tmp_path)


def test_empty_file():
    """测试空文件"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        h = get_file_hash(tmp_path, "sha256")
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected, f"空文件哈希不匹配: {h} != {expected}"
        print("✓ 空文件测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_small_file_less_than_chunk():
    """测试小于块大小的小文件"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"Small file")
        tmp_path = tmp.name
    
    try:
        h = get_file_hash(tmp_path, "sha256", 8192)
        expected = hashlib.sha256(b"Small file").hexdigest()
        assert h == expected, f"小文件哈希不匹配"
        print("✓ 小文件测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_hash_with_time():
    """测试哈希计算带耗时统计"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(os.urandom(2 * 1024 * 1024))
        tmp_path = tmp.name
    
    try:
        hash_value, elapsed = get_file_hash_with_time(tmp_path, "sha256")
        
        with open(tmp_path, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        
        assert hash_value == expected, "哈希值不匹配"
        assert elapsed > 0, "耗时应该大于0"
        print(f"✓ 耗时统计测试通过 (耗时: {elapsed:.4f}s)")
        return True
    finally:
        os.unlink(tmp_path)


def test_hmac_correctness():
    """测试HMAC计算正确性"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"Test data for HMAC calculation")
        tmp_path = tmp.name
    
    try:
        key = "my_secret_key"
        
        with open(tmp_path, "rb") as f:
            data = f.read()
        expected_hmac = hmac.new(key.encode(), data, "sha256").hexdigest()
        
        chunk_hmac = get_file_hmac(tmp_path, key, "sha256")
        
        assert chunk_hmac == expected_hmac, f"HMAC不匹配: {chunk_hmac} != {expected_hmac}"
        print("✓ HMAC正确性测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_hmac_with_time():
    """测试HMAC带耗时统计"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(os.urandom(1024 * 1024))
        tmp_path = tmp.name
    
    try:
        key = b"test_key_123"
        hmac_value, elapsed = get_file_hmac_with_time(tmp_path, key, "sha256")
        
        with open(tmp_path, "rb") as f:
            expected = hmac.new(key, f.read(), "sha256").hexdigest()
        
        assert hmac_value == expected, "HMAC值不匹配"
        assert elapsed > 0, "耗时应该大于0"
        print(f"✓ HMAC耗时统计通过 (耗时: {elapsed:.4f}s)")
        return True
    finally:
        os.unlink(tmp_path)


def test_hmac_different_keys():
    """测试不同密钥产生不同HMAC"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"Test data")
        tmp_path = tmp.name
    
    try:
        hmac1 = get_file_hmac(tmp_path, "key1", "sha256")
        hmac2 = get_file_hmac(tmp_path, "key2", "sha256")
        
        assert hmac1 != hmac2, "不同密钥应该产生不同HMAC"
        print("✓ 不同密钥HMAC测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_hmac_verify():
    """测试HMAC验证功能"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"API response data")
        tmp_path = tmp.name
    
    try:
        key = "api_secret_key"
        correct_hmac = get_file_hmac(tmp_path, key, "sha256")
        
        assert verify_file_hmac(tmp_path, key, correct_hmac, "sha256") == True, "验证应该通过"
        assert verify_file_hmac(tmp_path, key, "wrong_hmac", "sha256") == False, "验证应该失败"
        assert verify_file_hmac(tmp_path, "wrong_key", correct_hmac, "sha256") == False, "验证应该失败"
        
        print("✓ HMAC验证功能测试通过")
        return True
    finally:
        os.unlink(tmp_path)


def test_batch_file_hash():
    """测试批量文件哈希计算"""
    tmp_files = []
    try:
        for i in range(3):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(f"File content {i}".encode())
            tmp_files.append(tmp.name)
            tmp.close()
        
        results = batch_file_hash(tmp_files, "sha256", include_metadata=True)
        
        assert len(results) == 3, "结果数量不匹配"
        for i, result in enumerate(results):
            assert result["success"] == True, f"文件{i}计算失败"
            assert result["file_path"] == tmp_files[i], f"文件{i}路径不匹配"
            assert "hash" in result, f"文件{i}缺少hash字段"
            assert "elapsed_time" in result, f"文件{i}缺少耗时字段"
            assert result["elapsed_time"] >= 0, f"文件{i}耗时应该>=0"
            print(f"  文件{i}: {result['hash'][:16]}... (耗时: {result['elapsed_time']:.4f}s)")
        
        print("✓ 批量哈希计算测试通过")
        return True
    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_batch_file_hash_with_errors():
    """测试批量哈希计算包含错误的情况"""
    tmp_files = []
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"valid file")
        tmp_files.append(tmp.name)
        tmp.close()
        
        tmp_files.append("nonexistent_file_12345.bin")
        
        results = batch_file_hash(tmp_files, "sha256")
        
        assert len(results) == 2, "结果数量不匹配"
        assert results[0]["success"] == True, "有效文件应该成功"
        assert results[1]["success"] == False, "不存在文件应该失败"
        assert "error" in results[1], "错误结果应该包含error字段"
        
        print(f"  有效文件: {results[0]['hash'][:16]}...")
        print(f"  无效文件: {results[1]['error']}")
        print("✓ 批量哈希错误处理测试通过")
        return True
    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_batch_file_hmac():
    """测试批量文件HMAC计算"""
    tmp_files = []
    try:
        for i in range(3):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(f"Batch HMAC test {i}".encode())
            tmp_files.append(tmp.name)
            tmp.close()
        
        key = "batch_key"
        results = batch_file_hmac(tmp_files, key, "sha256", include_metadata=True)
        
        assert len(results) == 3, "结果数量不匹配"
        for i, result in enumerate(results):
            assert result["success"] == True, f"文件{i}计算失败"
            assert "hmac" in result, f"文件{i}缺少hmac字段"
            
            expected = get_file_hmac(tmp_files[i], key, "sha256")
            assert result["hmac"] == expected, f"文件{i}HMAC不匹配"
        
        print("✓ 批量HMAC计算测试通过")
        return True
    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_hmac_algorithms():
    """测试HMAC支持多种算法"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"Multi-algo test")
        tmp_path = tmp.name
    
    try:
        key = "test_key"
        for algo in ["md5", "sha1", "sha256", "sha512"]:
            hmac_value = get_file_hmac(tmp_path, key, algo)
            
            with open(tmp_path, "rb") as f:
                expected = hmac.new(key.encode(), f.read(), algo).hexdigest()
            
            assert hmac_value == expected, f"{algo} HMAC不匹配"
            print(f"  {algo:>6}: {hmac_value[:16]}... ✓")
        
        print("✓ HMAC多算法测试通过")
        return True
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("大文件分块哈希计算 - 完整测试套件")
    print("=" * 60)
    
    tests = [
        ("正确性测试", test_correctness),
        ("不同块大小测试", test_different_chunk_sizes),
        ("不同算法测试", test_different_algorithms),
        ("进度回调测试", test_progress_callback),
        ("空文件测试", test_empty_file),
        ("小文件测试", test_small_file_less_than_chunk),
        ("耗时统计测试", test_hash_with_time),
        ("HMAC正确性测试", test_hmac_correctness),
        ("HMAC耗时统计", test_hmac_with_time),
        ("HMAC不同密钥测试", test_hmac_different_keys),
        ("HMAC验证功能测试", test_hmac_verify),
        ("HMAC多算法测试", test_hmac_algorithms),
        ("批量哈希计算测试", test_batch_file_hash),
        ("批量哈希错误处理测试", test_batch_file_hash_with_errors),
        ("批量HMAC计算测试", test_batch_file_hmac),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed > 0:
        exit(1)
