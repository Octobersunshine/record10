from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes, padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import UnsupportedAlgorithm
import os
import base64
import json


def generate_rsa_keypair(key_size: int = 2048):
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return {
        "private_key_pem": private_pem.decode('utf-8'),
        "public_key_pem": public_pem.decode('utf-8'),
        "key_size": key_size
    }


def get_key_size_from_public_key(public_key_pem: str):
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode('utf-8'),
        backend=default_backend()
    )
    return public_key.key_size


def get_key_size_from_private_key(private_key_pem: str):
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )
    return private_key.key_size


def get_max_plaintext_length(key_size: int, padding_type: str = 'pkcs1'):
    key_bytes = key_size // 8
    if padding_type == 'pkcs1':
        return key_bytes - 11
    elif padding_type == 'oaep_sha256':
        return key_bytes - 2 * 32 - 2
    else:
        raise ValueError(f"不支持的填充类型: {padding_type}")


def public_key_encrypt(public_key_pem: str, plaintext: str):
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode('utf-8'),
        backend=default_backend()
    )

    plaintext_bytes = plaintext.encode('utf-8')
    max_len = get_max_plaintext_length(public_key.key_size, 'oaep_sha256')

    if len(plaintext_bytes) > max_len:
        raise ValueError(
            f"明文过长！当前明文长度: {len(plaintext_bytes)} 字节, "
            f"OAEP+SHA256最大支持: {max_len} 字节。"
            f"请使用分段加密函数 public_key_encrypt_pkcs1_segmented()"
        )

    ciphertext = public_key.encrypt(
        plaintext_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return base64.b64encode(ciphertext).decode('utf-8')


def private_key_decrypt(private_key_pem: str, ciphertext_b64: str):
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )

    ciphertext = base64.b64decode(ciphertext_b64)

    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return plaintext.decode('utf-8')


def public_key_encrypt_pkcs1_segmented(public_key_pem: str, plaintext: str):
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode('utf-8'),
        backend=default_backend()
    )

    plaintext_bytes = plaintext.encode('utf-8')
    key_size = public_key.key_size
    max_chunk_len = get_max_plaintext_length(key_size, 'pkcs1')

    if max_chunk_len <= 0:
        raise ValueError(f"密钥长度 {key_size} 位太小，不支持PKCS1加密")

    chunks = []
    for i in range(0, len(plaintext_bytes), max_chunk_len):
        chunk = plaintext_bytes[i:i + max_chunk_len]
        encrypted_chunk = public_key.encrypt(
            chunk,
            padding.PKCS1v15()
        )
        chunks.append(encrypted_chunk)

    combined = b''.join(chunks)
    return {
        "ciphertext": base64.b64encode(combined).decode('utf-8'),
        "chunk_count": len(chunks),
        "key_size": key_size,
        "max_chunk_length": max_chunk_len,
        "original_length": len(plaintext_bytes),
        "padding": "PKCS1v15"
    }


def private_key_decrypt_pkcs1_segmented(private_key_pem: str, ciphertext_info: dict):
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )

    if isinstance(ciphertext_info, str):
        ciphertext_b64 = ciphertext_info
        chunk_count = None
        key_size = private_key.key_size
    else:
        ciphertext_b64 = ciphertext_info.get("ciphertext")
        chunk_count = ciphertext_info.get("chunk_count")
        key_size = ciphertext_info.get("key_size", private_key.key_size)

    ciphertext_bytes = base64.b64decode(ciphertext_b64)
    chunk_size = key_size // 8

    if len(ciphertext_bytes) % chunk_size != 0:
        raise ValueError(
            f"密文长度 {len(ciphertext_bytes)} 字节不是密钥块大小 {chunk_size} 字节的整数倍"
        )

    total_chunks = len(ciphertext_bytes) // chunk_size
    if chunk_count is not None and total_chunks != chunk_count:
        raise ValueError(
            f"密文段数不匹配: 期望 {chunk_count} 段, 实际 {total_chunks} 段"
        )

    plaintext_chunks = []
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = ciphertext_bytes[start:end]
        decrypted_chunk = private_key.decrypt(
            chunk,
            padding.PKCS1v15()
        )
        plaintext_chunks.append(decrypted_chunk)

    plaintext_bytes = b''.join(plaintext_chunks)
    return plaintext_bytes.decode('utf-8')


def private_key_sign(private_key_pem: str, message: str):
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )

    signature = private_key.sign(
        message.encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    return base64.b64encode(signature).decode('utf-8')


def public_key_verify(public_key_pem: str, message: str, signature_b64: str):
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode('utf-8'),
        backend=default_backend()
    )

    signature = base64.b64decode(signature_b64)

    try:
        public_key.verify(
            signature,
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False


def save_private_key(private_key_pem: str, file_path: str, password: str = None):
    if password:
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
        encrypted_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(password.encode('utf-8'))
        )
        with open(file_path, 'wb') as f:
            f.write(encrypted_pem)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(private_key_pem)
    return file_path


def save_public_key(public_key_pem: str, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(public_key_pem)
    return file_path


def load_private_key(file_path: str, password: str = None):
    with open(file_path, 'rb') as f:
        pem_data = f.read()
    private_key = serialization.load_pem_private_key(
        pem_data,
        password=password.encode('utf-8') if password else None,
        backend=default_backend()
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return private_pem.decode('utf-8')


def load_public_key(file_path: str):
    with open(file_path, 'rb') as f:
        pem_data = f.read()
    public_key = serialization.load_pem_public_key(
        pem_data,
        backend=default_backend()
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return public_pem.decode('utf-8')


def generate_aes_key(key_size: int = 256):
    return os.urandom(key_size // 8)


def aes_encrypt(data: bytes, aes_key: bytes):
    iv = os.urandom(16)
    cipher = Cipher(
        algorithms.AES(aes_key),
        modes.CBC(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    padder = sym_padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext


def aes_decrypt(encrypted_data: bytes, aes_key: bytes):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = Cipher(
        algorithms.AES(aes_key),
        modes.CBC(iv),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = sym_padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data


def hybrid_encrypt(public_key_pem: str, plaintext: str):
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode('utf-8'),
        backend=default_backend()
    )

    aes_key = generate_aes_key(256)
    plaintext_bytes = plaintext.encode('utf-8')
    encrypted_data = aes_encrypt(plaintext_bytes, aes_key)
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    return {
        "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
        "encrypted_aes_key": base64.b64encode(encrypted_aes_key).decode('utf-8'),
        "aes_key_size": 256,
        "padding": "OAEP+SHA256 (RSA), CBC+PKCS7 (AES)",
        "original_length": len(plaintext_bytes)
    }


def hybrid_decrypt(private_key_pem: str, encrypted_info: dict):
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )

    if isinstance(encrypted_info, str):
        raise ValueError("需要传入包含 encrypted_data 和 encrypted_aes_key 的字典")

    encrypted_data = base64.b64decode(encrypted_info["encrypted_data"])
    encrypted_aes_key = base64.b64decode(encrypted_info["encrypted_aes_key"])

    aes_key = private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    plaintext_bytes = aes_decrypt(encrypted_data, aes_key)
    return plaintext_bytes.decode('utf-8')


def hybrid_encrypt_file(public_key_pem: str, input_file: str, output_file: str = None):
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode('utf-8'),
        backend=default_backend()
    )

    if output_file is None:
        output_file = input_file + ".enc"

    aes_key = generate_aes_key(256)
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    with open(input_file, 'rb') as f:
        file_data = f.read()

    encrypted_data = aes_encrypt(file_data, aes_key)

    with open(output_file, 'wb') as f:
        key_len = len(encrypted_aes_key).to_bytes(4, 'big')
        f.write(key_len)
        f.write(encrypted_aes_key)
        f.write(encrypted_data)

    return {
        "output_file": output_file,
        "encrypted_aes_key_b64": base64.b64encode(encrypted_aes_key).decode('utf-8'),
        "aes_key_size": 256,
        "original_file_size": len(file_data),
        "encrypted_file_size": 4 + len(encrypted_aes_key) + len(encrypted_data)
    }


def hybrid_decrypt_file(private_key_pem: str, input_file: str, output_file: str = None):
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )

    if output_file is None:
        if input_file.endswith('.enc'):
            output_file = input_file[:-4]
        else:
            output_file = input_file + ".dec"

    with open(input_file, 'rb') as f:
        key_len_bytes = f.read(4)
        key_len = int.from_bytes(key_len_bytes, 'big')
        encrypted_aes_key = f.read(key_len)
        encrypted_data = f.read()

    aes_key = private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    file_data = aes_decrypt(encrypted_data, aes_key)

    with open(output_file, 'wb') as f:
        f.write(file_data)

    return {
        "output_file": output_file,
        "decrypted_file_size": len(file_data)
    }


if __name__ == "__main__":
    print("=" * 80)
    print("RSA 2048位密钥对生成、加解密、签名验签、混合加密演示")
    print("=" * 80)

    print("\n[1] 生成RSA 2048位密钥对...")
    keypair = generate_rsa_keypair(2048)
    print(f"密钥长度: {keypair['key_size']} 位")

    key_size = keypair["key_size"]
    max_pkcs1 = get_max_plaintext_length(key_size, 'pkcs1')
    max_oaep = get_max_plaintext_length(key_size, 'oaep_sha256')
    print(f"\n[2] 最大明文长度限制:")
    print(f"  - PKCS1v15 填充: {max_pkcs1} 字节")
    print(f"  - OAEP+SHA256 填充: {max_oaep} 字节")

    message = "Hello, RSA 2048!"
    print(f"\n[3] 测试短明文: {message}")
    print(f"  明文长度: {len(message.encode('utf-8'))} 字节")

    print("\n[4] OAEP 公钥加密...")
    encrypted = public_key_encrypt(keypair["public_key_pem"], message)
    print(f"  密文 (Base64): {encrypted[:60]}...")

    print("\n[5] OAEP 私钥解密...")
    decrypted = private_key_decrypt(keypair["private_key_pem"], encrypted)
    print(f"  解密结果: {decrypted}")
    print(f"  解密是否正确: {decrypted == message}")

    print("\n[6] 测试明文过长错误提示...")
    long_message = "A" * (max_oaep + 1)
    try:
        public_key_encrypt(keypair["public_key_pem"], long_message)
        print("  错误: 应该抛出异常但没有！")
    except ValueError as e:
        print(f"  正确捕获异常: {e}")

    print(f"\n[7] PKCS1 分段加密测试...")
    long_plaintext = "这是一段用于测试RSA分段加密的长文本。" * 50
    print(f"  原始明文长度: {len(long_plaintext.encode('utf-8'))} 字节")
    print(f"  单段最大长度: {max_pkcs1} 字节")

    encrypted_result = public_key_encrypt_pkcs1_segmented(keypair["public_key_pem"], long_plaintext)
    print(f"  分段数: {encrypted_result['chunk_count']} 段")

    print("\n[8] PKCS1 分段解密测试...")
    decrypted_long = private_key_decrypt_pkcs1_segmented(keypair["private_key_pem"], encrypted_result)
    print(f"  解密后长度: {len(decrypted_long.encode('utf-8'))} 字节")
    print(f"  解密是否正确: {decrypted_long == long_plaintext}")

    print("\n[9] 私钥签名...")
    signature = private_key_sign(keypair["private_key_pem"], message)
    print(f"  签名 (Base64): {signature[:60]}...")

    print("\n[10] 公钥验签...")
    verify_result = public_key_verify(keypair["public_key_pem"], message, signature)
    print(f"  验签结果: {verify_result}")

    print("\n[11] 保存密钥对到文件...")
    save_private_key(keypair["private_key_pem"], "private_key.pem")
    save_public_key(keypair["public_key_pem"], "public_key.pem")
    print("  已保存: private_key.pem, public_key.pem")

    print("\n[12] 从文件加载密钥对...")
    loaded_private = load_private_key("private_key.pem")
    loaded_public = load_public_key("public_key.pem")
    print(f"  私钥加载成功: {len(loaded_private) > 0}")
    print(f"  公钥加载成功: {len(loaded_public) > 0}")

    print("\n[13] RSA+AES 混合加密（超长文本）...")
    huge_text = "这是一段用于测试RSA+AES混合加密的超长文本。" * 200
    print(f"  原始文本长度: {len(huge_text.encode('utf-8'))} 字节")
    hybrid_result = hybrid_encrypt(keypair["public_key_pem"], huge_text)
    print(f"  加密后AES密钥长度: {len(hybrid_result['encrypted_aes_key'])} 字符")
    print(f"  加密后数据长度: {len(hybrid_result['encrypted_data'])} 字符")

    print("\n[14] RSA+AES 混合解密...")
    hybrid_decrypted = hybrid_decrypt(keypair["private_key_pem"], hybrid_result)
    print(f"  解密后长度: {len(hybrid_decrypted.encode('utf-8'))} 字节")
    print(f"  解密是否正确: {hybrid_decrypted == huge_text}")

    print("\n[15] RSA+AES 文件混合加密...")
    test_file_content = "这是一个测试文件的内容，用于测试RSA+AES混合文件加密功能。" * 100
    with open("test_file.txt", "w", encoding="utf-8") as f:
        f.write(test_file_content)
    print(f"  原始文件: test_file.txt ({len(test_file_content.encode('utf-8'))} 字节)")
    file_enc_result = hybrid_encrypt_file(keypair["public_key_pem"], "test_file.txt")
    print(f"  加密后文件: {file_enc_result['output_file']}")
    print(f"  加密后大小: {file_enc_result['encrypted_file_size']} 字节")

    print("\n[16] RSA+AES 文件混合解密...")
    file_dec_result = hybrid_decrypt_file(keypair["private_key_pem"], "test_file.txt.enc", "test_file_decrypted.txt")
    print(f"  解密后文件: {file_dec_result['output_file']}")
    with open("test_file_decrypted.txt", "r", encoding="utf-8") as f:
        decrypted_content = f.read()
    print(f"  文件内容是否一致: {decrypted_content == test_file_content}")

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)

    result = {
        "keypair": {
            "key_size": keypair["key_size"]
        },
        "max_length_limits": {
            "pkcs1_v15": max_pkcs1,
            "oaep_sha256": max_oaep
        },
        "oaep_encryption_test": {
            "success": decrypted == message
        },
        "pkcs1_segmented_test": {
            "decrypted_success": decrypted_long == long_plaintext
        },
        "signature_test": {
            "verify_success": verify_result
        },
        "key_storage_test": {
            "save_and_load_success": len(loaded_private) > 0 and len(loaded_public) > 0
        },
        "hybrid_encryption_test": {
            "original_length": len(huge_text.encode('utf-8')),
            "decrypted_success": hybrid_decrypted == huge_text
        },
        "file_encryption_test": {
            "file_match": decrypted_content == test_file_content
        }
    }

    print("\n测试结果汇总 (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
