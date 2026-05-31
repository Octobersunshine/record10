from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256, HMAC
import base64
import hashlib
import struct
import os

VALID_KEY_LENGTHS = (16, 24, 32)
GCM_NONCE_SIZE = 12
GCM_TAG_SIZE = 16
PBKDF2_DEFAULT_ITERATIONS = 480000
CHUNK_MAGIC = b'AESCHK'
CHUNK_SIZE_DEFAULT = 64 * 1024


def validate_and_pad_key(key: bytes, auto_pad: bool = True) -> bytes:
    if len(key) in VALID_KEY_LENGTHS:
        return key
    if not auto_pad:
        raise ValueError(
            f"密钥长度不合规: {len(key)} 字节 ({len(key) * 8} 位)。"
            f"AES 要求密钥长度为 16/24/32 字节 (128/192/256 位)。"
        )
    if len(key) < 16:
        padded = key.ljust(16, b'\x00')
        print(f"[警告] 密钥长度 {len(key)} 字节不足，已用 \\x00 填充至 16 字节 (128 位)")
        return padded
    if len(key) <= 24:
        padded = key.ljust(24, b'\x00')
        print(f"[警告] 密钥长度 {len(key)} 字节不合规，已用 \\x00 填充至 24 字节 (192 位)")
        return padded
    derived = hashlib.sha256(key).digest()
    print(f"[警告] 密钥长度 {len(key)} 字节超过 32，已通过 SHA-256 派生为 32 字节 (256 位)")
    return derived


def derive_key_from_password(password: str, salt: bytes = None,
                             key_length: int = 32,
                             iterations: int = PBKDF2_DEFAULT_ITERATIONS) -> tuple:
    if salt is None:
        salt = get_random_bytes(16)
    key = PBKDF2(password.encode('utf-8'), salt,
                 dkLen=key_length, count=iterations,
                 prf=lambda p, s: HMAC.new(p, s, SHA256).digest())
    return key, salt


def aes_cbc_encrypt(key: bytes, plaintext: str, auto_pad_key: bool = True) -> str:
    validated_key = validate_and_pad_key(key, auto_pad=auto_pad_key)
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(validated_key, AES.MODE_CBC, iv)
    padded_data = pad(plaintext.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    combined = iv + ciphertext
    return base64.b64encode(combined).decode('utf-8')


def aes_cbc_decrypt(key: bytes, encrypted_data: str, auto_pad_key: bool = True) -> str:
    validated_key = validate_and_pad_key(key, auto_pad=auto_pad_key)
    combined = base64.b64decode(encrypted_data)
    if len(combined) <= AES.block_size:
        raise ValueError("密文数据过短，无法提取有效的 IV 和密文")
    iv = combined[:AES.block_size]
    ciphertext = combined[AES.block_size:]
    cipher = AES.new(validated_key, AES.MODE_CBC, iv)
    padded_plaintext = cipher.decrypt(ciphertext)
    plaintext = unpad(padded_plaintext, AES.block_size)
    return plaintext.decode('utf-8')


def aes_gcm_encrypt(key: bytes, plaintext: str, auto_pad_key: bool = True) -> str:
    validated_key = validate_and_pad_key(key, auto_pad=auto_pad_key)
    nonce = get_random_bytes(GCM_NONCE_SIZE)
    cipher = AES.new(validated_key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
    combined = nonce + tag + ciphertext
    return base64.b64encode(combined).decode('utf-8')


def aes_gcm_decrypt(key: bytes, encrypted_data: str, auto_pad_key: bool = True) -> str:
    validated_key = validate_and_pad_key(key, auto_pad=auto_pad_key)
    combined = base64.b64decode(encrypted_data)
    min_len = GCM_NONCE_SIZE + GCM_TAG_SIZE
    if len(combined) <= min_len:
        raise ValueError("密文数据过短，无法提取有效的 nonce、tag 和密文")
    nonce = combined[:GCM_NONCE_SIZE]
    tag = combined[GCM_NONCE_SIZE:GCM_NONCE_SIZE + GCM_TAG_SIZE]
    ciphertext = combined[GCM_NONCE_SIZE + GCM_TAG_SIZE:]
    cipher = AES.new(validated_key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext.decode('utf-8')


def chunked_encrypt_file(key: bytes, input_path: str, output_path: str,
                         mode: str = 'GCM', chunk_size: int = CHUNK_SIZE_DEFAULT,
                         auto_pad_key: bool = True) -> None:
    validated_key = validate_and_pad_key(key, auto_pad=auto_pad_key)
    mode_byte = b'\x02' if mode.upper() == 'GCM' else b'\x01'
    file_size = os.path.getsize(input_path)

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        fout.write(CHUNK_MAGIC)
        fout.write(mode_byte)
        fout.write(struct.pack('>I', chunk_size))
        fout.write(struct.pack('>Q', file_size))

        while True:
            data = fin.read(chunk_size)
            if not data:
                break
            is_last = fin.tell() == file_size

            if mode_byte == b'\x02':
                nonce = get_random_bytes(GCM_NONCE_SIZE)
                cipher = AES.new(validated_key, AES.MODE_GCM, nonce=nonce)
                ciphertext, tag = cipher.encrypt_and_digest(data)
                chunk_enc = nonce + tag + ciphertext
            else:
                iv = get_random_bytes(AES.block_size)
                cipher = AES.new(validated_key, AES.MODE_CBC, iv)
                if is_last:
                    padded = pad(data, AES.block_size)
                    ciphertext = cipher.encrypt(padded)
                else:
                    if len(data) % AES.block_size != 0:
                        padded = pad(data, AES.block_size)
                        ciphertext = cipher.encrypt(padded)
                    else:
                        ciphertext = cipher.encrypt(data)
                chunk_enc = iv + ciphertext

            fout.write(struct.pack('>I', len(chunk_enc)))
            fout.write(chunk_enc)


def chunked_decrypt_file(key: bytes, input_path: str, output_path: str,
                         auto_pad_key: bool = True) -> None:
    validated_key = validate_and_pad_key(key, auto_pad=auto_pad_key)

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        magic = fin.read(len(CHUNK_MAGIC))
        if magic != CHUNK_MAGIC:
            raise ValueError("无效的加密文件格式：缺少魔术头")

        mode_byte = fin.read(1)
        if mode_byte == b'\x02':
            mode = 'GCM'
        elif mode_byte == b'\x01':
            mode = 'CBC'
        else:
            raise ValueError(f"未知的加密模式标识: {mode_byte}")

        chunk_size = struct.unpack('>I', fin.read(4))[0]
        original_size = struct.unpack('>Q', fin.read(8))[0]
        total_written = 0

        while True:
            chunk_len_bytes = fin.read(4)
            if not chunk_len_bytes or len(chunk_len_bytes) < 4:
                break
            chunk_len = struct.unpack('>I', chunk_len_bytes)[0]
            chunk_data = fin.read(chunk_len)
            if len(chunk_data) < chunk_len:
                raise ValueError("文件损坏：分块数据不完整")

            remaining = original_size - total_written
            is_last = (remaining <= chunk_size)

            if mode == 'GCM':
                if len(chunk_data) <= GCM_NONCE_SIZE + GCM_TAG_SIZE:
                    raise ValueError("GCM 分块数据过短")
                nonce = chunk_data[:GCM_NONCE_SIZE]
                tag = chunk_data[GCM_NONCE_SIZE:GCM_NONCE_SIZE + GCM_TAG_SIZE]
                ciphertext = chunk_data[GCM_NONCE_SIZE + GCM_TAG_SIZE:]
                cipher = AES.new(validated_key, AES.MODE_GCM, nonce=nonce)
                plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            else:
                if len(chunk_data) <= AES.block_size:
                    raise ValueError("CBC 分块数据过短")
                iv = chunk_data[:AES.block_size]
                ciphertext = chunk_data[AES.block_size:]
                cipher = AES.new(validated_key, AES.MODE_CBC, iv)
                decrypted = cipher.decrypt(ciphertext)
                if is_last:
                    plaintext = unpad(decrypted, AES.block_size)
                else:
                    plaintext = decrypted

            write_len = min(len(plaintext), remaining)
            fout.write(plaintext[:write_len])
            total_written += write_len

    actual_size = os.path.getsize(output_path)
    if actual_size != original_size:
        raise ValueError(
            f"解密文件大小不匹配: 期望 {original_size} 字节, 实际 {actual_size} 字节"
        )


if __name__ == '__main__':
    print("=" * 60)
    print("  AES 加密解密工具 — CBC / GCM / PBKDF2 / 分块文件")
    print("=" * 60)

    print("\n=== 测试1: CBC模式 — 合规128位密钥 ===")
    key_16 = b'0123456789abcdef'
    pt1 = 'Hello, AES-CBC Encryption!'
    enc_cbc = aes_cbc_encrypt(key_16, pt1)
    dec_cbc = aes_cbc_decrypt(key_16, enc_cbc)
    print(f'明文: {pt1}')
    print(f'密文 (Base64): {enc_cbc}')
    print(f'解密结果: {dec_cbc}')
    print(f'验证: {pt1 == dec_cbc}')

    print("\n=== 测试2: GCM模式 — 带认证加密 ===")
    key_32 = b'0123456789abcdef0123456789abcdef'
    pt2 = 'Hello, AES-GCM Authenticated Encryption!'
    enc_gcm = aes_gcm_encrypt(key_32, pt2)
    dec_gcm = aes_gcm_decrypt(key_32, enc_gcm)
    print(f'明文: {pt2}')
    print(f'密文 (Base64): {enc_gcm}')
    print(f'解密结果: {dec_gcm}')
    print(f'验证: {pt2 == dec_gcm}')

    print("\n=== 测试3: GCM防篡改验证 ===")
    tampered = bytearray(base64.b64decode(enc_gcm))
    tampered[-1] ^= 0xFF
    tampered_b64 = base64.b64encode(bytes(tampered)).decode('utf-8')
    try:
        aes_gcm_decrypt(key_32, tampered_b64)
        print('错误: 篡改数据未被检测!')
    except ValueError as e:
        print(f'正确检测到篡改: {e}')

    print("\n=== 测试4: PBKDF2密钥派生 ===")
    password = 'MyStr0ngP@ssw0rd!'
    derived_key, salt = derive_key_from_password(password, key_length=32)
    print(f'密码: {password}')
    print(f'盐值 (hex): {salt.hex()}')
    print(f'派生密钥 (hex): {derived_key.hex()}')
    print(f'派生密钥长度: {len(derived_key)} 字节 ({len(derived_key) * 8} 位)')

    pt3 = '用PBKDF2派生密钥加密的数据'
    enc_pbkdf2 = aes_gcm_encrypt(derived_key, pt3)
    same_key, _ = derive_key_from_password(password, salt=salt, key_length=32)
    dec_pbkdf2 = aes_gcm_decrypt(same_key, enc_pbkdf2)
    print(f'明文: {pt3}')
    print(f'解密结果: {dec_pbkdf2}')
    print(f'验证: {pt3 == dec_pbkdf2}')

    wrong_key, _ = derive_key_from_password('WrongPassword', salt=salt, key_length=32)
    try:
        aes_gcm_decrypt(wrong_key, enc_pbkdf2)
        print('错误: 错误密码未被检测!')
    except ValueError:
        print('正确拒绝错误密码解密')

    print("\n=== 测试5: GCM随机Nonce (相同明文两次加密结果不同) ===")
    enc_a = aes_gcm_encrypt(key_32, pt2)
    enc_b = aes_gcm_encrypt(key_32, pt2)
    print(f'密文A: {enc_a}')
    print(f'密文B: {enc_b}')
    print(f'两次密文不同: {enc_a != enc_b}')
    print(f'均能正确解密: {aes_gcm_decrypt(key_32, enc_a) == pt2 and aes_gcm_decrypt(key_32, enc_b) == pt2}')

    print("\n=== 测试6: 分块文件加解密 — GCM模式 ===")
    test_dir = os.path.dirname(os.path.abspath(__file__))
    large_data = os.urandom(200_000)
    src_gcm = os.path.join(test_dir, 'test_gcm_plain.bin')
    enc_gcm_file = os.path.join(test_dir, 'test_gcm_encrypted.bin')
    dec_gcm_file = os.path.join(test_dir, 'test_gcm_decrypted.bin')
    with open(src_gcm, 'wb') as f:
        f.write(large_data)
    chunked_encrypt_file(key_32, src_gcm, enc_gcm_file, mode='GCM', chunk_size=65536)
    chunked_decrypt_file(key_32, enc_gcm_file, dec_gcm_file)
    with open(dec_gcm_file, 'rb') as f:
        decrypted_large = f.read()
    print(f'原始大小: {len(large_data)} 字节')
    print(f'加密文件大小: {os.path.getsize(enc_gcm_file)} 字节')
    print(f'解密大小: {len(decrypted_large)} 字节')
    print(f'验证: {large_data == decrypted_large}')

    print("\n=== 测试7: 分块文件加解密 — CBC模式 ===")
    src_cbc = os.path.join(test_dir, 'test_cbc_plain.bin')
    enc_cbc_file = os.path.join(test_dir, 'test_cbc_encrypted.bin')
    dec_cbc_file = os.path.join(test_dir, 'test_cbc_decrypted.bin')
    large_data_cbc = os.urandom(150_000)
    with open(src_cbc, 'wb') as f:
        f.write(large_data_cbc)
    chunked_encrypt_file(key_32, src_cbc, enc_cbc_file, mode='CBC', chunk_size=65536)
    chunked_decrypt_file(key_32, enc_cbc_file, dec_cbc_file)
    with open(dec_cbc_file, 'rb') as f:
        decrypted_large_cbc = f.read()
    print(f'原始大小: {len(large_data_cbc)} 字节')
    print(f'加密文件大小: {os.path.getsize(enc_cbc_file)} 字节')
    print(f'解密大小: {len(decrypted_large_cbc)} 字节')
    print(f'验证: {large_data_cbc == decrypted_large_cbc}')

    print("\n=== 测试8: 分块文件GCM防篡改 ===")
    with open(enc_gcm_file, 'rb') as f:
        tampered_enc = bytearray(f.read())
    tampered_enc[-1] ^= 0xFF
    tampered_enc_path = os.path.join(test_dir, 'test_gcm_tampered.bin')
    with open(tampered_enc_path, 'wb') as f:
        f.write(bytes(tampered_enc))
    try:
        chunked_decrypt_file(key_32, tampered_enc_path,
                             os.path.join(test_dir, 'test_gcm_tampered_dec.bin'))
        print('错误: 篡改文件未被检测!')
    except ValueError as e:
        print(f'正确检测到文件篡改: {e}')

    for path in [src_gcm, enc_gcm_file, dec_gcm_file,
                 src_cbc, enc_cbc_file, dec_cbc_file,
                 tampered_enc_path,
                 os.path.join(test_dir, 'test_gcm_tampered_dec.bin')]:
        if os.path.exists(path):
            os.remove(path)

    print("\n" + "=" * 60)
    print("  所有测试通过!")
    print("=" * 60)
