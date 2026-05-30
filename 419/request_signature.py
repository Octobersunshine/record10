import hmac
import hashlib
import time
import uuid
import urllib.parse
import secrets
from typing import Dict, Any, Optional, Tuple, Union, List
from collections import OrderedDict

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.exceptions import InvalidSignature
    _RSA_AVAILABLE = True
except ImportError:
    _RSA_AVAILABLE = False


class NonceStore:
    def __init__(self, expiry_seconds: int = 300):
        self._nonces: OrderedDict[str, float] = OrderedDict()
        self.expiry_seconds = expiry_seconds

    def _cleanup_expired(self) -> None:
        current_time = time.time()
        expired_keys = [k for k, v in self._nonces.items()
                        if current_time - v > self.expiry_seconds]
        for k in expired_keys:
            del self._nonces[k]

    def exists(self, nonce: str) -> bool:
        self._cleanup_expired()
        return nonce in self._nonces

    def add(self, nonce: str) -> bool:
        self._cleanup_expired()
        if nonce in self._nonces:
            return False
        self._nonces[nonce] = time.time()
        return True

    def size(self) -> int:
        self._cleanup_expired()
        return len(self._nonces)


class SignatureAlgorithm:
    HMAC_SHA1 = "HMAC-SHA1"
    HMAC_SHA256 = "HMAC-SHA256"
    RSA_SHA256 = "RSA-SHA256"

    @staticmethod
    def is_hmac(algorithm: str) -> bool:
        return algorithm in (SignatureAlgorithm.HMAC_SHA1, SignatureAlgorithm.HMAC_SHA256)

    @staticmethod
    def is_rsa(algorithm: str) -> bool:
        return algorithm == SignatureAlgorithm.RSA_SHA256


class RequestSigner:
    def __init__(self,
                 secret_key: Optional[str] = None,
                 private_key: Optional[str] = None,
                 public_key: Optional[str] = None,
                 algorithm: str = SignatureAlgorithm.HMAC_SHA256,
                 signature_header: str = "X-Signature",
                 timestamp_header: str = "X-Timestamp",
                 nonce_header: str = "X-Nonce",
                 algorithm_header: str = "X-Sign-Algorithm",
                 timestamp_tolerance: int = 300,
                 enable_timestamp_check: bool = True,
                 allow_future_timestamp: bool = True,
                 enable_nonce_check: bool = True,
                 nonce_expiry: int = 300,
                 nonce_store: Optional[NonceStore] = None):
        self.algorithm = algorithm
        self.secret_key = secret_key.encode("utf-8") if secret_key else None
        self.private_key = None
        self.public_key = None

        if SignatureAlgorithm.is_rsa(algorithm):
            if not _RSA_AVAILABLE:
                raise ImportError("cryptography library is required for RSA signatures. "
                                  "Install with: pip install cryptography")
            if private_key:
                self.private_key = serialization.load_pem_private_key(
                    private_key.encode("utf-8"), password=None)
            if public_key:
                self.public_key = serialization.load_pem_public_key(
                    public_key.encode("utf-8"))
        elif SignatureAlgorithm.is_hmac(algorithm):
            if not self.secret_key:
                raise ValueError("secret_key is required for HMAC algorithms")

        self.signature_header = signature_header
        self.timestamp_header = timestamp_header
        self.nonce_header = nonce_header
        self.algorithm_header = algorithm_header
        self.timestamp_tolerance = timestamp_tolerance
        self.enable_timestamp_check = enable_timestamp_check
        self.allow_future_timestamp = allow_future_timestamp
        self.enable_nonce_check = enable_nonce_check
        self.nonce_expiry = nonce_expiry
        self.nonce_store = nonce_store or NonceStore(nonce_expiry)

    def _get_hash_algorithm(self):
        if self.algorithm == SignatureAlgorithm.HMAC_SHA1:
            return hashlib.sha1
        elif self.algorithm in (SignatureAlgorithm.HMAC_SHA256, SignatureAlgorithm.RSA_SHA256):
            return hashlib.sha256
        return hashlib.sha256

    def _normalize_params(self, params: Dict[str, Any]) -> str:
        if not params:
            return ""
        sorted_items = sorted(params.items(), key=lambda x: x[0])
        normalized_parts = []
        for key, value in sorted_items:
            if isinstance(value, list):
                for v in sorted(value):
                    normalized_parts.append(f"{key}={urllib.parse.quote(str(v), safe='')}")
            else:
                normalized_parts.append(f"{key}={urllib.parse.quote(str(value), safe='')}")
        return "&".join(normalized_parts)

    def _build_sign_string(self, method: str, path: str, timestamp: int,
                           nonce: str, params: Optional[Dict[str, Any]] = None,
                           body: Optional[str] = None) -> str:
        method_upper = method.upper()
        normalized_params = self._normalize_params(params or {})
        body_hash = self._get_hash_algorithm()(
            (body or "").encode("utf-8")).hexdigest() if body is not None else ""
        sign_parts = [
            method_upper,
            path,
            str(timestamp),
            nonce,
            normalized_params,
            body_hash
        ]
        return "\n".join(sign_parts)

    def _sign(self, data: bytes) -> str:
        if SignatureAlgorithm.is_hmac(self.algorithm):
            hash_func = self._get_hash_algorithm()
            return hmac.new(self.secret_key, data, hash_func).hexdigest()
        elif SignatureAlgorithm.is_rsa(self.algorithm):
            if not self.private_key:
                raise ValueError("Private key is required for RSA signing")
            signature = self.private_key.sign(
                data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return signature.hex()
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _verify(self, data: bytes, signature: str) -> bool:
        if SignatureAlgorithm.is_hmac(self.algorithm):
            hash_func = self._get_hash_algorithm()
            expected = hmac.new(self.secret_key, data, hash_func).hexdigest()
            return hmac.compare_digest(expected, signature)
        elif SignatureAlgorithm.is_rsa(self.algorithm):
            if not self.public_key:
                raise ValueError("Public key is required for RSA verification")
            try:
                self.public_key.verify(
                    bytes.fromhex(signature),
                    data,
                    padding.PKCS1v15(),
                    hashes.SHA256()
                )
                return True
            except (InvalidSignature, ValueError):
                return False
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def generate_nonce(self) -> str:
        return secrets.token_hex(16)

    def generate_signature(self, method: str, path: str,
                           params: Optional[Dict[str, Any]] = None,
                           body: Optional[str] = None,
                           timestamp: Optional[int] = None,
                           nonce: Optional[str] = None) -> Tuple[str, int, str]:
        if timestamp is None:
            timestamp = int(time.time())
        if nonce is None:
            nonce = self.generate_nonce()
        sign_string = self._build_sign_string(method, path, timestamp, nonce, params, body)
        signature = self._sign(sign_string.encode("utf-8"))
        return signature, timestamp, nonce

    def generate_headers(self, method: str, path: str,
                         params: Optional[Dict[str, Any]] = None,
                         body: Optional[str] = None,
                         include_algorithm: bool = True) -> Dict[str, str]:
        signature, timestamp, nonce = self.generate_signature(method, path, params, body)
        headers = {
            self.signature_header: signature,
            self.timestamp_header: str(timestamp),
            self.nonce_header: nonce
        }
        if include_algorithm:
            headers[self.algorithm_header] = self.algorithm
        return headers

    def _check_timestamp(self, timestamp: int) -> bool:
        if not self.enable_timestamp_check:
            return True
        if timestamp is None:
            return False
        current_time = int(time.time())
        time_diff = current_time - timestamp
        if self.allow_future_timestamp:
            return abs(time_diff) <= self.timestamp_tolerance
        else:
            return 0 <= time_diff <= self.timestamp_tolerance

    def _check_nonce(self, nonce: str) -> bool:
        if not self.enable_nonce_check:
            return True
        if not nonce:
            return False
        return self.nonce_store.add(nonce)

    def verify_signature(self, method: str, path: str,
                         params: Optional[Dict[str, Any]] = None,
                         body: Optional[str] = None,
                         signature: str = "",
                         timestamp: Optional[int] = None,
                         nonce: Optional[str] = None) -> bool:
        if not signature:
            return False
        if not self._check_timestamp(timestamp):
            return False
        if not self._check_nonce(nonce):
            return False
        sign_string = self._build_sign_string(method, path, timestamp, nonce, params, body)
        return self._verify(sign_string.encode("utf-8"), signature)

    def verify_request(self, method: str, path: str,
                       headers: Dict[str, str],
                       params: Optional[Dict[str, Any]] = None,
                       body: Optional[str] = None) -> bool:
        signature = headers.get(self.signature_header, "")
        timestamp_str = headers.get(self.timestamp_header, "")
        nonce = headers.get(self.nonce_header, "")
        try:
            timestamp = int(timestamp_str) if timestamp_str else None
        except (ValueError, TypeError):
            return False
        return self.verify_signature(method, path, params, body, signature, timestamp, nonce)

    def debug_signature(self, method: str, path: str,
                        params: Optional[Dict[str, Any]] = None,
                        body: Optional[str] = None,
                        timestamp: Optional[int] = None,
                        nonce: Optional[str] = None) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = int(time.time())
        if nonce is None:
            nonce = self.generate_nonce()
        normalized_params = self._normalize_params(params or {})
        body_hash = self._get_hash_algorithm()(
            (body or "").encode("utf-8")).hexdigest() if body is not None else ""
        sign_string = self._build_sign_string(method, path, timestamp, nonce, params, body)
        signature = self._sign(sign_string.encode("utf-8"))
        return {
            "algorithm": self.algorithm,
            "method": method.upper(),
            "path": path,
            "timestamp": timestamp,
            "nonce": nonce,
            "normalized_params": normalized_params,
            "params": params,
            "body": body,
            "body_hash": body_hash,
            "sign_string": sign_string,
            "sign_string_encoded": sign_string.encode("utf-8"),
            "signature": signature,
            "headers": {
                self.signature_header: signature,
                self.timestamp_header: str(timestamp),
                self.nonce_header: nonce,
                self.algorithm_header: self.algorithm
            }
        }

    def debug_verify(self, method: str, path: str,
                     headers: Dict[str, str],
                     params: Optional[Dict[str, Any]] = None,
                     body: Optional[str] = None) -> Dict[str, Any]:
        signature = headers.get(self.signature_header, "")
        timestamp_str = headers.get(self.timestamp_header, "")
        nonce = headers.get(self.nonce_header, "")
        algorithm = headers.get(self.algorithm_header, self.algorithm)
        try:
            timestamp = int(timestamp_str) if timestamp_str else None
        except (ValueError, TypeError):
            timestamp = None
        normalized_params = self._normalize_params(params or {})
        body_hash = self._get_hash_algorithm()(
            (body or "").encode("utf-8")).hexdigest() if body is not None else ""
        sign_string = self._build_sign_string(method, path, timestamp, nonce, params, body) if timestamp and nonce else None
        timestamp_valid = self._check_timestamp(timestamp) if timestamp is not None else False
        nonce_valid = self._check_nonce(nonce) if nonce else False
        signature_valid = False
        expected_signature = None
        if sign_string:
            expected_signature = self._sign(sign_string.encode("utf-8"))
            signature_valid = self._verify(sign_string.encode("utf-8"), signature)
        return {
            "provided": {
                "signature": signature,
                "timestamp": timestamp,
                "nonce": nonce,
                "algorithm": algorithm
            },
            "checks": {
                "timestamp_valid": timestamp_valid,
                "nonce_valid": nonce_valid,
                "signature_valid": signature_valid,
                "all_valid": timestamp_valid and nonce_valid and signature_valid
            },
            "normalized_params": normalized_params,
            "body_hash": body_hash,
            "sign_string": sign_string,
            "expected_signature": expected_signature,
            "algorithm": self.algorithm
        }


def generate_rsa_keys() -> Tuple[str, str]:
    if not _RSA_AVAILABLE:
        raise ImportError("cryptography library is required for RSA key generation.")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode("utf-8")
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")
    return private_pem, public_pem


if __name__ == "__main__":
    method = "POST"
    path = "/api/v1/orders"
    params = {"user_id": "1001", "amount": "99.99", "product_id": "P500"}
    body = '{"item":"book","quantity":2}'

    print("=" * 60)
    print("HMAC-SHA256 测试")
    print("=" * 60)

    signer_sha256 = RequestSigner(secret_key="my-secret-key-123",
                                   algorithm=SignatureAlgorithm.HMAC_SHA256)

    print("\n=== 签名生成 ===")
    headers = signer_sha256.generate_headers(method, path, params, body)
    print(f"请求方法: {method}")
    print(f"请求路径: {path}")
    print(f"请求参数: {params}")
    print(f"请求体: {body}")
    print(f"签名头: {headers[signer_sha256.signature_header]}")
    print(f"时间戳头: {headers[signer_sha256.timestamp_header]}")
    print(f"Nonce头: {headers[signer_sha256.nonce_header]}")
    print(f"算法头: {headers[signer_sha256.algorithm_header]}")

    print("\n=== 签名验证 ===")
    is_valid = signer_sha256.verify_request(method, path, headers, params, body)
    print(f"原始请求验证结果: {'通过' if is_valid else '失败'}")

    print("\n=== 重放攻击测试 ===")
    is_valid_replay = signer_sha256.verify_request(method, path, headers, params, body)
    print(f"重放请求(相同nonce)验证结果: {'通过' if is_valid_replay else '失败'}")

    print("\n=== 篡改测试 ===")
    headers2 = signer_sha256.generate_headers(method, path, params, body)
    tampered_params = {"user_id": "1001", "amount": "999.99", "product_id": "P500"}
    is_valid_tampered = signer_sha256.verify_request(method, path, headers2, tampered_params, body)
    print(f"参数篡改后验证结果: {'通过' if is_valid_tampered else '失败'}")

    print("\n=== 时间戳容差测试 ===")
    past_timestamp = int(time.time()) - 280
    past_nonce = signer_sha256.generate_nonce()
    past_sig, _, _ = signer_sha256.generate_signature(method, path, params, body, past_timestamp, past_nonce)
    past_headers = {
        signer_sha256.signature_header: past_sig,
        signer_sha256.timestamp_header: str(past_timestamp),
        signer_sha256.nonce_header: past_nonce
    }
    is_valid_past = signer_sha256.verify_request(method, path, past_headers, params, body)
    print(f"过去280秒(在容差内)验证结果: {'通过' if is_valid_past else '失败'}")

    expired_timestamp = int(time.time()) - 600
    expired_nonce = signer_sha256.generate_nonce()
    expired_sig, _, _ = signer_sha256.generate_signature(method, path, params, body, expired_timestamp, expired_nonce)
    expired_headers = {
        signer_sha256.signature_header: expired_sig,
        signer_sha256.timestamp_header: str(expired_timestamp),
        signer_sha256.nonce_header: expired_nonce
    }
    is_valid_expired = signer_sha256.verify_request(method, path, expired_headers, params, body)
    print(f"过去600秒(超出容差)验证结果: {'通过' if is_valid_expired else '失败'}")

    print("\n" + "=" * 60)
    print("HMAC-SHA1 测试")
    print("=" * 60)

    signer_sha1 = RequestSigner(secret_key="my-secret-key-123",
                                 algorithm=SignatureAlgorithm.HMAC_SHA1)
    headers_sha1 = signer_sha1.generate_headers(method, path, params, body)
    is_valid_sha1 = signer_sha1.verify_request(method, path, headers_sha1, params, body)
    print(f"HMAC-SHA1 签名长度: {len(headers_sha1[signer_sha1.signature_header])} 字符")
    print(f"HMAC-SHA1 验证结果: {'通过' if is_valid_sha1 else '失败'}")

    print("\n" + "=" * 60)
    print("RSA-SHA256 测试")
    print("=" * 60)

    if _RSA_AVAILABLE:
        private_pem, public_pem = generate_rsa_keys()
        print("RSA密钥对生成成功")

        signer_rsa = RequestSigner(private_key=private_pem,
                                    algorithm=SignatureAlgorithm.RSA_SHA256)
        verifier_rsa = RequestSigner(public_key=public_pem,
                                      algorithm=SignatureAlgorithm.RSA_SHA256)

        headers_rsa = signer_rsa.generate_headers(method, path, params, body)
        print(f"RSA签名长度: {len(headers_rsa[signer_rsa.signature_header])} 字符")

        is_valid_rsa = verifier_rsa.verify_request(method, path, headers_rsa, params, body)
        print(f"RSA签名验证结果: {'通过' if is_valid_rsa else '失败'}")

        headers_rsa_tampered = headers_rsa.copy()
        headers_rsa_tampered[signer_rsa.signature_header] = "0" * len(headers_rsa[signer_rsa.signature_header])
        is_valid_rsa_tampered = verifier_rsa.verify_request(method, path, headers_rsa_tampered, params, body)
        print(f"RSA签名篡改后验证结果: {'通过' if is_valid_rsa_tampered else '失败'}")
    else:
        print("未安装 cryptography 库，跳过RSA测试")
        print("安装命令: pip install cryptography")

    print("\n" + "=" * 60)
    print("签名调试接口测试")
    print("=" * 60)

    debug_signer = RequestSigner(secret_key="debug-key", algorithm=SignatureAlgorithm.HMAC_SHA256)
    debug_info = debug_signer.debug_signature("GET", "/api/users", {"page": "1", "limit": "10"})
    print(f"\n算法: {debug_info['algorithm']}")
    print(f"方法: {debug_info['method']}")
    print(f"路径: {debug_info['path']}")
    print(f"时间戳: {debug_info['timestamp']}")
    print(f"Nonce: {debug_info['nonce']}")
    print(f"参数排序后: {debug_info['normalized_params']}")
    print(f"Body哈希: {debug_info['body_hash']}")
    print(f"\n待签名字符串:")
    print(repr(debug_info['sign_string']))
    print(f"\n签名结果: {debug_info['signature']}")

    print("\n=== 验证调试 ===")
    debug_headers = debug_info["headers"]
    verify_debug = debug_signer.debug_verify("GET", "/api/users", debug_headers, {"page": "1", "limit": "10"})
    print(f"时间戳检查: {'通过' if verify_debug['checks']['timestamp_valid'] else '失败'}")
    print(f"Nonce检查: {'通过' if verify_debug['checks']['nonce_valid'] else '失败'}")
    print(f"签名检查: {'通过' if verify_debug['checks']['signature_valid'] else '失败'}")
    print(f"整体结果: {'通过' if verify_debug['checks']['all_valid'] else '失败'}")

    print("\n=== 禁用时间戳和Nonce检查 ===")
    signer_no_checks = RequestSigner(secret_key="test-key",
                                      enable_timestamp_check=False,
                                      enable_nonce_check=False)
    old_ts = int(time.time()) - 3600
    old_nonce = "reused-nonce-12345"
    old_sig, _, _ = signer_no_checks.generate_signature(method, path, params, body, old_ts, old_nonce)
    old_headers = {
        signer_no_checks.signature_header: old_sig,
        signer_no_checks.timestamp_header: str(old_ts),
        signer_no_checks.nonce_header: old_nonce
    }
    is_valid1 = signer_no_checks.verify_request(method, path, old_headers, params, body)
    is_valid2 = signer_no_checks.verify_request(method, path, old_headers, params, body)
    print(f"第一次验证(过期时间戳+重复nonce): {'通过' if is_valid1 else '失败'}")
    print(f"第二次验证(重放): {'通过' if is_valid2 else '失败'}")
