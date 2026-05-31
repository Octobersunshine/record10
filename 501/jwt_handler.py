import jwt
import time
import uuid
import secrets
from typing import Optional, Dict, Any, Tuple
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


HS256 = "HS256"
RS256 = "RS256"
SUPPORTED_ALGORITHMS = [HS256, RS256]

SECRET_KEY = "your-secret-key-change-in-production"

_private_key = None
_public_key = None

_token_blacklist: set = set()
_refresh_token_store: Dict[str, Dict[str, Any]] = {}


def generate_rsa_key_pair(key_size: int = 2048) -> Tuple[bytes, bytes]:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend()
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem


def load_rsa_keys(private_pem: Optional[bytes] = None, public_pem: Optional[bytes] = None) -> None:
    global _private_key, _public_key
    if private_pem:
        _private_key = serialization.load_pem_private_key(
            private_pem,
            password=None,
            backend=default_backend()
        )
    if public_pem:
        _public_key = serialization.load_pem_public_key(
            public_pem,
            backend=default_backend()
        )


def ensure_rsa_keys() -> None:
    global _private_key, _public_key
    if _private_key is None or _public_key is None:
        private_pem, public_pem = generate_rsa_key_pair()
        load_rsa_keys(private_pem, public_pem)


def _get_signing_key(algorithm: str):
    if algorithm == HS256:
        return SECRET_KEY
    elif algorithm == RS256:
        ensure_rsa_keys()
        return _private_key
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _get_verifying_key(algorithm: str):
    if algorithm == HS256:
        return SECRET_KEY
    elif algorithm == RS256:
        ensure_rsa_keys()
        return _public_key
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _generate_jti() -> str:
    return str(uuid.uuid4())


def generate_token(
    user_id: str,
    expires_in_seconds: int,
    algorithm: str = HS256,
    token_type: str = "access"
) -> str:
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Algorithm must be one of {SUPPORTED_ALGORITHMS}")

    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + expires_in_seconds,
        "iat": int(time.time()),
        "jti": _generate_jti(),
        "type": token_type,
    }
    key = _get_signing_key(algorithm)
    token = jwt.encode(payload, key, algorithm=algorithm)
    return token


def generate_refresh_token(user_id: str, expires_in_days: int = 7) -> Tuple[str, str]:
    refresh_token = secrets.token_urlsafe(64)
    jti = _generate_jti()
    expires_at = int(time.time()) + (expires_in_days * 24 * 60 * 60)
    _refresh_token_store[refresh_token] = {
        "user_id": user_id,
        "jti": jti,
        "expires_at": expires_at,
        "created_at": int(time.time()),
        "revoked": False,
    }
    return refresh_token, jti


def verify_token(token: str, algorithm: str = HS256, token_type: Optional[str] = None) -> dict:
    if algorithm not in SUPPORTED_ALGORITHMS:
        return {"user_id": None, "status": "invalid_algorithm", "message": f"Unsupported algorithm: {algorithm}"}

    try:
        key = _get_verifying_key(algorithm)
        payload = jwt.decode(token, key, algorithms=[algorithm])

        if "jti" in payload and is_token_revoked(payload["jti"]):
            return {"user_id": None, "status": "revoked", "message": "Token has been revoked"}

        if "exp" in payload and payload["exp"] < time.time():
            return {"user_id": None, "status": "expired", "message": "Token has expired"}

        if "user_id" not in payload:
            return {"user_id": None, "status": "invalid_format", "message": "Missing user_id in payload"}

        if token_type and payload.get("type") != token_type:
            return {"user_id": None, "status": "invalid_type", "message": f"Expected {token_type} token"}

        return {
            "user_id": payload["user_id"],
            "status": "valid",
            "message": "Token is valid",
            "jti": payload.get("jti"),
            "exp": payload.get("exp"),
        }
    except jwt.ExpiredSignatureError:
        return {"user_id": None, "status": "expired", "message": "Token has expired"}
    except jwt.InvalidSignatureError:
        return {"user_id": None, "status": "invalid_signature", "message": "Signature verification failed"}
    except jwt.DecodeError:
        return {"user_id": None, "status": "invalid_format", "message": "Token format is invalid"}
    except jwt.InvalidTokenError:
        return {"user_id": None, "status": "invalid", "message": "Token is invalid"}


def refresh_access_token(
    refresh_token: str,
    access_expires_in_seconds: int = 3600,
    algorithm: str = HS256
) -> dict:
    refresh_data = _refresh_token_store.get(refresh_token)
    if not refresh_data:
        return {"access_token": None, "status": "invalid", "message": "Invalid refresh token"}

    if refresh_data["revoked"]:
        return {"access_token": None, "status": "revoked", "message": "Refresh token has been revoked"}

    if refresh_data["expires_at"] < time.time():
        return {"access_token": None, "status": "expired", "message": "Refresh token has expired"}

    new_access_token = generate_token(
        user_id=refresh_data["user_id"],
        expires_in_seconds=access_expires_in_seconds,
        algorithm=algorithm,
        token_type="access"
    )

    return {
        "access_token": new_access_token,
        "status": "valid",
        "message": "Token refreshed successfully",
        "user_id": refresh_data["user_id"],
    }


def revoke_token(jti: str) -> None:
    _token_blacklist.add(jti)


def revoke_refresh_token(refresh_token: str) -> bool:
    if refresh_token in _refresh_token_store:
        _refresh_token_store[refresh_token]["revoked"] = True
        return True
    return False


def is_token_revoked(jti: str) -> bool:
    return jti in _token_blacklist


def cleanup_expired_tokens() -> None:
    current_time = time.time()
    expired_tokens = [
        token for token, data in _refresh_token_store.items()
        if data["expires_at"] < current_time
    ]
    for token in expired_tokens:
        del _refresh_token_store[token]


if __name__ == "__main__":
    user_id = "user_12345"
    expires_in = 3600

    print("=" * 60)
    print("HS256 Algorithm Tests")
    print("=" * 60)

    hs_token = generate_token(user_id, expires_in, algorithm=HS256)
    print(f"Generated HS256 JWT: {hs_token[:50]}...")

    hs_result = verify_token(hs_token, algorithm=HS256)
    print(f"Valid HS256 token: {hs_result}")

    hs_expired = generate_token(user_id, -1, algorithm=HS256)
    print(f"Expired HS256: {verify_token(hs_expired, algorithm=HS256)}")

    print("\n" + "=" * 60)
    print("RS256 Algorithm Tests")
    print("=" * 60)

    private_pem, public_pem = generate_rsa_key_pair()
    load_rsa_keys(private_pem, public_pem)

    rs_token = generate_token(user_id, expires_in, algorithm=RS256)
    print(f"Generated RS256 JWT: {rs_token[:50]}...")

    rs_result = verify_token(rs_token, algorithm=RS256)
    print(f"Valid RS256 token: {rs_result}")

    rs_tampered = rs_token[:-5] + "XXXXX"
    print(f"Tampered RS256: {verify_token(rs_tampered, algorithm=RS256)}")

    print("\n" + "=" * 60)
    print("Refresh Token Tests")
    print("=" * 60)

    refresh_token, refresh_jti = generate_refresh_token(user_id, expires_in_days=7)
    print(f"Generated refresh token: {refresh_token[:30]}...")
    print(f"Refresh token JTI: {refresh_jti}")

    refresh_result = refresh_access_token(refresh_token, algorithm=HS256)
    print(f"Refresh result: status={refresh_result['status']}, user_id={refresh_result['user_id']}")
    print(f"New access token: {refresh_result['access_token'][:50]}...")

    new_access_verify = verify_token(refresh_result["access_token"], algorithm=HS256)
    print(f"New access token verified: {new_access_verify['status']}")

    print("\n" + "=" * 60)
    print("Token Revocation Tests")
    print("=" * 60)

    token_to_revoke = generate_token(user_id, expires_in, algorithm=HS256)
    verify_before = verify_token(token_to_revoke, algorithm=HS256)
    print(f"Before revocation: {verify_before['status']}")

    revoke_token(verify_before["jti"])
    verify_after = verify_token(token_to_revoke, algorithm=HS256)
    print(f"After revocation: {verify_after}")

    print(f"\nRefresh token before revocation: {refresh_access_token(refresh_token)['status']}")
    revoke_refresh_token(refresh_token)
    print(f"Refresh token after revocation: {refresh_access_token(refresh_token)['status']}")

    print("\n" + "=" * 60)
    print("Token Type Verification")
    print("=" * 60)

    access_token = generate_token(user_id, expires_in, token_type="access")
    refresh_attempt = verify_token(access_token, token_type="refresh")
    print(f"Verify access token as refresh: {refresh_attempt}")

    correct_verify = verify_token(access_token, token_type="access")
    print(f"Verify access token as access: {correct_verify['status']}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
