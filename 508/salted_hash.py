import hashlib
import hmac
import secrets
import binascii
import re
from collections import deque
from datetime import datetime, timezone

try:
    import bcrypt
except ImportError:
    bcrypt = None

try:
    from argon2 import PasswordHasher as Argon2Hasher
    from argon2.exceptions import VerifyMismatchError
except ImportError:
    Argon2Hasher = None
    VerifyMismatchError = None


def generate_salt(length=16):
    return secrets.token_bytes(length)


def validate_password_strength(password, min_length=8, require_upper=True,
                               require_lower=True, require_digit=True, require_special=True):
    if not isinstance(password, str):
        raise ValueError("Password must be a string.")

    errors = []

    if len(password) < min_length:
        errors.append(f"密码长度至少需要 {min_length} 个字符（当前 {len(password)} 个）")

    if require_upper and not re.search(r"[A-Z]", password):
        errors.append("密码需要包含至少一个大写字母")

    if require_lower and not re.search(r"[a-z]", password):
        errors.append("密码需要包含至少一个小写字母")

    if require_digit and not re.search(r"\d", password):
        errors.append("密码需要包含至少一个数字")

    if require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\\/\[\]\'`~]', password):
        errors.append("密码需要包含至少一个特殊字符（如 !@#$%^&* 等）")

    common_patterns = ["123456", "password", "qwerty", "abc123", "password1", "12345678"]
    for pattern in common_patterns:
        if pattern.lower() in password.lower():
            errors.append(f"密码不能包含常见弱口令模式 '{pattern}'")
            break

    if errors:
        raise ValueError("\n".join(errors))

    return True


def hash_password(password, salt=None, algorithm="sha256", work_factor=None):
    if algorithm == "md5":
        raise ValueError("MD5 is cryptographically broken. Use SHA-256, bcrypt, or argon2 instead.")

    if isinstance(password, str):
        password = password.encode("utf-8")

    if algorithm == "sha256":
        if salt is None:
            salt = generate_salt()
        hasher = hashlib.sha256()
        hasher.update(salt + password)
        hashed = hasher.digest()
        return binascii.hexlify(salt).decode("ascii"), binascii.hexlify(hashed).decode("ascii")

    elif algorithm == "bcrypt":
        if bcrypt is None:
            raise RuntimeError("bcrypt module is not installed. Run: pip install bcrypt")
        if work_factor is None:
            work_factor = 12
        salted_hash = bcrypt.hashpw(password, bcrypt.gensalt(rounds=work_factor))
        return "", salted_hash.decode("ascii")

    elif algorithm == "argon2":
        if Argon2Hasher is None:
            raise RuntimeError("argon2-cffi module is not installed. Run: pip install argon2-cffi")
        if work_factor is None:
            work_factor = 3
        ph = Argon2Hasher(time_cost=work_factor, memory_cost=65536, parallelism=4)
        return "", ph.hash(password.decode("utf-8"))

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'sha256', 'bcrypt', or 'argon2'.")


def verify_password(password, stored_salt_hex, stored_hash_hex, algorithm="sha256"):
    if algorithm == "md5":
        raise ValueError("MD5 is cryptographically broken. Use SHA-256, bcrypt, or argon2 instead.")

    if isinstance(password, str):
        password = password.encode("utf-8")

    if algorithm == "sha256":
        salt = binascii.unhexlify(stored_salt_hex)
        hasher = hashlib.sha256()
        hasher.update(salt + password)
        computed_hash = binascii.hexlify(hasher.digest()).decode("ascii")
        return hmac.compare_digest(computed_hash, stored_hash_hex)

    elif algorithm == "bcrypt":
        if bcrypt is None:
            raise RuntimeError("bcrypt module is not installed. Run: pip install bcrypt")
        try:
            return bcrypt.checkpw(password, stored_hash_hex.encode("utf-8"))
        except Exception:
            return False

    elif algorithm == "argon2":
        if Argon2Hasher is None:
            raise RuntimeError("argon2-cffi module is not installed. Run: pip install argon2-cffi")
        ph = Argon2Hasher()
        try:
            return ph.verify(stored_hash_hex, password.decode("utf-8"))
        except VerifyMismatchError:
            return False
        except Exception:
            return False

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'sha256', 'bcrypt', or 'argon2'.")


class PasswordHistory:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self._history = deque(maxlen=max_history)

    def add(self, password_hash, algorithm="sha256"):
        entry = {
            "hash": password_hash,
            "algorithm": algorithm,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._history.append(entry)

    def check_reuse(self, password, algorithm="sha256", salt_hex=""):
        for entry in self._history:
            if entry["algorithm"] != algorithm:
                continue
            if verify_password(password, salt_hex, entry["hash"], algorithm):
                return True
        return False

    def get_history(self):
        return list(self._history)

    def clear(self):
        self._history.clear()


def security_recommendations():
    return [
        "1. 首选算法：生产环境强烈推荐使用 Argon2（抗GPU暴力破解的密码哈希竞赛冠军算法）。",
        "2. bcrypt 备选：bcrypt 经过广泛验证，自适应 cost 参数可抵御硬件升级。",
        "3. SHA-256 局限：SHA-256 计算太快，需结合高迭代次数或 KDF 使用。",
        "4. 密码强度：最小 8 位，包含大小写字母、数字、特殊字符四种类别。",
        "5. 密码历史：禁止重用最近 5 个密码，配合密码过期策略。",
        "6. 时序攻击：所有哈希比较使用恒定时间函数 hmac.compare_digest()。",
        "7. 不要自行实现：优先使用 passlib、Werkzeug 等成熟安全库。"
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("密码安全存储系统 - 综合测试")
    print("=" * 60)

    print("\n=== 1. 密码强度测试 ===")
    test_cases = [
        ("short", "太短"),
        ("alllowercase123", "缺大写和特殊字符"),
        ("ALLUPPERCASE123!", "缺小写"),
        ("NoDigitsHere!", "缺数字"),
        ("NoSpecialChar123", "缺特殊字符"),
        ("password123!", "包含弱口令模式"),
        ("MyStr0ng!Pass", "符合要求"),
    ]

    for pwd, desc in test_cases:
        try:
            validate_password_strength(pwd)
            print(f"  ✓ '{pwd}' -> 通过 ({desc})")
        except ValueError as e:
            first_error = str(e).split("\n")[0]
            print(f"  ✗ '{pwd}' -> 拒绝: {first_error}")

    print("\n=== 2. 三种哈希算法对比 ===")
    strong_pwd = "MyStr0ng!Pass"

    for algo in ["sha256", "bcrypt", "argon2"]:
        print(f"\n  --- {algo.upper()} ---")
        salt_hex, hash_hex = hash_password(strong_pwd, algorithm=algo)
        if algo == "sha256":
            print(f"  Salt:  {salt_hex}")
        print(f"  Hash:  {hash_hex[:80]}..." if len(hash_hex) > 80 else f"  Hash:  {hash_hex}")
        print(f"  Verify correct: {verify_password(strong_pwd, salt_hex, hash_hex, algorithm=algo)}")
        print(f"  Verify wrong:   {verify_password('Wrong123!', salt_hex, hash_hex, algorithm=algo)}")

    print("\n=== 3. 密码历史记录测试 ===")
    history = PasswordHistory(max_history=5)
    passwords = ["OldPass1!", "OldPass2!", "OldPass3!", "OldPass4!", "OldPass5!", "NewPass6!"]

    print("  依次设置 5 个历史密码...")
    for i, pwd in enumerate(passwords[:5], 1):
        _, hash_hex = hash_password(pwd, algorithm="bcrypt")
        history.add(hash_hex, algorithm="bcrypt")
        print(f"  {i}. {pwd} -> 已存入历史")

    print(f"\n  检查密码重用（尝试 'OldPass3!'）:")
    reused = history.check_reuse("OldPass3!", algorithm="bcrypt")
    print(f"  是否重用: {reused} (预期: True)")

    print(f"\n  检查新密码（尝试 'NewPass6!'）:")
    reused = history.check_reuse("NewPass6!", algorithm="bcrypt")
    print(f"  是否重用: {reused} (预期: False)")

    print(f"\n  历史记录条数: {len(history.get_history())} (预期: 5)")

    print("\n=== 4. 安全建议 ===")
    for rec in security_recommendations():
        print(f"  {rec}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
