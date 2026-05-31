import os
import math
import random
import secrets
import string
import argparse
from enum import Enum
from typing import List, Tuple, Dict


AMBIGUOUS_CHARS = "0O1lI"


class RandomMode(str, Enum):
    STANDARD = "standard"
    SECURE = "secure"
    TRUE_RANDOM = "true_random"


class PasswordStrength(str, Enum):
    WEAK = "weak"
    FAIR = "fair"
    GOOD = "good"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


def _get_choice_function(mode: RandomMode):
    if mode == RandomMode.STANDARD:
        return random.choice
    elif mode == RandomMode.SECURE:
        return secrets.choice
    elif mode == RandomMode.TRUE_RANDOM:
        return _true_random_choice
    else:
        raise ValueError(f"未知的随机模式: {mode}")


def _true_random_choice(seq):
    if not seq:
        raise IndexError("不能从空序列中选择")
    n = len(seq)
    if n == 1:
        return seq[0]
    bits_needed = n.bit_length()
    bytes_needed = (bits_needed + 7) // 8
    while True:
        rand_bytes = os.urandom(bytes_needed)
        rand_int = int.from_bytes(rand_bytes, byteorder="big")
        rand_int = rand_int % n
        return seq[rand_int]


def _remove_ambiguous_chars(charset: str) -> str:
    return ''.join(c for c in charset if c not in AMBIGUOUS_CHARS)


def calculate_entropy(password: str) -> float:
    """
    计算密码的熵值（比特）

    Args:
        password: 密码字符串

    Returns:
        熵值（比特）
    """
    if not password:
        return 0.0

    charset_size = 0
    has_lower = has_upper = has_digit = has_special = False

    for c in password:
        if c in string.ascii_lowercase:
            has_lower = True
        elif c in string.ascii_uppercase:
            has_upper = True
        elif c in string.digits:
            has_digit = True
        else:
            has_special = True

    if has_lower:
        charset_size += 26
    if has_upper:
        charset_size += 26
    if has_digit:
        charset_size += 10
    if has_special:
        charset_size += 32

    if charset_size == 0:
        return 0.0

    return len(password) * math.log2(charset_size)


def get_password_strength(entropy: float) -> PasswordStrength:
    """
    根据熵值判断密码强度等级

    Args:
        entropy: 熵值（比特）

    Returns:
        密码强度等级
    """
    if entropy < 28:
        return PasswordStrength.WEAK
    elif entropy < 36:
        return PasswordStrength.FAIR
    elif entropy < 60:
        return PasswordStrength.GOOD
    elif entropy < 80:
        return PasswordStrength.STRONG
    else:
        return PasswordStrength.VERY_STRONG


def analyze_string_quality(s: str) -> Dict:
    """
    分析随机字符串质量

    Args:
        s: 字符串

    Returns:
        包含质量分析信息的字典
    """
    entropy = calculate_entropy(s)
    strength = get_password_strength(entropy)

    has_lower = any(c in string.ascii_lowercase for c in s)
    has_upper = any(c in string.ascii_uppercase for c in s)
    has_digit = any(c in string.digits for c in s)
    has_special = any(c not in string.ascii_letters + string.digits for c in s)

    meets_complexity = has_lower and has_upper and has_digit and has_special

    strength_names = {
        PasswordStrength.WEAK: "弱",
        PasswordStrength.FAIR: "一般",
        PasswordStrength.GOOD: "良好",
        PasswordStrength.STRONG: "强",
        PasswordStrength.VERY_STRONG: "非常强"
    }

    return {
        "length": len(s),
        "entropy": round(entropy, 2),
        "strength": strength_names[strength],
        "has_lowercase": has_lower,
        "has_uppercase": has_upper,
        "has_digit": has_digit,
        "has_special": has_special,
        "meets_complexity": meets_complexity,
        "unique_chars": len(set(s))
    }


def _ensure_character_types(
    s: str,
    include_digits: bool,
    include_lowercase: bool,
    include_uppercase: bool,
    include_special: bool,
    charset: str,
    choice_func,
    no_ambiguous: bool
) -> str:
    """
    确保字符串包含所有要求的字符类型
    """
    s_list = list(s)
    pos = 0

    lower_chars = ''.join(c for c in string.ascii_lowercase if c not in (AMBIGUOUS_CHARS if no_ambiguous else ''))
    upper_chars = ''.join(c for c in string.ascii_uppercase if c not in (AMBIGUOUS_CHARS if no_ambiguous else ''))
    digit_chars = ''.join(c for c in string.digits if c not in (AMBIGUOUS_CHARS if no_ambiguous else ''))
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    if include_lowercase and not any(c in lower_chars for c in s_list):
        s_list[pos] = choice_func(lower_chars)
        pos += 1
    if include_uppercase and not any(c in upper_chars for c in s_list):
        s_list[pos] = choice_func(upper_chars)
        pos += 1
    if include_digits and not any(c in digit_chars for c in s_list):
        s_list[pos] = choice_func(digit_chars)
        pos += 1
    if include_special and not any(c in special_chars for c in s_list):
        s_list[pos] = choice_func(special_chars)
        pos += 1

    return ''.join(s_list)


def generate_random_string(
    length: int = 12,
    include_digits: bool = True,
    include_lowercase: bool = True,
    include_uppercase: bool = True,
    include_special: bool = True,
    custom_chars: str = "",
    mode: RandomMode = RandomMode.STANDARD,
    no_ambiguous: bool = False,
    enforce_complexity: bool = False
) -> str:
    """
    生成随机字符串

    Args:
        length: 字符串长度，默认12
        include_digits: 是否包含数字，默认True
        include_lowercase: 是否包含小写字母，默认True
        include_uppercase: 是否包含大写字母，默认True
        include_special: 是否包含特殊字符，默认True
        custom_chars: 自定义字符集，非空时覆盖其他选项
        mode: 随机模式
            - standard: 标准模式（random模块，速度快但不安全）
            - secure: 安全模式（secrets模块，加密安全伪随机）
            - true_random: 真随机模式（os.urandom系统熵源）
        no_ambiguous: 是否排除歧义字符（0, O, 1, l, I）
        enforce_complexity: 是否强制满足密码复杂度（至少包含大小写、数字、特殊字符）

    Returns:
        生成的随机字符串

    Raises:
        ValueError: 参数无效时抛出
    """
    if length <= 0:
        raise ValueError("长度必须大于0")

    if enforce_complexity:
        if length < 4:
            raise ValueError("启用复杂度要求时，长度至少为4")
        include_digits = True
        include_lowercase = True
        include_uppercase = True
        include_special = True

    if custom_chars:
        charset = custom_chars
        if no_ambiguous:
            charset = _remove_ambiguous_chars(charset)
    else:
        charset = ""
        if include_digits:
            digits = string.digits
            if no_ambiguous:
                digits = _remove_ambiguous_chars(digits)
            charset += digits
        if include_lowercase:
            lowers = string.ascii_lowercase
            if no_ambiguous:
                lowers = _remove_ambiguous_chars(lowers)
            charset += lowers
        if include_uppercase:
            uppers = string.ascii_uppercase
            if no_ambiguous:
                uppers = _remove_ambiguous_chars(uppers)
            charset += uppers
        if include_special:
            charset += "!@#$%^&*()_+-=[]{}|;:,.<>?"

    if not charset:
        raise ValueError("字符集不能为空")

    choice_func = _get_choice_function(mode)
    result = ''.join(choice_func(charset) for _ in range(length))

    if enforce_complexity and not custom_chars:
        result = _ensure_character_types(
            result, include_digits, include_lowercase,
            include_uppercase, include_special, charset,
            choice_func, no_ambiguous
        )
        result_list = list(result)
        if mode == RandomMode.STANDARD:
            random.shuffle(result_list)
        else:
            for i in range(len(result_list) - 1, 0, -1):
                j = secrets.randbelow(i + 1)
                result_list[i], result_list[j] = result_list[j], result_list[i]
        result = ''.join(result_list)

    return result


def generate_multiple_strings(
    count: int = 1,
    length: int = 12,
    include_digits: bool = True,
    include_lowercase: bool = True,
    include_uppercase: bool = True,
    include_special: bool = True,
    custom_chars: str = "",
    mode: RandomMode = RandomMode.STANDARD,
    no_ambiguous: bool = False,
    enforce_complexity: bool = False
) -> List[str]:
    """
    批量生成随机字符串

    Args:
        count: 生成数量，默认1
        length: 每个字符串长度
        include_digits: 是否包含数字
        include_lowercase: 是否包含小写字母
        include_uppercase: 是否包含大写字母
        include_special: 是否包含特殊字符
        custom_chars: 自定义字符集
        mode: 随机模式
        no_ambiguous: 是否排除歧义字符
        enforce_complexity: 是否强制满足密码复杂度

    Returns:
        随机字符串列表

    Raises:
        ValueError: 参数无效时抛出
    """
    if count <= 0:
        raise ValueError("生成数量必须大于0")

    return [
        generate_random_string(
            length,
            include_digits,
            include_lowercase,
            include_uppercase,
            include_special,
            custom_chars,
            mode,
            no_ambiguous,
            enforce_complexity
        )
        for _ in range(count)
    ]


def main():
    parser = argparse.ArgumentParser(description="随机字符串生成工具")
    parser.add_argument("-l", "--length", type=int, default=12, help="字符串长度 (默认: 12)")
    parser.add_argument("-c", "--count", type=int, default=1, help="生成数量 (默认: 1)")
    parser.add_argument("--no-digits", action="store_true", help="不包含数字")
    parser.add_argument("--no-lower", action="store_true", help="不包含小写字母")
    parser.add_argument("--no-upper", action="store_true", help="不包含大写字母")
    parser.add_argument("--no-special", action="store_true", help="不包含特殊字符")
    parser.add_argument("--custom", type=str, default="", help="自定义字符集")
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="standard",
        choices=["standard", "secure", "true_random"],
        help="随机模式: standard(标准伪随机), secure(加密安全), true_random(真随机/系统熵源) (默认: standard)"
    )
    parser.add_argument("--no-ambiguous", action="store_true", help="排除歧义字符 (0, O, 1, l, I)")
    parser.add_argument("--enforce-complexity", action="store_true", help="强制密码复杂度 (必须包含大小写、数字、特殊字符)")
    parser.add_argument("--show-quality", action="store_true", help="显示字符串质量分析")

    args = parser.parse_args()

    try:
        mode = RandomMode(args.mode)
        strings = generate_multiple_strings(
            count=args.count,
            length=args.length,
            include_digits=not args.no_digits,
            include_lowercase=not args.no_lower,
            include_uppercase=not args.no_upper,
            include_special=not args.no_special,
            custom_chars=args.custom,
            mode=mode,
            no_ambiguous=args.no_ambiguous,
            enforce_complexity=args.enforce_complexity
        )

        mode_names = {
            RandomMode.STANDARD: "标准模式 (random)",
            RandomMode.SECURE: "安全模式 (secrets/CSPRNG)",
            RandomMode.TRUE_RANDOM: "真随机模式 (os.urandom/系统熵源)"
        }
        print(f"使用随机模式: {mode_names[mode]}")
        print(f"生成 {len(strings)} 个长度为 {args.length} 的随机字符串:\n")

        for i, s in enumerate(strings, 1):
            if args.show_quality:
                quality = analyze_string_quality(s)
                print(f"{i}. {s}")
                print(f"   熵值: {quality['entropy']} bits | 强度: {quality['strength']}")
                print(f"   包含: 小写={quality['has_lowercase']}, 大写={quality['has_uppercase']}, "
                      f"数字={quality['has_digit']}, 特殊={quality['has_special']}")
                print(f"   满足复杂度要求: {'是' if quality['meets_complexity'] else '否'}\n")
            else:
                print(f"{i}. {s}")

    except ValueError as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()
