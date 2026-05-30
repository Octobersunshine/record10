DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DIGIT_TO_VALUE = {ch: i for i, ch in enumerate(DIGITS)}


def to_decimal(num_str, from_base, with_steps=False):
    steps = []
    num_str = num_str.strip()
    negative = num_str.startswith("-")
    if negative:
        num_str = num_str[1:]

    if "." in num_str:
        int_part, frac_part = num_str.split(".", 1)
    else:
        int_part, frac_part = num_str, ""

    if not int_part and not frac_part:
        raise ValueError("输入不能为空")

    int_result = 0
    if int_part:
        steps.append({"type": "int_start", "value": int_part, "base": from_base})
        for i, ch in enumerate(int_part):
            value = DIGIT_TO_VALUE[ch.upper() if from_base <= 36 else ch]
            if value >= from_base:
                raise ValueError(f"数字 '{ch}' 不符合 {from_base} 进制")
            old_result = int_result
            int_result = int_result * from_base + value
            steps.append({
                "type": "int_step",
                "digit": ch,
                "digit_value": value,
                "formula": f"{old_result} × {from_base} + {value}",
                "result": int_result
            })
        steps.append({"type": "int_end", "result": int_result})

    frac_value = 0.0
    if frac_part:
        steps.append({"type": "frac_start", "value": frac_part, "base": from_base})
        for i, ch in enumerate(frac_part):
            value = DIGIT_TO_VALUE[ch.upper() if from_base <= 36 else ch]
            if value >= from_base:
                raise ValueError(f"数字 '{ch}' 不符合 {from_base} 进制")
            weight = from_base ** -(i + 1)
            contribution = value * weight
            old_frac = frac_value
            frac_value += contribution
            steps.append({
                "type": "frac_step",
                "digit": ch,
                "digit_value": value,
                "position": i + 1,
                "weight": weight,
                "formula": f"{old_frac} + {value} × {weight:.10f}",
                "result": frac_value
            })
        steps.append({"type": "frac_end", "result": frac_value})

    decimal_val = int_result + frac_value
    if negative:
        decimal_val = -decimal_val
        steps.append({"type": "apply_negative", "result": decimal_val})

    if with_steps:
        return {"value": decimal_val, "steps": steps}
    return decimal_val


def from_decimal(decimal_val, to_base, precision=10, with_steps=False):
    steps = []
    if decimal_val == 0:
        result = "0" if precision == 0 else "0." + "0" * precision
        if with_steps:
            return {"value": result, "steps": [{"type": "zero", "result": result}]}
        return result

    negative = decimal_val < 0
    decimal_val = abs(decimal_val)

    if negative:
        steps.append({"type": "handle_negative", "note": "取绝对值处理后再加负号"})

    int_val = int(decimal_val)
    frac_val = decimal_val - int_val

    int_result = []
    if int_val == 0:
        int_result.append("0")
        steps.append({"type": "int_zero", "note": "整数部分为0"})
    else:
        steps.append({"type": "int_start", "value": int_val, "base": to_base})
        temp = int_val
        step_num = 1
        while temp > 0:
            remainder = temp % to_base
            quotient = temp // to_base
            digit = DIGITS[remainder]
            int_result.append(digit)
            steps.append({
                "type": "int_step",
                "step": step_num,
                "dividend": temp,
                "quotient": quotient,
                "remainder": remainder,
                "digit": digit,
                "formula": f"{temp} ÷ {to_base} = {quotient} 余 {remainder} → '{digit}'"
            })
            temp = quotient
            step_num += 1
    int_str = "".join(reversed(int_result))
    steps.append({"type": "int_end", "digits": int_result[::-1], "result": int_str})

    if precision > 0:
        frac_result = []
        steps.append({"type": "frac_start", "value": frac_val, "base": to_base, "precision": precision})
        temp_frac = frac_val
        for i in range(precision):
            temp_frac *= to_base
            digit = int(temp_frac)
            digit_char = DIGITS[digit]
            frac_result.append(digit_char)
            steps.append({
                "type": "frac_step",
                "step": i + 1,
                "multiplied": temp_frac,
                "digit_value": digit,
                "digit": digit_char,
                "formula": f"{temp_frac / to_base:.10f} × {to_base} = {temp_frac:.10f} → 取整 {digit} → '{digit_char}'"
            })
            temp_frac -= digit
        frac_str = "".join(frac_result)
        full_str = f"{int_str}.{frac_str}"
        steps.append({"type": "frac_end", "digits": frac_result, "result": frac_str})
    else:
        full_str = int_str

    if negative:
        full_str = "-" + full_str
        steps.append({"type": "apply_negative", "result": full_str})

    if with_steps:
        return {"value": full_str, "steps": steps}
    return full_str


def convert(num_str, from_base, to_base, precision=None, with_steps=False):
    if not (2 <= from_base <= 62) or not (2 <= to_base <= 62):
        raise ValueError("进制必须在 2-62 之间")
    if precision is not None and precision < 0:
        raise ValueError("精度不能为负数")
    if not num_str or num_str in ("-", ".", "-."):
        raise ValueError("输入不能为空")

    has_dot = "." in num_str

    to_dec_result = to_decimal(num_str, from_base, with_steps=True)
    decimal_val = to_dec_result["value"]
    to_dec_steps = to_dec_result["steps"]

    if precision is None:
        default_precision = 10
        effective_precision = default_precision if has_dot else 0
    else:
        effective_precision = precision

    from_dec_result = from_decimal(decimal_val, to_base, effective_precision, with_steps=True)
    result = from_dec_result["value"]
    from_dec_steps = from_dec_result["steps"]

    if with_steps:
        return {
            "input": num_str,
            "from_base": from_base,
            "to_base": to_base,
            "precision": effective_precision,
            "result": result,
            "steps": {
                "to_decimal": to_dec_steps,
                "from_decimal": from_dec_steps
            }
        }
    return result


def batch_convert(items, with_steps=False):
    """
    批量转换多个数字
    
    items: 列表，每个元素是字典或元组，包含：
           - num_str: 数字字符串
           - from_base: 原进制
           - to_base: 目标进制
           - precision: 可选，精度
    
    返回: 列表，每个元素是转换结果（包含步骤如果 with_steps=True）
    """
    results = []
    for item in items:
        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                num_str, from_base, to_base = item
                precision = None
            else:
                num_str, from_base, to_base, precision = item
        else:
            num_str = item["num_str"]
            from_base = item["from_base"]
            to_base = item["to_base"]
            precision = item.get("precision")
        
        result = convert(num_str, from_base, to_base, precision, with_steps)
        results.append(result)
    return results


def format_steps(result, indent=0):
    """格式化打印转换步骤"""
    prefix = "  " * indent
    output = []
    output.append(f"{prefix}输入: {result['input']} (进制 {result['from_base']})")
    output.append(f"{prefix}目标: 进制 {result['to_base']}")
    output.append(f"{prefix}结果: {result['result']}")
    output.append("")
    
    output.append(f"{prefix}=== 步骤1: 转换为十进制 ===")
    for step in result["steps"]["to_decimal"]:
        if step["type"] == "int_start":
            output.append(f"{prefix}  整数部分 '{step['value']}' 转换开始:")
        elif step["type"] == "int_step":
            output.append(f"{prefix}    {step['formula']} = {step['result']}")
        elif step["type"] == "int_end":
            output.append(f"{prefix}    整数部分结果: {step['result']}")
        elif step["type"] == "frac_start":
            output.append(f"{prefix}  小数部分 '{step['value']}' 转换开始:")
        elif step["type"] == "frac_step":
            output.append(f"{prefix}    第{step['position']}位: {step['formula']} = {step['result']:.10f}")
        elif step["type"] == "frac_end":
            output.append(f"{prefix}    小数部分结果: {step['result']:.10f}")
        elif step["type"] == "apply_negative":
            output.append(f"{prefix}  应用负号: {step['result']}")
    
    output.append("")
    output.append(f"{prefix}=== 步骤2: 十进制转换为目标进制 ===")
    for step in result["steps"]["from_decimal"]:
        if step["type"] == "handle_negative":
            output.append(f"{prefix}  {step['note']}")
        elif step["type"] == "int_start":
            output.append(f"{prefix}  整数部分 {step['value']} 转换开始 (除基取余):")
        elif step["type"] == "int_step":
            output.append(f"{prefix}    步骤{step['step']}: {step['formula']}")
        elif step["type"] == "int_end":
            output.append(f"{prefix}    逆序排列: {step['digits']} → {step['result']}")
        elif step["type"] == "frac_start":
            output.append(f"{prefix}  小数部分 {step['value']:.10f} 转换开始 (乘基取整, 精度{step['precision']}):")
        elif step["type"] == "frac_step":
            output.append(f"{prefix}    步骤{step['step']}: {step['formula']}")
        elif step["type"] == "frac_end":
            output.append(f"{prefix}    小数部分: {step['result']}")
        elif step["type"] == "apply_negative":
            output.append(f"{prefix}  应用负号: {step['result']}")
        elif step["type"] == "int_zero":
            output.append(f"{prefix}  {step['note']}")
        elif step["type"] == "zero":
            output.append(f"{prefix}  值为0: {step['result']}")
    
    return "\n".join(output)


if __name__ == "__main__":
    print("=== 62进制测试 ===")
    print(convert("123456", 10, 62))
    print(convert("Z", 62, 10))
    print(convert("a", 62, 10))
    print(convert("10", 62, 10))
    print(convert("zz", 62, 10))
    
    print("\n=== 负数测试 ===")
    print(convert("-1a", 16, 10))
    print(convert("-1010", 2, 16))
    
    print("\n=== 小数测试 ===")
    print(convert("10.101", 2, 10))
    print(convert("3.14", 10, 2, precision=5))
    
    print("\n=== 批量转换测试 ===")
    batch_items = [
        ("1010", 2, 10),
        ("FF", 16, 2),
        ("255", 10, 16),
        ("123456", 10, 62),
    ]
    batch_results = batch_convert(batch_items)
    for i, res in enumerate(batch_results):
        print(f"  {i+1}. {batch_items[i][0]} ({batch_items[i][1]}) → {res} ({batch_items[i][2]})")
    
    print("\n=== 详细步骤测试 (1010 二进制转十进制) ===")
    result_with_steps = convert("1010", 2, 10, with_steps=True)
    print(format_steps(result_with_steps))
    
    print("\n" + "="*60)
    print("=== 详细步骤测试 (255 十进制转十六进制) ===")
    result_with_steps2 = convert("255", 10, 16, with_steps=True)
    print(format_steps(result_with_steps2))
