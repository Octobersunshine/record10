import base64
import os
import re
import json
from io import BytesIO
from PIL import Image

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
COMPRESSIBLE_FORMATS = {'JPEG', 'JPG', 'WEBP'}

DATA_URL_PATTERN = re.compile(
    r'^data:image/[a-zA-Z0-9+\-.]+;base64,',
    re.IGNORECASE
)

def get_image_format(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的图片格式: {ext}。支持的格式: {', '.join(SUPPORTED_FORMATS)}")
    return ext[1:] if ext != '.jpg' else 'jpeg'

def strip_data_url_prefix(base64_str):
    if not base64_str:
        raise ValueError("Base64字符串为空")
    
    cleaned = base64_str.strip()
    
    match = DATA_URL_PATTERN.match(cleaned)
    if match:
        prefix = match.group(0)
        cleaned = cleaned[len(prefix):]
        if not cleaned:
            raise ValueError(
                f"检测到Data URL前缀 ({prefix.strip(',')})，但其后没有Base64数据内容"
            )
        return cleaned, True
    
    if 'data:' in cleaned.lower() and 'base64' in cleaned.lower():
        parts = cleaned.split(',', 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip(), True
    
    return cleaned, False

def validate_base64_format(base64_str):
    if not base64_str:
        raise ValueError("Base64字符串不能为空")
    
    cleaned = base64_str.strip()
    
    if len(cleaned) < 4:
        raise ValueError(
            "Base64字符串过短，有效Base64数据至少需要4个字符"
        )
    
    if len(cleaned) % 4 != 0:
        padding_needed = 4 - (len(cleaned) % 4)
        raise ValueError(
            f"Base64格式错误：长度不是4的倍数。"
            f"当前长度: {len(cleaned)}，需要补充 {padding_needed} 个 '=' 填充字符"
        )
    
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
    invalid_chars = set(cleaned) - valid_chars
    if invalid_chars:
        invalid_preview = ''.join(sorted(invalid_chars))[:10]
        raise ValueError(
            f"Base64字符串包含无效字符: {invalid_preview}...\n"
            f"有效Base64字符只能包含: A-Z, a-z, 0-9, +, /, =\n"
            f"请检查字符串中是否包含换行符、空格或其他特殊字符"
        )

def image_to_base64(image_path, output_file=None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    img_format = get_image_format(image_path)
    
    with open(image_path, 'rb') as image_file:
        base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(base64_str)
        print(f"Base64字符串已保存到: {output_file}")
    
    return base64_str

def image_to_base64_with_data_uri(image_path, output_file=None):
    base64_str = image_to_base64(image_path)
    img_format = get_image_format(image_path)
    data_uri = f"data:image/{img_format};base64,{base64_str}"
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(data_uri)
        print(f"Data URI已保存到: {output_file}")
    
    return data_uri

def _process_base64_string(base64_str):
    try:
        cleaned, had_prefix = strip_data_url_prefix(base64_str)
        validate_base64_format(cleaned)
        return cleaned, had_prefix
    except ValueError as e:
        raise e

def base64_to_image(base64_str, output_path):
    try:
        cleaned, had_prefix = _process_base64_string(base64_str)
    except ValueError as e:
        raise
    
    try:
        image_data = base64.b64decode(cleaned)
    except (base64.binascii.Error, ValueError) as e:
        prefix_info = "（已去除Data URL前缀）" if had_prefix else ""
        raise ValueError(
            f"Base64解码失败{prefix_info}: {str(e)}\n"
            f"提示：请确保Base64字符串完整且格式正确。"
            f"如果字符串包含换行符，请先去除所有空白字符。"
        )
    
    try:
        image = Image.open(BytesIO(image_data))
        image.verify()
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        prefix_info = "（已去除Data URL前缀）" if had_prefix else ""
        raise ValueError(
            f"解码后的数据不是有效的图片{prefix_info}\n"
            f"错误详情: {str(e)}\n"
            f"提示：\n"
            f"  1. 请确认Base64字符串确实对应图片数据\n"
            f"  2. 支持的图片格式: {', '.join(sorted(SUPPORTED_FORMATS))}\n"
            f"  3. 检查字符串是否被截断或损坏"
        )
    
    try:
        image.save(output_path)
        print(f"图片已保存到: {output_path}")
        return output_path
    except Exception as e:
        raise ValueError(
            f"图片保存失败: {str(e)}\n"
            f"提示：请检查输出路径是否有写入权限，以及文件扩展名是否正确"
        )

def base64_file_to_image(base64_file, output_path):
    if not os.path.exists(base64_file):
        raise FileNotFoundError(f"Base64文件不存在: {base64_file}")
    
    try:
        with open(base64_file, 'r', encoding='utf-8') as f:
            base64_str = f.read()
    except UnicodeDecodeError:
        raise ValueError(
            f"文件读取失败: {base64_file}\n"
            f"提示：Base64文件应为纯文本格式，请检查文件编码"
        )
    
    if not base64_str.strip():
        raise ValueError(f"Base64文件为空: {base64_file}")
    
    return base64_to_image(base64_str, output_path)

def get_image_from_base64(base64_str):
    try:
        cleaned, had_prefix = _process_base64_string(base64_str)
    except ValueError as e:
        raise
    
    try:
        image_data = base64.b64decode(cleaned)
    except (base64.binascii.Error, ValueError) as e:
        prefix_info = "（已去除Data URL前缀）" if had_prefix else ""
        raise ValueError(
            f"Base64解码失败{prefix_info}: {str(e)}\n"
            f"提示：请确保Base64字符串完整且格式正确。"
        )
    
    try:
        image = Image.open(BytesIO(image_data))
        image.load()
        return image
    except Exception as e:
        prefix_info = "（已去除Data URL前缀）" if had_prefix else ""
        raise ValueError(
            f"解码后的数据不是有效的图片{prefix_info}\n"
            f"错误详情: {str(e)}\n"
            f"提示：请确认Base64字符串对应支持的图片格式"
        )

def get_image_info_from_base64(base64_str):
    cleaned, had_prefix = strip_data_url_prefix(base64_str)
    validate_base64_format(cleaned)
    
    base64_length = len(cleaned)
    estimated_data_size = int(base64_length * 3 / 4)
    
    try:
        image_data = base64.b64decode(cleaned)
        actual_data_size = len(image_data)
    except Exception:
        actual_data_size = estimated_data_size
    
    try:
        with Image.open(BytesIO(image_data)) as img:
            width, height = img.size
            img_format = img.format
            mode = img.mode
            
            info = {
                "width": width,
                "height": height,
                "dimensions": f"{width}x{height}",
                "aspect_ratio": round(width / height, 4) if height > 0 else 0,
                "format": img_format,
                "mode": mode,
                "data_size_bytes": actual_data_size,
                "data_size_kb": round(actual_data_size / 1024, 2),
                "base64_length": base64_length,
                "base64_size_kb": round(base64_length / 1024, 2),
                "has_data_url_prefix": had_prefix,
                "megapixels": round((width * height) / 1_000_000, 4)
            }
            
            return info
            
    except Exception as e:
        prefix_info = "（已去除Data URL前缀）" if had_prefix else ""
        raise ValueError(
            f"无法解析图片信息{prefix_info}\n"
            f"错误详情: {str(e)}\n"
            f"已检测的Base64信息:\n"
            f"  - 长度: {base64_length} 字符\n"
            f"  - 预估数据大小: {estimated_data_size} 字节"
        )

def image_to_base64_compressed(image_path, quality=85, output_file=None, data_uri=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    img_format = get_image_format(image_path).upper()
    
    if img_format not in COMPRESSIBLE_FORMATS:
        raise ValueError(
            f"不支持压缩的图片格式: {img_format}\n"
            f"支持压缩的格式: {', '.join(sorted(COMPRESSIBLE_FORMATS))}\n"
            f"提示: PNG、GIF、BMP、TIFF 格式请使用 image_to_base64 函数"
        )
    
    if not isinstance(quality, int) or quality < 1 or quality > 100:
        raise ValueError(f"质量参数必须是1-100之间的整数，当前值: {quality}")
    
    original_size = os.path.getsize(image_path)
    
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA', 'P') and img_format in ('JPEG', 'JPG'):
            img = img.convert('RGB')
        
        buffer = BytesIO()
        save_format = 'JPEG' if img_format in ('JPEG', 'JPG') else img_format
        img.save(buffer, format=save_format, quality=quality, optimize=True)
        compressed_data = buffer.getvalue()
    
    compressed_size = len(compressed_data)
    compression_ratio = round((1 - compressed_size / original_size) * 100, 2) if original_size > 0 else 0
    
    base64_str = base64.b64encode(compressed_data).decode('utf-8')
    
    if data_uri:
        base64_str = f"data:image/{img_format.lower()};base64,{base64_str}"
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(base64_str)
        print(f"压缩后的Base64已保存到: {output_file}")
    
    result = {
        "base64": base64_str,
        "original_size_bytes": original_size,
        "original_size_kb": round(original_size / 1024, 2),
        "compressed_size_bytes": compressed_size,
        "compressed_size_kb": round(compressed_size / 1024, 2),
        "base64_length": len(base64_str),
        "base64_size_kb": round(len(base64_str) / 1024, 2),
        "quality": quality,
        "compression_ratio_percent": compression_ratio
    }
    
    return result

def batch_image_to_base64(image_paths, output_dir=None, data_uri=False, quality=None):
    if not image_paths:
        raise ValueError("图片路径列表不能为空")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    failed = []
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] 处理: {image_path}")
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"文件不存在: {image_path}")
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_file = None
            
            if output_dir:
                suffix = "_data_uri.txt" if data_uri else ".txt"
                output_file = os.path.join(output_dir, f"{base_name}{suffix}")
            
            if quality is not None:
                try:
                    result = image_to_base64_compressed(image_path, quality=quality, output_file=output_file, data_uri=data_uri)
                    result["compressed"] = True
                except ValueError as e:
                    if "不支持压缩" in str(e):
                        base64_str = image_to_base64_with_data_uri(image_path, output_file) if data_uri else image_to_base64(image_path, output_file)
                        file_size = os.path.getsize(image_path)
                        result = {
                            "base64": base64_str,
                            "original_size_bytes": file_size,
                            "original_size_kb": round(file_size / 1024, 2),
                            "compressed_size_bytes": file_size,
                            "compressed_size_kb": round(file_size / 1024, 2),
                            "base64_length": len(base64_str),
                            "base64_size_kb": round(len(base64_str) / 1024, 2),
                            "quality": None,
                            "compression_ratio_percent": 0,
                            "compressed": False,
                            "note": "该格式不支持压缩，已使用原始质量"
                        }
                    else:
                        raise
            else:
                base64_str = image_to_base64_with_data_uri(image_path, output_file) if data_uri else image_to_base64(image_path, output_file)
                file_size = os.path.getsize(image_path)
                result = {
                    "base64": base64_str,
                    "original_size_bytes": file_size,
                    "original_size_kb": round(file_size / 1024, 2),
                    "compressed": False
                }
            
            result["source"] = image_path
            result["success"] = True
            results.append(result)
            
        except Exception as e:
            error_info = {
                "source": image_path,
                "success": False,
                "error": str(e)
            }
            failed.append(error_info)
            print(f"  ✗ 失败: {str(e)}")
    
    summary = {
        "total": len(image_paths),
        "success": len(results),
        "failed": len(failed),
        "results": results,
        "failed_items": failed
    }
    
    if output_dir:
        summary_file = os.path.join(output_dir, "batch_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n批量转换完成，汇总信息已保存到: {summary_file}")
    
    return summary

def batch_base64_to_image(base64_items, output_dir):
    if not base64_items:
        raise ValueError("Base64项目列表不能为空")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    failed = []
    
    for idx, item in enumerate(base64_items, 1):
        print(f"[{idx}/{len(base64_items)}] 处理...")
        
        try:
            if isinstance(item, dict):
                base64_str = item.get("base64") or item.get("data")
                output_name = item.get("name") or item.get("filename") or f"image_{idx}"
                if not base64_str:
                    raise ValueError("字典中未找到 'base64' 或 'data' 字段")
            elif isinstance(item, str):
                base64_str = item
                output_name = f"image_{idx}"
            else:
                raise ValueError(f"不支持的项目类型: {type(item)}")
            
            info = get_image_info_from_base64(base64_str)
            ext = info["format"].lower()
            ext = "jpg" if ext == "jpeg" else ext
            output_path = os.path.join(output_dir, f"{output_name}.{ext}")
            
            base64_to_image(base64_str, output_path)
            
            result = {
                "name": output_name,
                "output_path": output_path,
                "format": info["format"],
                "dimensions": info["dimensions"],
                "success": True
            }
            results.append(result)
            
        except Exception as e:
            error_info = {
                "item_index": idx,
                "success": False,
                "error": str(e)
            }
            failed.append(error_info)
            print(f"  ✗ 失败: {str(e)}")
    
    summary = {
        "total": len(base64_items),
        "success": len(results),
        "failed": len(failed),
        "results": results,
        "failed_items": failed
    }
    
    summary_file = os.path.join(output_dir, "batch_decode_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n批量解码完成，汇总信息已保存到: {summary_file}")
    
    return summary

def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='图片与Base64双向转换工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    encode_parser = subparsers.add_parser('encode', help='图片转Base64')
    encode_parser.add_argument('image_path', help='输入图片路径')
    encode_parser.add_argument('-o', '--output', help='输出Base64文件路径（可选）')
    encode_parser.add_argument('--data-uri', action='store_true', help='输出Data URI格式')
    encode_parser.add_argument('-q', '--quality', type=int, help='压缩质量 1-100（仅JPEG/WebP）')
    
    decode_parser = subparsers.add_parser('decode', help='Base64转图片')
    decode_parser.add_argument('base64_input', help='Base64字符串或包含Base64的文件路径')
    decode_parser.add_argument('output_path', help='输出图片路径')
    decode_parser.add_argument('--from-file', action='store_true', help='从文件读取Base64')
    
    info_parser = subparsers.add_parser('info', help='提取Base64图片信息')
    info_parser.add_argument('base64_input', help='Base64字符串或包含Base64的文件路径')
    info_parser.add_argument('--from-file', action='store_true', help='从文件读取Base64')
    
    batch_encode_parser = subparsers.add_parser('batch-encode', help='批量图片转Base64')
    batch_encode_parser.add_argument('inputs', nargs='+', help='图片路径或通配符（如 *.jpg）')
    batch_encode_parser.add_argument('-o', '--output-dir', required=True, help='输出目录')
    batch_encode_parser.add_argument('--data-uri', action='store_true', help='输出Data URI格式')
    batch_encode_parser.add_argument('-q', '--quality', type=int, help='压缩质量 1-100')
    
    batch_decode_parser = subparsers.add_parser('batch-decode', help='批量Base64转图片')
    batch_decode_parser.add_argument('inputs', nargs='+', help='Base64文件路径或通配符')
    batch_decode_parser.add_argument('-o', '--output-dir', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    if args.command == 'encode':
        if args.quality is not None:
            result = image_to_base64_compressed(
                args.image_path, 
                quality=args.quality, 
                output_file=args.output,
                data_uri=args.data_uri
            )
            if not args.output:
                print(result["base64"])
                print(f"\n压缩信息: 原始 {result['original_size_kb']}KB → "
                      f"压缩后 {result['compressed_size_kb']}KB "
                      f"(质量 {result['quality']}, 压缩率 {result['compression_ratio_percent']}%)")
        elif args.data_uri:
            result = image_to_base64_with_data_uri(args.image_path, args.output)
            if not args.output:
                print(result)
        else:
            result = image_to_base64(args.image_path, args.output)
            if not args.output:
                print(result)
    
    elif args.command == 'decode':
        if args.from_file:
            base64_file_to_image(args.base64_input, args.output_path)
        else:
            base64_to_image(args.base64_input, args.output_path)
    
    elif args.command == 'info':
        if args.from_file:
            with open(args.base64_input, 'r', encoding='utf-8') as f:
                base64_str = f.read()
        else:
            base64_str = args.base64_input
        
        info = get_image_info_from_base64(base64_str)
        print("\n" + "=" * 50)
        print("图片信息")
        print("=" * 50)
        for key, value in info.items():
            print(f"{key:25s}: {value}")
        print("=" * 50 + "\n")
    
    elif args.command == 'batch-encode':
        image_paths = []
        for pattern in args.inputs:
            matched = glob.glob(pattern)
            if matched:
                image_paths.extend(matched)
            elif os.path.exists(pattern):
                image_paths.append(pattern)
        
        if not image_paths:
            print("未找到任何匹配的图片文件")
            return
        
        print(f"找到 {len(image_paths)} 个图片文件\n")
        summary = batch_image_to_base64(
            image_paths, 
            output_dir=args.output_dir,
            data_uri=args.data_uri,
            quality=args.quality
        )
        print(f"\n成功: {summary['success']}/{summary['total']}, 失败: {summary['failed']}")
    
    elif args.command == 'batch-decode':
        base64_items = []
        for pattern in args.inputs:
            matched = glob.glob(pattern)
            for file_path in matched:
                with open(file_path, 'r', encoding='utf-8') as f:
                    base64_str = f.read().strip()
                name = os.path.splitext(os.path.basename(file_path))[0]
                base64_items.append({"base64": base64_str, "name": name})
        
        if not base64_items:
            print("未找到任何匹配的Base64文件")
            return
        
        print(f"找到 {len(base64_items)} 个Base64文件\n")
        summary = batch_base64_to_image(base64_items, output_dir=args.output_dir)
        print(f"\n成功: {summary['success']}/{summary['total']}, 失败: {summary['failed']}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
