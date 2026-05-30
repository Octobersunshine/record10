import zipfile
import base64
import os
import re
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Dict, Union, Optional, Tuple

try:
    import pyzipper
    _HAS_PYZIPPER = True
except ImportError:
    _HAS_PYZIPPER = False

MAX_SINGLE_FILE_SIZE = 50 * 1024 * 1024
MAX_TOTAL_ZIP_SIZE = 500 * 1024 * 1024


@dataclass
class FileStat:
    filename: str
    original_size: int
    compressed_size: int

    @property
    def compression_ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size

    @property
    def saved_percent(self) -> float:
        return (1.0 - self.compression_ratio) * 100


@dataclass
class CompressResult:
    data: str
    output_mode: str
    original_size: int
    compressed_size: int
    file_count: int
    files: List[FileStat] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size

    @property
    def saved_percent(self) -> float:
        return (1.0 - self.compression_ratio) * 100

    def summary(self) -> str:
        lines = [
            f"压缩统计:",
            f"  文件数量:       {self.file_count}",
            f"  原始总大小:     {self.original_size:,} bytes ({_fmt_size(self.original_size)})",
            f"  压缩后总大小:   {self.compressed_size:,} bytes ({_fmt_size(self.compressed_size)})",
            f"  压缩率:         {self.compression_ratio:.2%}",
            f"  节省空间:       {self.saved_percent:.1f}%",
        ]
        if self.files:
            lines.append("  各文件明细:")
            for fs in self.files:
                lines.append(
                    f"    {fs.filename}: "
                    f"{_fmt_size(fs.original_size)} -> {_fmt_size(fs.compressed_size)} "
                    f"(节省 {fs.saved_percent:.1f}%)"
                )
        return "\n".join(lines)


@dataclass
class ExtractResult:
    files: Dict[str, bytes]
    total_size: int
    file_count: int


@dataclass
class VolumeInfo:
    volume_paths: List[str]
    volume_size: int
    total_volumes: int
    total_size: int


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"


def sanitize_filename(filename: str) -> str:
    if not filename or not filename.strip():
        raise ValueError("文件名不能为空")

    filename = filename.replace("\\", "/")

    if re.match(r'^[A-Za-z]:', filename):
        filename = filename.split(":", 1)[1]

    filename = filename.lstrip("/")

    segments = filename.split("/")
    safe_segments = []
    for seg in segments:
        if seg == ".." or seg == ".":
            continue
        if seg == "":
            continue
        safe_segments.append(seg)

    if not safe_segments:
        raise ValueError(f"文件名 '{filename}' 经安全过滤后为空（可能全为路径遍历组件）")

    result = "/".join(safe_segments)

    if result.startswith("/") or result.startswith("\\"):
        result = result.lstrip("/\\")

    return result


def _check_single_file_size(size: int, filepath: str = ""):
    if size > MAX_SINGLE_FILE_SIZE:
        label = filepath if filepath else "文件"
        raise ValueError(
            f"{label} 大小 {size} 字节超过单文件限制 "
            f"{MAX_SINGLE_FILE_SIZE} 字节（{MAX_SINGLE_FILE_SIZE // (1024*1024)}MB）"
        )


def _check_total_size(total: int):
    if total > MAX_TOTAL_ZIP_SIZE:
        raise ValueError(
            f"总大小 {total} 字节超过压缩包限制 "
            f"{MAX_TOTAL_ZIP_SIZE} 字节（{MAX_TOTAL_ZIP_SIZE // (1024*1024)}MB）"
        )


def _matches_extension(filename: str, extensions: Optional[List[str]]) -> bool:
    if not extensions:
        return True
    _, ext = os.path.splitext(filename)
    return ext.lower() in [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions]


def _create_zip_writer(target, compression, password=None):
    if password:
        if not _HAS_PYZIPPER:
            raise ImportError(
                "加密ZIP需要 pyzipper 库，请运行: pip install pyzipper"
            )
        zf = pyzipper.AESZipFile(
            target, "w",
            compression=compression,
        )
        zf.setpassword(password.encode("utf-8") if isinstance(password, str) else password)
        zf.setencryption(pyzipper.WZ_AES, nbits=256)
        return zf
    else:
        return zipfile.ZipFile(target, "w", compression)


def _open_zip_reader(source, password=None):
    pwd = None
    if password:
        pwd = password.encode("utf-8") if isinstance(password, str) else password

    if _HAS_PYZIPPER:
        try:
            zf = pyzipper.AESZipFile(source, "r")
            if pwd:
                zf.setpassword(pwd)
            return zf
        except Exception:
            source.seek(0)
            zf = zipfile.ZipFile(source, "r")
            if pwd:
                zf.setpassword(pwd)
            return zf
    else:
        zf = zipfile.ZipFile(source, "r")
        if pwd:
            zf.setpassword(pwd)
        return zf


def compress_files(
    file_paths: List[str],
    output_mode: str = "base64",
    output_path: Optional[str] = None,
    compression: int = zipfile.ZIP_DEFLATED,
    password: Optional[str] = None,
    extensions: Optional[List[str]] = None,
) -> CompressResult:
    """
    压缩多个文件为ZIP包

    Args:
        file_paths: 要压缩的文件路径列表
        output_mode: 输出模式，"base64" / "tempfile" / "file"
        output_path: 当 output_mode 为 "file" 时，指定保存路径
        compression: 压缩算法，默认 ZIP_DEFLATED
        password: 加密密码（AES-256），None 表示不加密
        extensions: 仅压缩指定扩展名的文件，如 [".txt", ".csv"]，None 表示不过滤

    Returns:
        CompressResult 包含压缩数据和统计信息
    """
    filtered_paths = []
    for fp in file_paths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"文件不存在: {fp}")
        if not os.path.isfile(fp):
            raise ValueError(f"路径不是文件: {fp}")
        if not _matches_extension(os.path.basename(fp), extensions):
            continue
        filtered_paths.append(fp)

    if not filtered_paths:
        raise ValueError("没有符合条件（扩展名过滤）的文件可压缩")

    total_original = 0
    file_sizes = {}
    for fp in filtered_paths:
        size = os.path.getsize(fp)
        _check_single_file_size(size, fp)
        total_original += size
        file_sizes[fp] = size

    _check_total_size(total_original)

    if output_mode == "base64":
        buffer = BytesIO()
        with _create_zip_writer(buffer, compression, password) as zf:
            for fp in filtered_paths:
                arcname = sanitize_filename(os.path.basename(fp))
                zf.write(fp, arcname)
        buffer.seek(0)
        zip_bytes = buffer.read()
        b64_data = base64.b64encode(zip_bytes).decode("utf-8")
    elif output_mode == "tempfile":
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            temp_path = tmp.name
        with _create_zip_writer(temp_path, compression, password) as zf:
            for fp in filtered_paths:
                arcname = sanitize_filename(os.path.basename(fp))
                zf.write(fp, arcname)
        zip_bytes = open(temp_path, "rb").read()
        b64_data = temp_path
    elif output_mode == "file":
        if not output_path:
            raise ValueError("output_mode 为 'file' 时必须指定 output_path")
        with _create_zip_writer(output_path, compression, password) as zf:
            for fp in filtered_paths:
                arcname = sanitize_filename(os.path.basename(fp))
                zf.write(fp, arcname)
        zip_bytes = open(output_path, "rb").read()
        b64_data = output_path
    else:
        raise ValueError(f"无效的 output_mode: {output_mode}，可选值: base64, tempfile, file")

    compressed_size = len(zip_bytes)
    file_stats = []
    for fp in filtered_paths:
        arcname = sanitize_filename(os.path.basename(fp))
        orig = file_sizes[fp]
        file_stats.append(FileStat(
            filename=arcname,
            original_size=orig,
            compressed_size=0,
        ))

    with _open_zip_reader(BytesIO(zip_bytes), password) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            for stat in file_stats:
                if stat.filename == info.filename:
                    stat.compressed_size = info.compress_size
                    break

    return CompressResult(
        data=b64_data,
        output_mode=output_mode,
        original_size=total_original,
        compressed_size=compressed_size,
        file_count=len(filtered_paths),
        files=file_stats,
    )


def compress_bytes(
    files_data: List[Tuple[str, bytes]],
    output_mode: str = "base64",
    output_path: Optional[str] = None,
    compression: int = zipfile.ZIP_DEFLATED,
    password: Optional[str] = None,
) -> CompressResult:
    """
    压缩内存中的字节数据为ZIP包

    Args:
        files_data: 文件数据列表，每个元素为 (文件名, 字节内容) 的元组
        output_mode: 输出模式，"base64" / "tempfile" / "file"
        output_path: 当 output_mode 为 "file" 时，指定保存路径
        compression: 压缩算法，默认 ZIP_DEFLATED
        password: 加密密码（AES-256），None 表示不加密

    Returns:
        CompressResult 包含压缩数据和统计信息
    """
    total_original = 0
    file_stats = []
    for filename, content in files_data:
        safe_name = sanitize_filename(filename)
        _check_single_file_size(len(content), safe_name)
        total_original += len(content)
        file_stats.append(FileStat(
            filename=safe_name,
            original_size=len(content),
            compressed_size=0,
        ))

    _check_total_size(total_original)

    if output_mode == "base64":
        buffer = BytesIO()
        with _create_zip_writer(buffer, compression, password) as zf:
            for filename, content in files_data:
                zf.writestr(sanitize_filename(filename), content)
        buffer.seek(0)
        zip_bytes = buffer.read()
        b64_data = base64.b64encode(zip_bytes).decode("utf-8")
    elif output_mode == "tempfile":
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            temp_path = tmp.name
        with _create_zip_writer(temp_path, compression, password) as zf:
            for filename, content in files_data:
                zf.writestr(sanitize_filename(filename), content)
        zip_bytes = open(temp_path, "rb").read()
        b64_data = temp_path
    elif output_mode == "file":
        if not output_path:
            raise ValueError("output_mode 为 'file' 时必须指定 output_path")
        with _create_zip_writer(output_path, compression, password) as zf:
            for filename, content in files_data:
                zf.writestr(sanitize_filename(filename), content)
        zip_bytes = open(output_path, "rb").read()
        b64_data = output_path
    else:
        raise ValueError(f"无效的 output_mode: {output_mode}，可选值: base64, tempfile, file")

    compressed_size = len(zip_bytes)

    with _open_zip_reader(BytesIO(zip_bytes), password) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            for stat in file_stats:
                if stat.filename == info.filename:
                    stat.compressed_size = info.compress_size
                    break

    return CompressResult(
        data=b64_data,
        output_mode=output_mode,
        original_size=total_original,
        compressed_size=compressed_size,
        file_count=len(files_data),
        files=file_stats,
    )


def _read_zip_source(
    zip_source: Union[str, bytes],
    source_type: str,
) -> bytes:
    if source_type == "file":
        if not os.path.exists(zip_source):
            raise FileNotFoundError(f"ZIP文件不存在: {zip_source}")
        with open(zip_source, "rb") as f:
            return f.read()
    elif source_type == "base64":
        if isinstance(zip_source, str):
            return base64.b64decode(zip_source)
        raise ValueError("source_type 为 'base64' 时，zip_source 必须是字符串")
    elif source_type == "bytes":
        if isinstance(zip_source, bytes):
            return zip_source
        raise ValueError("source_type 为 'bytes' 时，zip_source 必须是字节类型")
    else:
        raise ValueError(f"无效的 source_type: {source_type}，可选值: file, base64, bytes")


def extract_zip(
    zip_source: Union[str, bytes],
    source_type: str = "file",
    extract_dir: Optional[str] = None,
    password: Optional[str] = None,
) -> ExtractResult:
    """
    解压ZIP文件并返回文件列表和内容

    安全措施：
    - 过滤文件名中的路径遍历组件（../），防止 ZipSlip 攻击
    - 逐文件安全解压，确保目标路径在 extract_dir 内
    - 检查单文件大小限制

    Args:
        zip_source: ZIP来源
        source_type: 来源类型，"file" / "base64" / "bytes"
        extract_dir: 如果指定，同时安全解压到该目录
        password: 解压密码，None 表示无密码

    Returns:
        ExtractResult 包含文件内容和统计信息
    """
    zip_data = _read_zip_source(zip_source, source_type)
    pwd = password.encode("utf-8") if isinstance(password, str) else password

    result_files = {}
    with _open_zip_reader(BytesIO(zip_data), password) as zf:
        if extract_dir:
            extract_dir = os.path.realpath(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)

        for file_info in zf.infolist():
            if file_info.is_dir():
                continue

            safe_name = sanitize_filename(file_info.filename)

            if file_info.file_size > MAX_SINGLE_FILE_SIZE:
                raise ValueError(
                    f"解压文件 '{safe_name}' 大小 {file_info.file_size} 字节"
                    f"超过单文件限制 {MAX_SINGLE_FILE_SIZE} 字节（{MAX_SINGLE_FILE_SIZE // (1024*1024)}MB）"
                )

            with zf.open(file_info, pwd=pwd) as f:
                content = f.read()

            result_files[safe_name] = content

            if extract_dir:
                dest_path = os.path.realpath(os.path.join(extract_dir, safe_name))
                if not dest_path.startswith(extract_dir + os.sep) and dest_path != extract_dir:
                    raise ValueError(
                        f"路径遍历攻击检测：文件 '{file_info.filename}' 解压目标 "
                        f"'{dest_path}' 在目录 '{extract_dir}' 之外"
                    )
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as out_f:
                    out_f.write(content)

    total_size = sum(len(v) for v in result_files.values())
    return ExtractResult(
        files=result_files,
        total_size=total_size,
        file_count=len(result_files),
    )


def list_zip_contents(
    zip_source: Union[str, bytes],
    source_type: str = "file",
    password: Optional[str] = None,
) -> List[Dict]:
    """
    列出ZIP文件中的内容（不解压）

    Args:
        zip_source: ZIP来源
        source_type: 来源类型
        password: 密码

    Returns:
        列表，每个元素为文件信息字典
    """
    zip_data = _read_zip_source(zip_source, source_type)

    result = []
    with _open_zip_reader(BytesIO(zip_data), password) as zf:
        for file_info in zf.infolist():
            try:
                safe_name = sanitize_filename(file_info.filename)
            except ValueError:
                safe_name = "<FILTERED>"
            ratio = 0.0
            if file_info.file_size > 0:
                ratio = file_info.compress_size / file_info.file_size
            result.append({
                "filename": file_info.filename,
                "safe_filename": safe_name,
                "size": file_info.file_size,
                "compress_size": file_info.compress_size,
                "compression_ratio": round(ratio, 4),
                "saved_percent": round((1.0 - ratio) * 100, 1),
                "is_dir": file_info.is_dir(),
                "compress_type": file_info.compress_type,
                "crc": file_info.CRC,
                "encrypted": file_info.flag_bits & 0x1 != 0,
            })
    return result


def split_zip_to_volumes(
    zip_source: Union[str, bytes],
    source_type: str = "file",
    volume_size: int = 10 * 1024 * 1024,
    output_dir: Optional[str] = None,
    base_name: str = "archive",
) -> VolumeInfo:
    """
    将ZIP文件分卷为多个固定大小的文件

    分卷命名规则：archive.z01, archive.z02, ..., archive.zip（最后一个分卷）

    Args:
        zip_source: ZIP来源
        source_type: 来源类型
        volume_size: 每个分卷的大小（字节），默认 10MB
        output_dir: 输出目录，None 则使用临时目录
        base_name: 分卷文件基础名

    Returns:
        VolumeInfo 包含分卷文件路径和统计信息
    """
    if volume_size <= 0:
        raise ValueError("分卷大小必须大于 0")

    zip_data = _read_zip_source(zip_source, source_type)
    total_size = len(zip_data)

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    os.makedirs(output_dir, exist_ok=True)

    if total_size <= volume_size:
        single_path = os.path.join(output_dir, f"{base_name}.zip")
        with open(single_path, "wb") as f:
            f.write(zip_data)
        return VolumeInfo(
            volume_paths=[single_path],
            volume_size=volume_size,
            total_volumes=1,
            total_size=total_size,
        )

    total_volumes = (total_size + volume_size - 1) // volume_size
    volume_paths = []

    for i in range(total_volumes):
        start = i * volume_size
        end = min(start + volume_size, total_size)
        chunk = zip_data[start:end]

        if i < total_volumes - 1:
            ext = f".z{i+1:02d}"
        else:
            ext = ".zip"

        vol_path = os.path.join(output_dir, f"{base_name}{ext}")
        with open(vol_path, "wb") as f:
            f.write(chunk)
        volume_paths.append(vol_path)

    return VolumeInfo(
        volume_paths=volume_paths,
        volume_size=volume_size,
        total_volumes=len(volume_paths),
        total_size=total_size,
    )


def merge_volumes_to_zip(
    volume_files: List[str],
    output_mode: str = "base64",
    output_path: Optional[str] = None,
) -> CompressResult:
    """
    将分卷文件合并还原为完整的ZIP

    分卷文件应按顺序提供：.z01, .z02, ..., .zip

    Args:
        volume_files: 分卷文件路径列表，需按正确顺序排列
        output_mode: 输出模式，"base64" / "tempfile" / "file"
        output_path: 当 output_mode 为 "file" 时，指定保存路径

    Returns:
        CompressResult 包含合并后的ZIP数据和统计信息
    """
    if not volume_files:
        raise ValueError("分卷文件列表不能为空")

    for vf in volume_files:
        if not os.path.exists(vf):
            raise FileNotFoundError(f"分卷文件不存在: {vf}")

    merged_data = b""
    total_volume_size = 0
    for vf in volume_files:
        with open(vf, "rb") as f:
            chunk = f.read()
            merged_data += chunk
            total_volume_size += len(chunk)

    if output_mode == "base64":
        b64_data = base64.b64encode(merged_data).decode("utf-8")
    elif output_mode == "tempfile":
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(merged_data)
            temp_path = tmp.name
        b64_data = temp_path
    elif output_mode == "file":
        if not output_path:
            raise ValueError("output_mode 为 'file' 时必须指定 output_path")
        with open(output_path, "wb") as f:
            f.write(merged_data)
        b64_data = output_path
    else:
        raise ValueError(f"无效的 output_mode: {output_mode}")

    original_size = 0
    file_count = 0
    file_stats = []
    with zipfile.ZipFile(BytesIO(merged_data), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            safe_name = ""
            try:
                safe_name = sanitize_filename(info.filename)
            except ValueError:
                safe_name = info.filename
            original_size += info.file_size
            file_count += 1
            file_stats.append(FileStat(
                filename=safe_name,
                original_size=info.file_size,
                compressed_size=info.compress_size,
            ))

    return CompressResult(
        data=b64_data,
        output_mode=output_mode,
        original_size=original_size,
        compressed_size=total_volume_size,
        file_count=file_count,
        files=file_stats,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("=== 开始测试 ZIP 工具 ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file1 = os.path.join(tmpdir, "test1.txt")
            test_file2 = os.path.join(tmpdir, "test2.txt")
            test_file3 = os.path.join(tmpdir, "data.csv")
            test_file4 = os.path.join(tmpdir, "image.png")

            with open(test_file1, "w", encoding="utf-8") as f:
                f.write("这是测试文件1的内容\nHello World!" * 100)
            with open(test_file2, "w", encoding="utf-8") as f:
                f.write("这是测试文件2的内容\nPython ZIP Test" * 100)
            with open(test_file3, "w", encoding="utf-8") as f:
                f.write("id,name,value\n1,A,100\n2,B,200\n3,C,300\n" * 50)
            with open(test_file4, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

            print("1. 测试 compress_files - 压缩率统计")
            result = compress_files([test_file1, test_file2, test_file3, test_file4], output_mode="base64")
            print(f"   {result.summary()}")

            print("\n2. 测试 compress_files - 扩展名过滤（仅 .txt）")
            result_txt = compress_files(
                [test_file1, test_file2, test_file3, test_file4],
                output_mode="base64",
                extensions=[".txt"],
            )
            print(f"   文件数: {result_txt.file_count}")
            for fs in result_txt.files:
                print(f"   - {fs.filename}")

            print("\n3. 测试 compress_files - 扩展名过滤（.txt + .csv）")
            result_tc = compress_files(
                [test_file1, test_file2, test_file3, test_file4],
                output_mode="base64",
                extensions=[".txt", ".csv"],
            )
            print(f"   文件数: {result_tc.file_count}")
            for fs in result_tc.files:
                print(f"   - {fs.filename}")

            print("\n4. 测试加密压缩与解密解压（AES-256）")
            if _HAS_PYZIPPER:
                enc_result = compress_bytes(
                    [("secret.txt", b"This is top secret data!"), ("notes.txt", b"Private notes here.")],
                    output_mode="base64",
                    password="my_password_123",
                )
                print(f"   加密压缩完成，Base64 长度: {len(enc_result.data)}")
                print(f"   文件数: {enc_result.file_count}")

                enc_extract = extract_zip(enc_result.data, source_type="base64", password="my_password_123")
                print(f"   解密解压成功，文件数: {enc_extract.file_count}")
                for fname, content in enc_extract.files.items():
                    print(f"   - {fname}: {content.decode('utf-8')}")

                print("   测试错误密码:")
                try:
                    extract_zip(enc_result.data, source_type="base64", password="wrong_password")
                    print("   [FAIL] 错误密码未拦截")
                except Exception as e:
                    print(f"   [OK] 错误密码被拦截: {type(e).__name__}")
            else:
                print("   pyzipper 未安装，跳过加密测试")

            print("\n5. 测试 compress_bytes - 压缩率统计")
            data_result = compress_bytes(
                [
                    ("repetitive.txt", b"A" * 10000),
                    ("random.bin", os.urandom(1000)),
                ],
                output_mode="base64",
            )
            print(f"   {data_result.summary()}")

            print("\n6. 测试分卷压缩")
            all_result = compress_files(
                [test_file1, test_file2, test_file3, test_file4],
                output_mode="file",
                output_path=os.path.join(tmpdir, "full.zip"),
            )

            vol_dir = os.path.join(tmpdir, "volumes")
            vol_info = split_zip_to_volumes(
                all_result.data,
                source_type="file",
                volume_size=256,
                output_dir=vol_dir,
                base_name="split_test",
            )
            print(f"   分卷数: {vol_info.total_volumes}")
            print(f"   总大小: {_fmt_size(vol_info.total_size)}")
            for i, vp in enumerate(vol_info.volume_paths):
                print(f"   分卷 {i+1}: {os.path.basename(vp)} ({_fmt_size(os.path.getsize(vp))})")

            print("\n7. 测试分卷合并与解压")
            merged = merge_volumes_to_zip(vol_info.volume_paths, output_mode="tempfile")
            print(f"   合并完成，文件数: {merged.file_count}")
            print(f"   {merged.summary()}")

            merged_extract = extract_zip(merged.data, source_type="file")
            print(f"   合并后解压文件数: {merged_extract.file_count}")

            print("\n8. 测试 list_zip_contents - 含压缩率")
            contents = list_zip_contents(all_result.data, source_type="file")
            for item in contents:
                print(
                    f"   - {item['safe_filename']}: "
                    f"{_fmt_size(item['size'])} -> {_fmt_size(item['compress_size'])} "
                    f"(节省 {item['saved_percent']:.1f}%, 加密: {item['encrypted']})"
                )

            print("\n9. 测试加密ZIP的 list_zip_contents")
            if _HAS_PYZIPPER:
                enc_contents = list_zip_contents(enc_result.data, source_type="base64", password="my_password_123")
                for item in enc_contents:
                    print(
                        f"   - {item['safe_filename']}: "
                        f"加密={item['encrypted']}"
                    )

            print("\n10. 测试路径遍历防护（不变）")
            malicious_data = [
                ("../../evil.txt", b"malicious content"),
                ("../secret.txt", b"steal data"),
            ]
            safe_result = compress_bytes(malicious_data, output_mode="base64")
            safe_extract = extract_zip(safe_result.data, source_type="base64")
            all_safe = all(".. " not in fn for fn in safe_extract.files)
            print(f"   [{'OK' if all_safe else 'FAIL'}] 所有文件名无路径遍历")
            for fn in safe_extract.files:
                print(f"   - {fn}")

        print("\n=== 测试完成 ===")
