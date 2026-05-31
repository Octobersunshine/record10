import base64
import io
import os
import struct
import zipfile
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


class MagicSignature(NamedTuple):
    magic: bytes
    offset: int
    min_length: int
    file_type: str
    mime_type: str
    secondary_magic: Optional[bytes] = None
    secondary_offset: int = 0


class DetectionResult(NamedTuple):
    file_type: Optional[str]
    mime_type: Optional[str]
    truncated: bool = False


class VerificationResult(NamedTuple):
    file_type: Optional[str]
    mime_type: Optional[str]
    truncated: bool
    declared_extension: Optional[str]
    expected_type: Optional[str]
    match: bool
    disguised: bool


class FileTypeDetector:

    SIGNATURES = [
        MagicSignature(
            magic=b'\xFF\xD8\xFF',
            offset=0,
            min_length=3,
            file_type='JPEG',
            mime_type='image/jpeg',
        ),
        MagicSignature(
            magic=b'\x89PNG\r\n\x1a\n',
            offset=0,
            min_length=8,
            file_type='PNG',
            mime_type='image/png',
        ),
        MagicSignature(
            magic=b'GIF87a',
            offset=0,
            min_length=6,
            file_type='GIF',
            mime_type='image/gif',
        ),
        MagicSignature(
            magic=b'GIF89a',
            offset=0,
            min_length=6,
            file_type='GIF',
            mime_type='image/gif',
        ),
        MagicSignature(
            magic=b'BM',
            offset=0,
            min_length=14,
            file_type='BMP',
            mime_type='image/bmp',
        ),
        MagicSignature(
            magic=b'II*\x00',
            offset=0,
            min_length=8,
            file_type='TIFF',
            mime_type='image/tiff',
        ),
        MagicSignature(
            magic=b'MM\x00*',
            offset=0,
            min_length=8,
            file_type='TIFF',
            mime_type='image/tiff',
        ),
        MagicSignature(
            magic=b'%PDF-',
            offset=0,
            min_length=5,
            file_type='PDF',
            mime_type='application/pdf',
        ),
        MagicSignature(
            magic=b'PK\x03\x04',
            offset=0,
            min_length=4,
            file_type='ZIP',
            mime_type='application/zip',
        ),
        MagicSignature(
            magic=b'PK\x05\x06',
            offset=0,
            min_length=4,
            file_type='ZIP',
            mime_type='application/zip',
        ),
        MagicSignature(
            magic=b'PK\x07\x08',
            offset=0,
            min_length=4,
            file_type='ZIP',
            mime_type='application/zip',
        ),
        MagicSignature(
            magic=b'Rar!\x1a\x07\x00',
            offset=0,
            min_length=7,
            file_type='RAR',
            mime_type='application/x-rar-compressed',
        ),
        MagicSignature(
            magic=b'Rar!\x1a\x07\x01\x00',
            offset=0,
            min_length=7,
            file_type='RAR5',
            mime_type='application/x-rar-compressed',
        ),
        MagicSignature(
            magic=b'\x1f\x8b',
            offset=0,
            min_length=10,
            file_type='GZIP',
            mime_type='application/gzip',
        ),
        MagicSignature(
            magic=b'BZh',
            offset=0,
            min_length=4,
            file_type='BZIP2',
            mime_type='application/x-bzip2',
        ),
        MagicSignature(
            magic=b'\xfd7zXZ\x00',
            offset=0,
            min_length=6,
            file_type='XZ',
            mime_type='application/x-xz',
        ),
        MagicSignature(
            magic=b'\x5d\x00\x00',
            offset=0,
            min_length=13,
            file_type='LZMA',
            mime_type='application/x-lzma',
        ),
        MagicSignature(
            magic=b'\x7fELF',
            offset=0,
            min_length=16,
            file_type='ELF',
            mime_type='application/x-executable',
        ),
        MagicSignature(
            magic=b'MZ',
            offset=0,
            min_length=2,
            file_type='EXE',
            mime_type='application/x-dosexec',
        ),
        MagicSignature(
            magic=b'\xca\xfe\xba\xbe',
            offset=0,
            min_length=4,
            file_type='Java Class',
            mime_type='application/java-vm',
        ),
        MagicSignature(
            magic=b'\xcf\xfa\xed\xfe',
            offset=0,
            min_length=4,
            file_type='Mach-O 64 LE',
            mime_type='application/x-mach-binary',
        ),
        MagicSignature(
            magic=b'\xfe\xed\xfa\xcf',
            offset=0,
            min_length=4,
            file_type='Mach-O 64 BE',
            mime_type='application/x-mach-binary',
        ),
        MagicSignature(
            magic=b'\xce\xfa\xed\xfe',
            offset=0,
            min_length=4,
            file_type='Mach-O 32 LE',
            mime_type='application/x-mach-binary',
        ),
        MagicSignature(
            magic=b'\xfe\xed\xfa\xce',
            offset=0,
            min_length=4,
            file_type='Mach-O 32 BE',
            mime_type='application/x-mach-binary',
        ),
        MagicSignature(
            magic=b'RIFF',
            offset=0,
            min_length=12,
            file_type='WAV',
            mime_type='audio/wav',
            secondary_magic=b'WAVE',
            secondary_offset=8,
        ),
        MagicSignature(
            magic=b'RIFF',
            offset=0,
            min_length=12,
            file_type='AVI',
            mime_type='video/avi',
            secondary_magic=b'AVI ',
            secondary_offset=8,
        ),
        MagicSignature(
            magic=b'RIFF',
            offset=0,
            min_length=12,
            file_type='WEBP',
            mime_type='image/webp',
            secondary_magic=b'WEBP',
            secondary_offset=8,
        ),
        MagicSignature(
            magic=b'OggS',
            offset=0,
            min_length=4,
            file_type='OGG',
            mime_type='audio/ogg',
        ),
        MagicSignature(
            magic=b'\x1a\x45\xdf\xa3',
            offset=0,
            min_length=4,
            file_type='MKV',
            mime_type='video/x-matroska',
        ),
        MagicSignature(
            magic=b'ftypheic',
            offset=4,
            min_length=12,
            file_type='HEIC',
            mime_type='image/heic',
        ),
        MagicSignature(
            magic=b'ftyphvc1',
            offset=4,
            min_length=12,
            file_type='HEIF',
            mime_type='image/heif',
        ),
        MagicSignature(
            magic=b'ftyp',
            offset=4,
            min_length=8,
            file_type='MP4',
            mime_type='video/mp4',
        ),
        MagicSignature(
            magic=b'ID3',
            offset=0,
            min_length=3,
            file_type='MP3',
            mime_type='audio/mpeg',
        ),
        MagicSignature(
            magic=b'\xff\xfb',
            offset=0,
            min_length=3,
            file_type='MP3',
            mime_type='audio/mpeg',
        ),
        MagicSignature(
            magic=b'\xff\xf3',
            offset=0,
            min_length=3,
            file_type='MP3',
            mime_type='audio/mpeg',
        ),
        MagicSignature(
            magic=b'\xff\xf2',
            offset=0,
            min_length=3,
            file_type='MP3',
            mime_type='audio/mpeg',
        ),
        MagicSignature(
            magic=b'\xff\xf1',
            offset=0,
            min_length=3,
            file_type='AAC',
            mime_type='audio/aac',
        ),
        MagicSignature(
            magic=b'\xff\xf9',
            offset=0,
            min_length=3,
            file_type='AAC',
            mime_type='audio/aac',
        ),
        MagicSignature(
            magic=b'fLaC',
            offset=0,
            min_length=4,
            file_type='FLAC',
            mime_type='audio/flac',
        ),
        MagicSignature(
            magic=b'MThd',
            offset=0,
            min_length=4,
            file_type='MIDI',
            mime_type='audio/midi',
        ),
        MagicSignature(
            magic=b'{\\rtf',
            offset=0,
            min_length=5,
            file_type='RTF',
            mime_type='application/rtf',
        ),
        MagicSignature(
            magic=b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',
            offset=0,
            min_length=8,
            file_type='DOC',
            mime_type='application/msword',
        ),
        MagicSignature(
            magic=b'\x25\x21\x50\x53',
            offset=0,
            min_length=4,
            file_type='PS',
            mime_type='application/postscript',
        ),
        MagicSignature(
            magic=b'\x00\x00\x01\x00',
            offset=0,
            min_length=6,
            file_type='ICO',
            mime_type='image/x-icon',
        ),
        MagicSignature(
            magic=b'\x00\x00\x02\x00',
            offset=0,
            min_length=6,
            file_type='CUR',
            mime_type='image/x-icon',
        ),
        MagicSignature(
            magic=b'8BPS',
            offset=0,
            min_length=4,
            file_type='PSD',
            mime_type='image/vnd.adobe.photoshop',
        ),
        MagicSignature(
            magic=b'FLV',
            offset=0,
            min_length=5,
            file_type='FLV',
            mime_type='video/x-flv',
        ),
        MagicSignature(
            magic=b'7z\xbc\xaf\x27\x1c',
            offset=0,
            min_length=6,
            file_type='7Z',
            mime_type='application/x-7z-compressed',
        ),
        MagicSignature(
            magic=b'\x00\x61\x73\x6d',
            offset=0,
            min_length=8,
            file_type='WASM',
            mime_type='application/wasm',
        ),
        MagicSignature(
            magic=b'\x1b\x5b\x4b\x2d\x4a\x04\x00',
            offset=0,
            min_length=7,
            file_type='EPS',
            mime_type='application/postscript',
        ),
        MagicSignature(
            magic=b'SQLite format 3\x00',
            offset=0,
            min_length=16,
            file_type='SQLite',
            mime_type='application/x-sqlite3',
        ),
        MagicSignature(
            magic=b'\xd4\xc3\xb2\xa1',
            offset=0,
            min_length=24,
            file_type='PCAP',
            mime_type='application/vnd.tcpdump.pcap',
        ),
        MagicSignature(
            magic=b'\xa1\xb2\xc3\xd4',
            offset=0,
            min_length=24,
            file_type='PCAP-NS',
            mime_type='application/vnd.tcpdump.pcap',
        ),
        MagicSignature(
            magic=b'ustar',
            offset=257,
            min_length=262,
            file_type='TAR',
            mime_type='application/x-tar',
        ),
        MagicSignature(
            magic=b'\x00\x00\x00\x1c\x66\x74\x79\x70',
            offset=0,
            min_length=12,
            file_type='3GP',
            mime_type='video/3gpp',
        ),
    ]

    ZIP_CONTENT_DETECTORS = {
        'DOCX': {
            'required_files': ['[Content_Types].xml', 'word/document.xml'],
            'mime_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        },
        'XLSX': {
            'required_files': ['[Content_Types].xml', 'xl/workbook.xml'],
            'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        },
        'PPTX': {
            'required_files': ['[Content_Types].xml', 'ppt/presentation.xml'],
            'mime_type': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        },
        'APK': {
            'required_files': ['AndroidManifest.xml'],
            'mime_type': 'application/vnd.android.package-archive',
        },
        'JAR': {
            'required_files': ['META-INF/MANIFEST.MF'],
            'mime_type': 'application/java-archive',
        },
        'ODT': {
            'required_files': ['mimetype', 'content.xml'],
            'mime_type': 'application/vnd.oasis.opendocument.text',
        },
        'ODS': {
            'required_files': ['mimetype', 'content.xml'],
            'mime_type': 'application/vnd.oasis.opendocument.spreadsheet',
        },
        'ODP': {
            'required_files': ['mimetype', 'content.xml'],
            'mime_type': 'application/vnd.oasis.opendocument.presentation',
        },
        'EPUB': {
            'required_files': ['mimetype', 'META-INF/container.xml'],
            'mime_type': 'application/epub+zip',
        },
    }

    EXTENSION_TYPE_MAP: Dict[str, Tuple[str, str]] = {
        '.jpg': ('JPEG', 'image/jpeg'),
        '.jpeg': ('JPEG', 'image/jpeg'),
        '.png': ('PNG', 'image/png'),
        '.gif': ('GIF', 'image/gif'),
        '.bmp': ('BMP', 'image/bmp'),
        '.tiff': ('TIFF', 'image/tiff'),
        '.tif': ('TIFF', 'image/tiff'),
        '.webp': ('WEBP', 'image/webp'),
        '.ico': ('ICO', 'image/x-icon'),
        '.cur': ('CUR', 'image/x-icon'),
        '.psd': ('PSD', 'image/vnd.adobe.photoshop'),
        '.heic': ('HEIC', 'image/heic'),
        '.heif': ('HEIF', 'image/heif'),
        '.svg': ('SVG', 'image/svg+xml'),
        '.pdf': ('PDF', 'application/pdf'),
        '.doc': ('DOC', 'application/msword'),
        '.docx': ('DOCX', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        '.xls': ('XLS', 'application/vnd.ms-excel'),
        '.xlsx': ('XLSX', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        '.ppt': ('PPT', 'application/vnd.ms-powerpoint'),
        '.pptx': ('PPTX', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
        '.odt': ('ODT', 'application/vnd.oasis.opendocument.text'),
        '.ods': ('ODS', 'application/vnd.oasis.opendocument.spreadsheet'),
        '.odp': ('ODP', 'application/vnd.oasis.opendocument.presentation'),
        '.epub': ('EPUB', 'application/epub+zip'),
        '.rtf': ('RTF', 'application/rtf'),
        '.ps': ('PS', 'application/postscript'),
        '.eps': ('EPS', 'application/postscript'),
        '.zip': ('ZIP', 'application/zip'),
        '.rar': ('RAR', 'application/x-rar-compressed'),
        '.7z': ('7Z', 'application/x-7z-compressed'),
        '.gz': ('GZIP', 'application/gzip'),
        '.bz2': ('BZIP2', 'application/x-bzip2'),
        '.xz': ('XZ', 'application/x-xz'),
        '.lzma': ('LZMA', 'application/x-lzma'),
        '.tar': ('TAR', 'application/x-tar'),
        '.jar': ('JAR', 'application/java-archive'),
        '.apk': ('APK', 'application/vnd.android.package-archive'),
        '.mp3': ('MP3', 'audio/mpeg'),
        '.wav': ('WAV', 'audio/wav'),
        '.flac': ('FLAC', 'audio/flac'),
        '.aac': ('AAC', 'audio/aac'),
        '.ogg': ('OGG', 'audio/ogg'),
        '.mid': ('MIDI', 'audio/midi'),
        '.midi': ('MIDI', 'audio/midi'),
        '.mp4': ('MP4', 'video/mp4'),
        '.avi': ('AVI', 'video/avi'),
        '.mkv': ('MKV', 'video/x-matroska'),
        '.flv': ('FLV', 'video/x-flv'),
        '.3gp': ('3GP', 'video/3gpp'),
        '.webm': ('WEBM', 'video/webm'),
        '.exe': ('EXE', 'application/x-dosexec'),
        '.dll': ('EXE', 'application/x-dosexec'),
        '.so': ('ELF', 'application/x-executable'),
        '.class': ('Java Class', 'application/java-vm'),
        '.wasm': ('WASM', 'application/wasm'),
        '.sqlite': ('SQLite', 'application/x-sqlite3'),
        '.db': ('SQLite', 'application/x-sqlite3'),
        '.pcap': ('PCAP', 'application/vnd.tcpdump.pcap'),
    }

    ZIP_TYPE_EXTENSIONS: Dict[str, List[str]] = {
        'DOCX': ['.docx'],
        'XLSX': ['.xlsx'],
        'PPTX': ['.pptx'],
        'APK': ['.apk'],
        'JAR': ['.jar'],
        'ODT': ['.odt'],
        'ODS': ['.ods'],
        'ODP': ['.odp'],
        'EPUB': ['.epub'],
    }

    @classmethod
    def _get_required_read_size(cls) -> int:
        max_needed = 0
        for sig in cls.SIGNATURES:
            primary_end = sig.offset + len(sig.magic)
            max_needed = max(max_needed, primary_end)
            if sig.secondary_magic is not None:
                secondary_end = sig.secondary_offset + len(sig.secondary_magic)
                max_needed = max(max_needed, secondary_end)
            max_needed = max(max_needed, sig.min_length)
        return max_needed

    MIN_PARTIAL_MATCH = 2

    @classmethod
    def _match_signature(cls, sig: MagicSignature, data: bytes) -> Optional[DetectionResult]:
        if len(data) <= sig.offset:
            return None

        available = data[sig.offset:]
        compare_len = min(len(available), len(sig.magic))

        if available[:compare_len] != sig.magic[:compare_len]:
            return None

        if len(available) < len(sig.magic):
            if compare_len >= cls.MIN_PARTIAL_MATCH:
                return DetectionResult(
                    file_type=sig.file_type,
                    mime_type=sig.mime_type,
                    truncated=True,
                )
            return None

        if sig.secondary_magic is not None:
            sec_end = sig.secondary_offset + len(sig.secondary_magic)
            if len(data) < sec_end:
                return DetectionResult(
                    file_type=sig.file_type,
                    mime_type=sig.mime_type,
                    truncated=True,
                )
            if data[sig.secondary_offset:sec_end] != sig.secondary_magic:
                return None

        if len(data) < sig.min_length:
            return DetectionResult(
                file_type=sig.file_type,
                mime_type=sig.mime_type,
                truncated=True,
            )

        return DetectionResult(
            file_type=sig.file_type,
            mime_type=sig.mime_type,
            truncated=False,
        )

    @classmethod
    def _detect_zip_subtype(cls, data: bytes) -> Optional[Tuple[str, str]]:
        try:
            bio = io.BytesIO(data)
            if not zipfile.is_zipfile(bio):
                return None
            bio.seek(0)
            with zipfile.ZipFile(bio, 'r') as zf:
                names = set(zf.namelist())
                for subtype, info in cls.ZIP_CONTENT_DETECTORS.items():
                    if all(f in names for f in info['required_files']):
                        if subtype == 'ODT':
                            try:
                                mimetype_content = zf.read('mimetype').decode('ascii').strip()
                                odf_map = {
                                    'application/vnd.oasis.opendocument.text': 'ODT',
                                    'application/vnd.oasis.opendocument.spreadsheet': 'ODS',
                                    'application/vnd.oasis.opendocument.presentation': 'ODP',
                                }
                                actual = odf_map.get(mimetype_content)
                                if actual and actual in cls.ZIP_CONTENT_DETECTORS:
                                    return actual, cls.ZIP_CONTENT_DETECTORS[actual]['mime_type']
                            except Exception:
                                pass
                        return subtype, info['mime_type']
        except Exception:
            pass
        return None

    @classmethod
    def _detect_zip_subtype_from_file(cls, file_path: str) -> Optional[Tuple[str, str]]:
        try:
            if not zipfile.is_zipfile(file_path):
                return None
            with zipfile.ZipFile(file_path, 'r') as zf:
                names = set(zf.namelist())
                for subtype, info in cls.ZIP_CONTENT_DETECTORS.items():
                    if all(f in names for f in info['required_files']):
                        if subtype == 'ODT':
                            try:
                                mimetype_content = zf.read('mimetype').decode('ascii').strip()
                                odf_map = {
                                    'application/vnd.oasis.opendocument.text': 'ODT',
                                    'application/vnd.oasis.opendocument.spreadsheet': 'ODS',
                                    'application/vnd.oasis.opendocument.presentation': 'ODP',
                                }
                                actual = odf_map.get(mimetype_content)
                                if actual and actual in cls.ZIP_CONTENT_DETECTORS:
                                    return actual, cls.ZIP_CONTENT_DETECTORS[actual]['mime_type']
                            except Exception:
                                pass
                        return subtype, info['mime_type']
        except Exception:
            pass
        return None

    @classmethod
    def detect_from_bytes(cls, data: bytes) -> DetectionResult:
        if not data:
            return DetectionResult(file_type=None, mime_type=None)

        for sig in cls.SIGNATURES:
            result = cls._match_signature(sig, data)
            if result is not None:
                if result.file_type == 'ZIP' and not result.truncated:
                    subtype_result = cls._detect_zip_subtype(data)
                    if subtype_result:
                        return DetectionResult(
                            file_type=subtype_result[0],
                            mime_type=subtype_result[1],
                            truncated=False,
                        )
                return result

        return DetectionResult(file_type='UNKNOWN', mime_type='application/octet-stream')

    @classmethod
    def detect_from_base64(cls, base64_str: str) -> DetectionResult:
        try:
            data = base64.b64decode(base64_str)
            return cls.detect_from_bytes(data)
        except Exception:
            return DetectionResult(file_type=None, mime_type=None)

    @classmethod
    def detect_from_file(cls, file_path: str) -> DetectionResult:
        try:
            read_size = cls._get_required_read_size()
            with open(file_path, 'rb') as f:
                header = f.read(read_size)
            result = cls.detect_from_bytes(header)
            if result.file_type == 'ZIP' and not result.truncated:
                subtype_result = cls._detect_zip_subtype_from_file(file_path)
                if subtype_result:
                    return DetectionResult(
                        file_type=subtype_result[0],
                        mime_type=subtype_result[1],
                        truncated=False,
                    )
            return result
        except Exception:
            return DetectionResult(file_type=None, mime_type=None)

    @classmethod
    def detect_from_stream(cls, stream: io.IOBase) -> DetectionResult:
        try:
            read_size = cls._get_required_read_size()
            current_pos = stream.tell()
            header = stream.read(read_size)
            stream.seek(current_pos)
            return cls.detect_from_bytes(header)
        except Exception:
            return DetectionResult(file_type=None, mime_type=None)

    @classmethod
    def _get_extension(cls, file_path: str) -> Optional[str]:
        _, ext = os.path.splitext(file_path)
        return ext.lower() if ext else None

    @classmethod
    def _types_compatible(cls, detected_type: Optional[str], expected_type: Optional[str]) -> bool:
        if detected_type is None or expected_type is None:
            return True
        if detected_type == expected_type:
            return True
        compatibility_groups = [
            {'JPEG', 'JPG'},
            {'TIFF', 'TIF'},
            {'MP3'},
            {'RAR', 'RAR5'},
            {'Mach-O Fat', 'Mach-O 64 LE', 'Mach-O 64 BE', 'Mach-O 32 LE', 'Mach-O 32 BE'},
            {'EXE', 'DLL'},
            {'DOC', 'XLS', 'PPT'},
            {'ZIP', 'DOCX', 'XLSX', 'PPTX', 'APK', 'JAR', 'ODT', 'ODS', 'ODP', 'EPUB'},
            {'PCAP', 'PCAP-NS'},
        ]
        for group in compatibility_groups:
            if detected_type in group and expected_type in group:
                return True
        return False

    @classmethod
    def verify_file_type(cls, file_path: str) -> VerificationResult:
        detection = cls.detect_from_file(file_path)
        ext = cls._get_extension(file_path)
        expected_type = None
        if ext and ext in cls.EXTENSION_TYPE_MAP:
            expected_type = cls.EXTENSION_TYPE_MAP[ext][0]

        compatible = cls._types_compatible(detection.file_type, expected_type)
        disguised = (
            expected_type is not None
            and detection.file_type is not None
            and detection.file_type != 'UNKNOWN'
            and not compatible
        )

        return VerificationResult(
            file_type=detection.file_type,
            mime_type=detection.mime_type,
            truncated=detection.truncated,
            declared_extension=ext,
            expected_type=expected_type,
            match=compatible,
            disguised=disguised,
        )

    @classmethod
    def verify_bytes(cls, data: bytes, declared_extension: Optional[str] = None) -> VerificationResult:
        detection = cls.detect_from_bytes(data)
        expected_type = None
        if declared_extension and declared_extension.lower() in cls.EXTENSION_TYPE_MAP:
            expected_type = cls.EXTENSION_TYPE_MAP[declared_extension.lower()][0]

        compatible = cls._types_compatible(detection.file_type, expected_type)
        disguised = (
            expected_type is not None
            and detection.file_type is not None
            and detection.file_type != 'UNKNOWN'
            and not compatible
        )

        return VerificationResult(
            file_type=detection.file_type,
            mime_type=detection.mime_type,
            truncated=detection.truncated,
            declared_extension=declared_extension,
            expected_type=expected_type,
            match=compatible,
            disguised=disguised,
        )

    @classmethod
    def batch_detect_files(cls, file_paths: List[str]) -> List[dict]:
        results = []
        for path in file_paths:
            result = detect_file_type(path)
            result['file_path'] = path
            results.append(result)
        return results

    @classmethod
    def batch_detect_bytes(cls, data_list: List[bytes]) -> List[dict]:
        results = []
        for data in data_list:
            result = detect_file_type(data)
            results.append(result)
        return results

    @classmethod
    def batch_verify_files(cls, file_paths: List[str]) -> List[dict]:
        results = []
        for path in file_paths:
            v = cls.verify_file_type(path)
            results.append({
                'file_path': path,
                'file_type': v.file_type,
                'mime_type': v.mime_type,
                'truncated': v.truncated,
                'declared_extension': v.declared_extension,
                'expected_type': v.expected_type,
                'match': v.match,
                'disguised': v.disguised,
            })
        return results


def detect_file_type(input_data: Union[str, bytes, io.IOBase]) -> dict:
    detector = FileTypeDetector

    if isinstance(input_data, bytes):
        result = detector.detect_from_bytes(input_data)
    elif isinstance(input_data, str):
        if os.path.isfile(input_data):
            result = detector.detect_from_file(input_data)
        else:
            result = detector.detect_from_base64(input_data)
    elif isinstance(input_data, io.IOBase):
        result = detector.detect_from_stream(input_data)
    else:
        return {'file_type': None, 'mime_type': None, 'truncated': False}

    return {
        'file_type': result.file_type,
        'mime_type': result.mime_type,
        'truncated': result.truncated,
    }


def verify_file_type(file_path: str) -> dict:
    v = FileTypeDetector.verify_file_type(file_path)
    return {
        'file_type': v.file_type,
        'mime_type': v.mime_type,
        'truncated': v.truncated,
        'declared_extension': v.declared_extension,
        'expected_type': v.expected_type,
        'match': v.match,
        'disguised': v.disguised,
    }


def verify_bytes(data: bytes, declared_extension: Optional[str] = None) -> dict:
    v = FileTypeDetector.verify_bytes(data, declared_extension)
    return {
        'file_type': v.file_type,
        'mime_type': v.mime_type,
        'truncated': v.truncated,
        'declared_extension': v.declared_extension,
        'expected_type': v.expected_type,
        'match': v.match,
        'disguised': v.disguised,
    }


def batch_detect_files(file_paths: List[str]) -> List[dict]:
    return FileTypeDetector.batch_detect_files(file_paths)


def batch_detect_bytes(data_list: List[bytes]) -> List[dict]:
    return FileTypeDetector.batch_detect_bytes(data_list)


def batch_verify_files(file_paths: List[str]) -> List[dict]:
    return FileTypeDetector.batch_verify_files(file_paths)


def magic_from_uint(value: int, byte_order: str, length: int = 4) -> bytes:
    fmt_map = {
        'little': '<I',
        'big': '>I',
    }
    if length == 2:
        fmt_map = {'little': '<H', 'big': '>H'}
    elif length == 8:
        fmt_map = {'little': '<Q', 'big': '>Q'}

    fmt = fmt_map.get(byte_order)
    if fmt is None:
        raise ValueError(f"Unsupported byte_order: {byte_order!r}. Use 'little' or 'big'.")

    return struct.pack(fmt, value)


if __name__ == '__main__':
    import tempfile

    print("=" * 60)
    print("1. 基本类型检测测试")
    print("=" * 60)

    test_cases = [
        ('JPEG', b'\xFF\xD8\xFF\xE0\x00\x10JFIF'),
        ('PNG', b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'),
        ('PDF', b'%PDF-1.4\n%...'),
        ('ZIP', b'PK\x03\x04\x14\x00\x00\x00'),
        ('GIF89a', b'GIF89a\x01\x00\x01\x00'),
        ('GIF87a', b'GIF87a\x01\x00\x01\x00'),
        ('FLAC', b'fLaC\x00\x00\x00\x22'),
        ('MIDI', b'MThd\x00\x00\x00\x06'),
        ('SQLite', b'SQLite format 3\x00\x10\x00\x01\x01'),
        ('XZ', b'\xfd7zXZ\x00\x00'),
        ('PCAP', b'\xd4\xc3\xb2\xa1\x02\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x01\x00\x00\x00'),
    ]

    for name, data in test_cases:
        result = detect_file_type(data)
        print(f"\n  {name:10s} -> type={result['file_type']}, "
              f"mime={result['mime_type']}, truncated={result['truncated']}")

    print("\n" + "=" * 60)
    print("2. MP4 / HEIC 偏移检测测试")
    print("=" * 60)

    mp4_header = struct.pack('>I', 32) + b'ftypisom'
    result = detect_file_type(mp4_header)
    print(f"\n  MP4 (ftyp@4) -> type={result['file_type']}, truncated={result['truncated']}")

    heic_header = struct.pack('>I', 24) + b'ftypheic'
    result = detect_file_type(heic_header)
    print(f"  HEIC (ftypheic@4) -> type={result['file_type']}, truncated={result['truncated']}")

    print("\n" + "=" * 60)
    print("3. RIFF 容器子类型区分测试")
    print("=" * 60)

    riff_tests = [
        ('WAV', b'RIFF\x24\x00\x00\x00WAVEfmt '),
        ('AVI', b'RIFF\x24\x00\x00\x00AVI LIST'),
        ('WEBP', b'RIFF\x24\x00\x00\x00WEBPVP8 '),
    ]
    for name, data in riff_tests:
        result = detect_file_type(data)
        print(f"\n  {name:6s} -> type={result['file_type']}, "
              f"mime={result['mime_type']}")

    print("\n" + "=" * 60)
    print("4. ZIP 内部结构二次检测 (DOCX/XLSX/PPTX)")
    print("=" * 60)

    def make_zip_bytes(files: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        return buf.getvalue()

    docx_data = make_zip_bytes({
        '[Content_Types].xml': '<?xml?>',
        'word/document.xml': '<?xml?>',
    })
    xlsx_data = make_zip_bytes({
        '[Content_Types].xml': '<?xml?>',
        'xl/workbook.xml': '<?xml?>',
    })
    pptx_data = make_zip_bytes({
        '[Content_Types].xml': '<?xml?>',
        'ppt/presentation.xml': '<?xml?>',
    })
    odt_data = make_zip_bytes({
        'mimetype': 'application/vnd.oasis.opendocument.text',
        'content.xml': '<?xml?>',
    })
    epub_data = make_zip_bytes({
        'mimetype': 'application/epub+zip',
        'META-INF/container.xml': '<?xml?>',
    })
    plain_zip_data = make_zip_bytes({
        'hello.txt': 'world',
    })

    zip_subtype_tests = [
        ('DOCX', docx_data),
        ('XLSX', xlsx_data),
        ('PPTX', pptx_data),
        ('ODT', odt_data),
        ('EPUB', epub_data),
        ('ZIP (plain)', plain_zip_data),
    ]
    for name, data in zip_subtype_tests:
        result = detect_file_type(data)
        print(f"\n  {name:12s} -> type={result['file_type']}, mime={result['mime_type']}")

    print("\n" + "=" * 60)
    print("5. 伪装文件检测测试")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir='.') as f:
        f.write(b'%PDF-1.4 fake content here')
        fake_png_path = f.name

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, dir='.') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')
        fake_pdf_path = f.name

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='.') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')
        fake_jpg_path = f.name

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir='.') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')
        real_png_path = f.name

    disguise_tests = [
        ('malware.png (is PDF)', fake_png_path),
        ('report.pdf (is PNG)', fake_pdf_path),
        ('photo.jpg (is PNG)', fake_jpg_path),
        ('image.png (is PNG)', real_png_path),
    ]
    for name, path in disguise_tests:
        v = verify_file_type(path)
        status = "DISGUISED!" if v['disguised'] else ("match" if v['match'] else "mismatch")
        print(f"\n  {name:25s} -> ext={v['declared_extension']}, "
              f"expected={v['expected_type']}, detected={v['file_type']}, "
              f"status={status}")

    for _, path in disguise_tests:
        try:
            os.unlink(path)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("6. 字节流伪装检测测试")
    print("=" * 60)

    jpeg_bytes = b'\xFF\xD8\xFF\xE0\x00\x10JFIF'
    v1 = verify_bytes(jpeg_bytes, '.png')
    v2 = verify_bytes(jpeg_bytes, '.jpg')
    print(f"\n  JPEG bytes declared as .png -> disguised={v1['disguised']}, "
          f"expected={v1['expected_type']}, detected={v1['file_type']}")
    print(f"  JPEG bytes declared as .jpg -> disguised={v2['disguised']}, "
          f"expected={v2['expected_type']}, detected={v2['file_type']}")

    print("\n" + "=" * 60)
    print("7. 批量检测测试")
    print("=" * 60)

    batch_data = [
        b'\xFF\xD8\xFF\xE0JFIF',
        b'\x89PNG\r\n\x1a\nIHDR',
        b'%PDF-1.4',
        b'PK\x03\x04\x14\x00\x00\x00',
    ]
    batch_results = batch_detect_bytes(batch_data)
    for i, r in enumerate(batch_results):
        print(f"\n  [{i}] type={r['file_type']}, mime={r['mime_type']}")

    print("\n" + "=" * 60)
    print("8. 截断文件检测测试")
    print("=" * 60)

    truncation_tests = [
        ('JPEG 3B (ok)', b'\xFF\xD8\xFF', False),
        ('JPEG 2B (short)', b'\xFF\xD8', True),
        ('BMP 2B (short)', b'BM', True),
        ('BMP 14B (ok)', b'BM\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', False),
        ('WAV 8B (no sub)', b'RIFF\x00\x00\x00\x00', True),
        ('PNG 4B (short)', b'\x89PNG', True),
        ('PDF 3B (short)', b'%PD', True),
    ]
    for name, data, expect_trunc in truncation_tests:
        result = detect_file_type(data)
        status = "PASS" if result['truncated'] == expect_trunc else "FAIL"
        print(f"\n  [{status}] {name:22s} -> type={result['file_type']}, "
              f"truncated={result['truncated']} (expected={expect_trunc})")

    print("\n" + "=" * 60)
