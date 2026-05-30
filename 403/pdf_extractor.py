import json
import argparse
import os
import shutil
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject


OCR_AVAILABLE = False
TESSERACT_CMD = None
POPPLER_PATH = None
OCR_IMPORT_ERROR = None


try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image

    def _find_tesseract() -> Optional[str]:
        env_path = os.environ.get("TESSERACT_CMD")
        if env_path and os.path.exists(env_path):
            return env_path
        candidates = [
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%USERPROFILE%\scoop\shims\tesseract.exe"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def _find_poppler() -> Optional[str]:
        env_path = os.environ.get("POPPLER_PATH")
        if env_path and os.path.exists(env_path):
            if os.path.isdir(env_path):
                return env_path
            return os.path.dirname(env_path)
        candidates = [
            shutil.which("pdftoppm"),
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin",
            os.path.expandvars(r"%PROGRAMFILES%\poppler\bin"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\poppler\bin"),
            os.path.expandvars(r"%USERPROFILE%\scoop\shims"),
        ]
        for c in candidates:
            if c:
                if os.path.isdir(c) and os.path.exists(os.path.join(c, "pdftoppm.exe")):
                    return c
                if os.path.isfile(c) and "pdftoppm" in os.path.basename(c):
                    return os.path.dirname(c)
        return None

    TESSERACT_CMD = _find_tesseract()
    POPPLER_PATH = _find_poppler()

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        OCR_AVAILABLE = True
    else:
        OCR_IMPORT_ERROR = "Tesseract OCR not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki"

except ImportError as e:
    OCR_IMPORT_ERROR = f"OCR dependencies not installed: {str(e)}. Install with: pip install pytesseract pdf2image pillow"
except Exception as e:
    OCR_IMPORT_ERROR = f"OCR initialization error: {str(e)}"


def parse_pdf_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    date_str = date_str.strip()
    if date_str.startswith("D:"):
        date_str = date_str[2:]
    formats = [
        "%Y%m%d%H%M%S%z",
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str[: len(fmt) + 6] if "%z" in fmt else date_str[: len(fmt)], fmt)
            return dt.isoformat()
        except ValueError:
            continue
    return date_str


def ocr_available() -> Tuple[bool, Optional[str]]:
    return OCR_AVAILABLE, OCR_IMPORT_ERROR


def check_ocr_languages(lang: str = "eng+chi_sim") -> bool:
    if not OCR_AVAILABLE:
        return False
    try:
        import pytesseract
        available = pytesseract.get_languages(config='')
        requested = lang.split('+')
        return all(l in available for l in requested)
    except Exception:
        return False


def extract_text_ocr(
    pdf_path: str,
    password: Optional[str] = None,
    lang: str = "eng+chi_sim",
    dpi: int = 300,
    page_numbers: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    if not OCR_AVAILABLE:
        raise RuntimeError(OCR_IMPORT_ERROR or "OCR not available")

    import pytesseract
    from pdf2image import convert_from_path

    pages_result = []

    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            userpw=password,
            poppler_path=POPPLER_PATH,
            first_page=page_numbers[0] if page_numbers else None,
            last_page=page_numbers[-1] if page_numbers else None,
        )

        for idx, image in enumerate(images):
            actual_page_num = (page_numbers[idx] if page_numbers else idx + 1)
            text = pytesseract.image_to_string(image, lang=lang)
            pages_result.append({
                "page_number": actual_page_num,
                "text": text,
                "character_count": len(text),
                "extraction_method": "ocr"
            })

    except Exception as e:
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            raise RuntimeError(
                "Poppler not found. Install from: https://github.com/oschwartz10612/poppler-windows/releases "
                "or set POPPLER_PATH environment variable."
            ) from e
        raise

    return pages_result


def extract_text_pypdf(reader: PdfReader) -> List[Dict[str, Any]]:
    pages_result = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_result.append({
            "page_number": i + 1,
            "text": text,
            "character_count": len(text),
            "extraction_method": "text_layer"
        })
    return pages_result


def has_text_layer(pages_result: List[Dict[str, Any]], threshold: int = 10) -> bool:
    if not pages_result:
        return False
    total_chars = sum(p["character_count"] for p in pages_result)
    avg_chars = total_chars / len(pages_result)
    return avg_chars >= threshold


def extract_pdf(
    pdf_path: str,
    password: Optional[str] = None,
    ocr_enabled: bool = True,
    ocr_lang: str = "eng+chi_sim",
    ocr_dpi: int = 300,
    ocr_fallback_threshold: int = 10,
    extract_images: bool = False
) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)

    result = {
        "file_path": str(pdf_path.absolute()),
        "file_name": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else None,
        "metadata": {},
        "pages": [],
        "images": [],
        "ocr": {
            "enabled": ocr_enabled,
            "available": OCR_AVAILABLE,
            "used": False,
            "lang": ocr_lang,
            "dpi": ocr_dpi,
            "error": None
        },
        "success": False
    }

    if not pdf_path.exists():
        result["error"] = f"File not found: {pdf_path}"
        return result

    try:
        reader = PdfReader(str(pdf_path))

        if reader.is_encrypted:
            if password is None:
                result["error"] = "PDF is password protected, please provide a password"
                result["success"] = False
                return result
            try:
                decrypt_result = reader.decrypt(password)
                if decrypt_result == 0:
                    result["error"] = "Failed to decrypt PDF: incorrect password"
                    result["success"] = False
                    return result
            except Exception as e:
                result["error"] = f"Failed to decrypt PDF: {str(e)}"
                result["success"] = False
                return result

        info = reader.metadata
        result["metadata"] = {
            "page_count": len(reader.pages),
            "title": info.title if info else None,
            "author": info.author if info else None,
            "subject": info.subject if info else None,
            "creator": info.creator if info else None,
            "producer": info.producer if info else None,
            "creation_date": parse_pdf_date(info.creation_date_raw) if info and info.creation_date_raw else None,
            "modification_date": parse_pdf_date(info.modification_date_raw) if info and info.modification_date_raw else None,
        }

        pages = extract_text_pypdf(reader)

        if ocr_enabled and not has_text_layer(pages, ocr_fallback_threshold):
            if OCR_AVAILABLE:
                try:
                    ocr_pages = extract_text_ocr(
                        str(pdf_path),
                        password=password,
                        lang=ocr_lang,
                        dpi=ocr_dpi
                    )
                    result["pages"] = ocr_pages
                    result["ocr"]["used"] = True
                    result["ocr"]["fallback_reason"] = "No text layer detected (scanned PDF)"
                except Exception as e:
                    result["pages"] = pages
                    result["ocr"]["error"] = str(e)
            else:
                result["pages"] = pages
                result["ocr"]["error"] = OCR_IMPORT_ERROR
                if not any(p["character_count"] for p in pages):
                    result["warning"] = (
                        "This appears to be a scanned PDF with no text layer. "
                        "Install Tesseract OCR to enable text extraction. "
                        f"Details: {OCR_IMPORT_ERROR}"
                    )
        else:
            result["pages"] = pages

        if extract_images:
            result["images"] = extract_images_from_pdf(reader)

        result["total_character_count"] = sum(p["character_count"] for p in result["pages"])
        result["success"] = True

    except Exception as e:
        result["error"] = f"Failed to process PDF: {str(e)}"
        result["success"] = False

    return result


def extract_images_from_pdf(
    pdf_source: Union[str, PdfReader],
    password: Optional[str] = None,
    output_format: str = "base64"
) -> List[Dict[str, Any]]:
    images_result = []

    if isinstance(pdf_source, str):
        pdf_path = Path(pdf_source)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted and password:
            reader.decrypt(password)
    else:
        reader = pdf_source

    try:
        from PIL import Image as PILImage

        for page_idx, page in enumerate(reader.pages):
            try:
                resources = page["/Resources"]
                if "/XObject" in resources:
                    xobjects = resources["/XObject"]
                    for obj_name in xobjects:
                        obj = xobjects[obj_name]
                        if obj.get("/Subtype") == "/Image":
                            try:
                                img_data = obj.get_data()
                                width = obj.get("/Width", 0)
                                height = obj.get("/Height", 0)
                                filter_type = obj.get("/Filter", "")

                                if isinstance(filter_type, list):
                                    filter_type = filter_type[0] if filter_type else ""

                                if filter_type == "/DCTDecode":
                                    img_format = "JPEG"
                                    extension = "jpg"
                                elif filter_type == "/JPXDecode":
                                    img_format = "JPEG2000"
                                    extension = "jp2"
                                elif filter_type == "/CCITTFaxDecode":
                                    img_format = "TIFF"
                                    extension = "tiff"
                                else:
                                    img_format = "PNG"
                                    extension = "png"

                                try:
                                    pil_img = PILImage.open(io.BytesIO(img_data))
                                    if pil_img.mode not in ('RGB', 'L'):
                                        pil_img = pil_img.convert('RGB')
                                    img_buffer = io.BytesIO()
                                    pil_img.save(img_buffer, format='PNG')
                                    img_bytes = img_buffer.getvalue()
                                    extension = "png"
                                except Exception:
                                    img_bytes = img_data

                                if output_format == "base64":
                                    encoded = base64.b64encode(img_bytes).decode('ascii')
                                    images_result.append({
                                        "page_number": page_idx + 1,
                                        "image_index": len(images_result),
                                        "name": str(obj_name),
                                        "format": extension,
                                        "width": width,
                                        "height": height,
                                        "size_bytes": len(img_bytes),
                                        "base64": f"data:image/{extension};base64,{encoded}"
                                    })
                                else:
                                    images_result.append({
                                        "page_number": page_idx + 1,
                                        "image_index": len(images_result),
                                        "name": str(obj_name),
                                        "format": extension,
                                        "width": width,
                                        "height": height,
                                        "size_bytes": len(img_bytes),
                                        "bytes": img_bytes
                                    })
                            except Exception as e:
                                continue
            except Exception:
                continue

    except ImportError:
        for page_idx, page in enumerate(reader.pages):
            try:
                resources = page["/Resources"]
                if "/XObject" in resources:
                    xobjects = resources["/XObject"]
                    for obj_name in xobjects:
                        obj = xobjects[obj_name]
                        if obj.get("/Subtype") == "/Image":
                            try:
                                img_data = obj.get_data()
                                width = obj.get("/Width", 0)
                                height = obj.get("/Height", 0)
                                extension = "bin"

                                if output_format == "base64":
                                    encoded = base64.b64encode(img_data).decode('ascii')
                                    images_result.append({
                                        "page_number": page_idx + 1,
                                        "image_index": len(images_result),
                                        "name": str(obj_name),
                                        "format": extension,
                                        "width": width,
                                        "height": height,
                                        "size_bytes": len(img_data),
                                        "base64": f"data:application/octet-stream;base64,{encoded}"
                                    })
                                else:
                                    images_result.append({
                                        "page_number": page_idx + 1,
                                        "image_index": len(images_result),
                                        "name": str(obj_name),
                                        "format": extension,
                                        "width": width,
                                        "height": height,
                                        "size_bytes": len(img_data),
                                        "bytes": img_data
                                    })
                            except Exception:
                                continue
            except Exception:
                continue

    return images_result


def merge_pdfs(
    input_pdfs: List[str],
    output_path: str,
    passwords: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    passwords = passwords or {}
    result = {
        "input_files": [],
        "output_path": str(Path(output_path).absolute()),
        "total_pages": 0,
        "success": False
    }

    try:
        writer = PdfWriter()

        for pdf_path in input_pdfs:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                result["input_files"].append({
                    "path": str(pdf_file.absolute()),
                    "pages": 0,
                    "error": "File not found"
                })
                continue

            try:
                reader = PdfReader(str(pdf_file))
                if reader.is_encrypted:
                    pw = passwords.get(pdf_path) or passwords.get(pdf_file.name)
                    if not pw:
                        result["input_files"].append({
                            "path": str(pdf_file.absolute()),
                            "pages": 0,
                            "error": "Password required"
                        })
                        continue
                    reader.decrypt(pw)

                for page in reader.pages:
                    writer.add_page(page)

                result["input_files"].append({
                    "path": str(pdf_file.absolute()),
                    "pages": len(reader.pages),
                    "error": None
                })
                result["total_pages"] += len(reader.pages)

            except Exception as e:
                result["input_files"].append({
                    "path": str(pdf_file.absolute()),
                    "pages": 0,
                    "error": str(e)
                })

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as f:
            writer.write(f)

        result["success"] = True
        result["output_size_bytes"] = output_file.stat().st_size

    except Exception as e:
        result["error"] = f"Failed to merge PDFs: {str(e)}"
        result["success"] = False

    return result


def crop_pdf_page(
    page,
    left: Optional[float] = None,
    right: Optional[float] = None,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    percentages: bool = False
):
    media_box = page.mediabox
    width = float(media_box.width)
    height = float(media_box.height)

    if percentages:
        if left is not None:
            left = width * left / 100
        if right is not None:
            right = width * right / 100
        if top is not None:
            top = height * top / 100
        if bottom is not None:
            bottom = height * bottom / 100

    new_left = float(media_box.left) + (left or 0)
    new_right = float(media_box.right) - (right or 0)
    new_bottom = float(media_box.bottom) + (bottom or 0)
    new_top = float(media_box.top) - (top or 0)

    new_left = max(new_left, float(media_box.left))
    new_right = min(new_right, float(media_box.right))
    new_bottom = max(new_bottom, float(media_box.bottom))
    new_top = min(new_top, float(media_box.top))

    if new_left >= new_right or new_bottom >= new_top:
        raise ValueError("Invalid crop dimensions resulting in zero or negative size")

    page.cropbox = RectangleObject([new_left, new_bottom, new_right, new_top])
    return page


def crop_pdf(
    input_path: str,
    output_path: str,
    left: Optional[float] = None,
    right: Optional[float] = None,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    percentages: bool = False,
    pages: Optional[List[int]] = None,
    password: Optional[str] = None
) -> Dict[str, Any]:
    result = {
        "input_path": str(Path(input_path).absolute()),
        "output_path": str(Path(output_path).absolute()),
        "crop_settings": {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "percentages": percentages,
            "pages": pages
        },
        "pages_cropped": 0,
        "success": False
    }

    try:
        input_file = Path(input_path)
        if not input_file.exists():
            result["error"] = f"Input file not found: {input_path}"
            return result

        reader = PdfReader(str(input_file))
        if reader.is_encrypted:
            if not password:
                result["error"] = "Password required for encrypted PDF"
                return result
            reader.decrypt(password)

        writer = PdfWriter()
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            should_crop = pages is None or page_num in pages

            if should_crop:
                try:
                    cropped_page = crop_pdf_page(page, left, right, top, bottom, percentages)
                    writer.add_page(cropped_page)
                    result["pages_cropped"] += 1
                except Exception as e:
                    result[f"page_{page_num}_error"] = str(e)
                    writer.add_page(page)
            else:
                writer.add_page(page)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as f:
            writer.write(f)

        result["total_pages"] = total_pages
        result["output_size_bytes"] = output_file.stat().st_size
        result["success"] = True

    except Exception as e:
        result["error"] = f"Failed to crop PDF: {str(e)}"
        result["success"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="PDF Toolkit - Extract, merge, crop PDFs with OCR support")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    extract_parser = subparsers.add_parser("extract", help="Extract metadata, text, and images from PDF")
    extract_parser.add_argument("pdf_path", help="Path to the PDF file")
    extract_parser.add_argument("-p", "--password", help="Password for encrypted PDF", default=None)
    extract_parser.add_argument("-o", "--output", help="Output JSON file path", default=None)
    extract_parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback for scanned PDFs")
    extract_parser.add_argument("--ocr-lang", default="eng+chi_sim", help="OCR language(s)")
    extract_parser.add_argument("--ocr-dpi", type=int, default=300, help="OCR DPI")
    extract_parser.add_argument("--ocr-threshold", type=int, default=10, help="OCR fallback threshold")
    extract_parser.add_argument("--extract-images", action="store_true", help="Extract images as Base64")

    merge_parser = subparsers.add_parser("merge", help="Merge multiple PDFs into one")
    merge_parser.add_argument("inputs", nargs="+", help="Input PDF files to merge")
    merge_parser.add_argument("-o", "--output", required=True, help="Output PDF file path")
    merge_parser.add_argument("-p", "--passwords", nargs="*", help="Passwords for encrypted PDFs (format: filename:password)")

    crop_parser = subparsers.add_parser("crop", help="Crop PDF pages")
    crop_parser.add_argument("input", help="Input PDF file")
    crop_parser.add_argument("-o", "--output", required=True, help="Output PDF file")
    crop_parser.add_argument("-l", "--left", type=float, help="Crop from left (points or percentage)")
    crop_parser.add_argument("-r", "--right", type=float, help="Crop from right (points or percentage)")
    crop_parser.add_argument("-t", "--top", type=float, help="Crop from top (points or percentage)")
    crop_parser.add_argument("-b", "--bottom", type=float, help="Crop from bottom (points or percentage)")
    crop_parser.add_argument("--percentages", action="store_true", help="Use percentages instead of points")
    crop_parser.add_argument("--pages", nargs="+", type=int, help="Specific pages to crop (1-based)")
    crop_parser.add_argument("-p", "--password", help="Password for encrypted PDF", default=None)

    parser.add_argument("--check-ocr", action="store_true", help="Check OCR availability and exit")

    args = parser.parse_args()

    if args.check_ocr:
        available, error = ocr_available()
        print(f"OCR Available: {available}")
        if available:
            try:
                import pytesseract
                langs = pytesseract.get_languages(config='')
                print(f"Installed languages: {', '.join(langs)}")
                print(f"Tesseract path: {TESSERACT_CMD}")
                print(f"Poppler path: {POPPLER_PATH}")
            except Exception as e:
                print(f"Error checking languages: {e}")
        else:
            print(f"Error: {error}")
            print()
            print("Installation instructions:")
            print("1. Install Tesseract OCR:")
            print("   - Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("   - Or run: winget install UB-Mannheim.TesseractOCR")
            print("2. Install Poppler (for PDF to image conversion):")
            print("   - Download from: https://github.com/oschwartz10612/poppler-windows/releases")
            print("   - Or run: winget install oschwartz10612.poppler")
            print("3. Set environment variables (if not in PATH):")
            print("   - set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
            print("   - set POPPLER_PATH=C:\\Program Files\\poppler\\bin")
        return

    if args.command is None:
        parser.print_help()
        return

    if args.command == "extract":
        result = extract_pdf(
            args.pdf_path,
            password=args.password,
            ocr_enabled=not args.no_ocr,
            ocr_lang=args.ocr_lang,
            ocr_dpi=args.ocr_dpi,
            ocr_fallback_threshold=args.ocr_threshold,
            extract_images=args.extract_images
        )
        json_output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"Result saved to: {output_path.absolute()}")
        else:
            print(json_output)

    elif args.command == "merge":
        password_dict = {}
        if args.passwords:
            for pw_entry in args.passwords:
                if ":" in pw_entry:
                    fname, pw = pw_entry.split(":", 1)
                    password_dict[fname] = pw
        
        result = merge_pdfs(args.inputs, args.output, password_dict)
        json_output = json.dumps(result, ensure_ascii=False, indent=2)
        print(json_output)

    elif args.command == "crop":
        pages_list = args.pages if args.pages else None
        result = crop_pdf(
            args.input,
            args.output,
            left=args.left,
            right=args.right,
            top=args.top,
            bottom=args.bottom,
            percentages=args.percentages,
            pages=pages_list,
            password=args.password
        )
        json_output = json.dumps(result, ensure_ascii=False, indent=2)
        print(json_output)


if __name__ == "__main__":
    main()
