from pdf_extractor import extract_pdf, ocr_available, check_ocr_languages
import json


def example_check_ocr():
    available, error = ocr_available()
    print("=== OCR Status Check ===")
    print(f"OCR Available: {available}")
    if available:
        langs_ok = check_ocr_languages("eng+chi_sim")
        print(f"Chinese + English language pack available: {langs_ok}")
    else:
        print(f"Error: {error}")
    print()


def example_normal_pdf():
    result = extract_pdf("test_normal.pdf")
    print("=== Normal PDF (with text layer) ===")
    print(f"Success: {result.get('success')}")
    print(f"OCR Used: {result['ocr']['used']}")
    print(f"Pages: {result['metadata']['page_count']}")
    print(f"Extraction method: {result['pages'][0]['extraction_method']}")
    print(f"Title: {result['metadata']['title']}")
    print(f"Author: {result['metadata']['author']}")
    print(f"Total characters: {result['total_character_count']}")
    print(f"First page preview: {result['pages'][0]['text'][:60]}...")
    print()


def example_scanned_pdf_no_ocr():
    result = extract_pdf("test_scanned.pdf", ocr_enabled=False)
    print("=== Scanned PDF (OCR disabled) ===")
    print(f"Success: {result.get('success')}")
    print(f"OCR Enabled: {result['ocr']['enabled']}")
    print(f"Total characters: {result['total_character_count']}")
    print(f"Text empty: {all(p['character_count'] == 0 for p in result['pages'])}")
    print()


def example_scanned_pdf_with_ocr():
    available, _ = ocr_available()
    print("=== Scanned PDF (OCR enabled) ===")
    if not available:
        print("OCR not available - showing simulated result:")
        print("  Would automatically detect no text layer")
        print("  Would fall back to OCR extraction")
        print("  Would extract text from page images")
        result = extract_pdf("test_scanned.pdf")
        if 'warning' in result:
            print(f"Warning: {result['warning'][:100]}...")
    else:
        result = extract_pdf("test_scanned.pdf")
        print(f"Success: {result.get('success')}")
        print(f"OCR Used: {result['ocr']['used']}")
        print(f"Fallback reason: {result['ocr'].get('fallback_reason', 'N/A')}")
        print(f"Extraction method: {result['pages'][0]['extraction_method']}")
        print(f"Total characters: {result['total_character_count']}")
        print(f"First page preview: {result['pages'][0]['text'][:60]}...")
    print()


def example_ocr_config():
    print("=== OCR Configuration Options ===")
    print("1. Disable OCR:")
    print("   result = extract_pdf('doc.pdf', ocr_enabled=False)")
    print()
    print("2. Specify language:")
    print("   result = extract_pdf('doc.pdf', ocr_lang='eng')  # English only")
    print("   result = extract_pdf('doc.pdf', ocr_lang='chi_sim')  # Chinese simplified")
    print("   result = extract_pdf('doc.pdf', ocr_lang='eng+chi_sim')  # Both")
    print()
    print("3. Adjust DPI (higher = better quality but slower):")
    print("   result = extract_pdf('doc.pdf', ocr_dpi=200)  # Faster")
    print("   result = extract_pdf('doc.pdf', ocr_dpi=400)  # Better quality")
    print()
    print("4. Adjust text layer detection threshold:")
    print("   result = extract_pdf('doc.pdf', ocr_fallback_threshold=50)")
    print()


def example_protected_pdf():
    available, _ = ocr_available()
    print("=== Protected PDF ===")
    result = extract_pdf("test_protected.pdf", password="test123")
    print(f"Success: {result.get('success')}")
    print(f"Pages: {result['metadata']['page_count']}")
    print(f"OCR Used: {result['ocr']['used']}")
    if available:
        print("Note: OCR works with password-protected PDFs too!")
    print()


def example_check_ocr_cmd():
    print("=== Command Line Usage ===")
    print("Check OCR status:")
    print("  python pdf_extractor.py --check-ocr")
    print()
    print("Extract with OCR (default):")
    print("  python pdf_extractor.py document.pdf")
    print()
    print("Extract without OCR:")
    print("  python pdf_extractor.py document.pdf --no-ocr")
    print()
    print("Extract with custom OCR settings:")
    print("  python pdf_extractor.py document.pdf --ocr-lang eng --ocr-dpi 200")
    print()
    print("Extract protected PDF:")
    print("  python pdf_extractor.py document.pdf -p password")
    print()


if __name__ == "__main__":
    example_check_ocr()
    example_normal_pdf()
    example_scanned_pdf_no_ocr()
    example_scanned_pdf_with_ocr()
    example_ocr_config()
    example_protected_pdf()
    example_check_ocr_cmd()
    print("All examples completed!")
