from pdf_extractor import (
    extract_pdf,
    merge_pdfs,
    crop_pdf,
    extract_images_from_pdf
)
import json
import os


def test_extract_with_images():
    print("=" * 60)
    print("TEST 1: Extract PDF with images")
    print("=" * 60)
    
    result = extract_pdf("test_scanned.pdf", extract_images=True)
    print(f"Success: {result.get('success')}")
    print(f"Pages: {result['metadata']['page_count']}")
    print(f"Images extracted: {len(result.get('images', []))}")
    
    if result['images']:
        img = result['images'][0]
        print(f"First image - Page: {img['page_number']}, Format: {img['format']}")
        print(f"  Size: {img['width']}x{img['height']}, Bytes: {img['size_bytes']}")
        print(f"  Base64 starts with: {img['base64'][:50]}...")
    else:
        print("  No embedded images found (expected for scanned PDF)")
    print()


def test_merge_pdfs():
    print("=" * 60)
    print("TEST 2: Merge PDFs")
    print("=" * 60)
    
    result = merge_pdfs(
        ["test_normal.pdf", "test_scanned.pdf"],
        "merged_output.pdf"
    )
    print(f"Success: {result.get('success')}")
    print(f"Total pages merged: {result.get('total_pages')}")
    print(f"Output file: {result.get('output_path')}")
    print(f"Output size: {result.get('output_size_bytes', 0)} bytes")
    
    for input_file in result.get('input_files', []):
        status = "OK" if input_file['error'] is None else "ERROR"
        print(f"  [{status}] {os.path.basename(input_file['path'])}: {input_file['pages']} pages")
    
    if os.path.exists("merged_output.pdf"):
        from pypdf import PdfReader
        reader = PdfReader("merged_output.pdf")
        print(f"Verification: Merged PDF has {len(reader.pages)} pages")
    print()


def test_crop_pdf():
    print("=" * 60)
    print("TEST 3: Crop PDF (percentages)")
    print("=" * 60)
    
    result = crop_pdf(
        "test_normal.pdf",
        "cropped_output.pdf",
        left=10,
        right=10,
        top=10,
        bottom=10,
        percentages=True
    )
    print(f"Success: {result.get('success')}")
    print(f"Pages cropped: {result.get('pages_cropped')}")
    print(f"Output file: {result.get('output_path')}")
    print(f"Output size: {result.get('output_size_bytes', 0)} bytes")
    
    if os.path.exists("cropped_output.pdf"):
        from pypdf import PdfReader
        reader = PdfReader("cropped_output.pdf")
        page = reader.pages[0]
        print(f"Verification: Cropped page size: {float(page.cropbox.width):.1f} x {float(page.cropbox.height):.1f} points")
    print()


def test_crop_specific_pages():
    print("=" * 60)
    print("TEST 4: Crop specific pages only")
    print("=" * 60)
    
    result = crop_pdf(
        "test_normal.pdf",
        "cropped_pages_1_3.pdf",
        left=50,
        right=50,
        top=50,
        bottom=50,
        percentages=False,
        pages=[1, 3]
    )
    print(f"Success: {result.get('success')}")
    print(f"Pages cropped: {result.get('pages_cropped')}")
    print(f"Output file: {result.get('output_path')}")
    
    if os.path.exists("cropped_pages_1_3.pdf"):
        from pypdf import PdfReader
        reader = PdfReader("cropped_pages_1_3.pdf")
        for i, page in enumerate(reader.pages):
            print(f"  Page {i+1}: {float(page.cropbox.width):.1f} x {float(page.cropbox.height):.1f} points")
    print()


def test_merge_with_password():
    print("=" * 60)
    print("TEST 5: Merge including encrypted PDF")
    print("=" * 60)
    
    result = merge_pdfs(
        ["test_normal.pdf", "test_protected.pdf"],
        "merged_with_protected.pdf",
        passwords={"test_protected.pdf": "test123"}
    )
    print(f"Success: {result.get('success')}")
    print(f"Total pages merged: {result.get('total_pages')}")
    
    for input_file in result.get('input_files', []):
        status = "OK" if input_file['error'] is None else "ERROR"
        print(f"  [{status}] {os.path.basename(input_file['path'])}: {input_file['pages']} pages")
    print()


def test_command_line_help():
    print("=" * 60)
    print("TEST 6: Command line interface")
    print("=" * 60)
    print("Available commands: extract, merge, crop")
    print()
    print("Extract usage:")
    print("  python pdf_extractor.py extract document.pdf")
    print("  python pdf_extractor.py extract document.pdf --extract-images")
    print()
    print("Merge usage:")
    print("  python pdf_extractor.py merge file1.pdf file2.pdf -o merged.pdf")
    print()
    print("Crop usage:")
    print("  python pdf_extractor.py crop input.pdf -o output.pdf -l 10 -r 10 -t 10 -b 10 --percentages")
    print()


def test_image_extraction_api():
    print("=" * 60)
    print("TEST 7: Direct image extraction API")
    print("=" * 60)
    
    try:
        images = extract_images_from_pdf("test_scanned.pdf")
        print(f"Images found: {len(images)}")
        if images:
            for img in images[:2]:
                print(f"  Page {img['page_number']}: {img['width']}x{img['height']} ({img['format']})")
    except Exception as e:
        print(f"Error: {e}")
    print()


if __name__ == "__main__":
    print("PDF Toolkit - New Features Test Suite")
    print("=" * 60)
    print()
    
    test_extract_with_images()
    test_merge_pdfs()
    test_crop_pdf()
    test_crop_specific_pages()
    test_merge_with_password()
    test_command_line_help()
    test_image_extraction_api()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
