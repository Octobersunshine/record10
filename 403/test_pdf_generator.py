from pypdf import PdfWriter, PdfReader
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def create_test_pdf(output_path, password=None):
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setTitle("Test Document")
    can.setAuthor("John Doe")
    can.setSubject("PDF Extraction Test")
    
    can.drawString(100, 750, "Page 1 - Hello World")
    can.drawString(100, 730, "This is a test PDF document for extraction testing.")
    can.drawString(100, 710, "It contains multiple pages with sample text.")
    can.showPage()
    
    can.drawString(100, 750, "Page 2 - Second Page")
    can.drawString(100, 730, "This is the second page of the test document.")
    can.drawString(100, 710, "The quick brown fox jumps over the lazy dog.")
    can.showPage()
    
    can.drawString(100, 750, "Page 3 - Third Page")
    can.drawString(100, 730, "Final page of the test document.")
    can.drawString(100, 710, "1234567890 !@#$%^&*()")
    can.save()
    
    packet.seek(0)
    new_pdf = PdfReader(packet)
    output = PdfWriter()
    
    for page in new_pdf.pages:
        output.add_page(page)
    
    output.add_metadata({
        "/Title": "Test Document",
        "/Author": "John Doe",
        "/Subject": "PDF Extraction Test",
        "/Creator": "Test Script",
        "/Producer": "pypdf + reportlab",
    })
    
    if password:
        output.encrypt(user_password=password, owner_password=password, use_128bit=True)
    
    with open(output_path, "wb") as f:
        output.write(f)


if __name__ == "__main__":
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        create_test_pdf("test_normal.pdf")
        print("Created test_normal.pdf")
        create_test_pdf("test_protected.pdf", password="test123")
        print("Created test_protected.pdf with password 'test123'")
    except ImportError:
        print("reportlab not installed, skipping test PDF generation")
        print("Install with: pip install reportlab")
