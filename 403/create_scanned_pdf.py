from pypdf import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import os


def create_text_image(text, width=800, height=1000, font_size=36):
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    
    y = 100
    lines = text.split('\n')
    for line in lines:
        draw.text((50, y), line, fill='black', font=font)
        y += font_size + 10
    
    return img


def create_scanned_pdf(output_path):
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    c.setTitle("Scanned Document Test")
    c.setAuthor("Test Author")
    
    page_texts = [
        "Page 1 - Scanned Document\n\nThis is a scanned PDF.\nIt contains only images,\nno selectable text layer.",
        "Page 2 - Second Page\n\nThe quick brown fox\njumps over the lazy dog.\n\n1234567890",
        "Page 3 - Final Page\n\nScanned documents require\nOCR to extract text.\n\nTesting OCR fallback."
    ]
    
    for i, text in enumerate(page_texts):
        img = create_text_image(text)
        img_path = f"temp_page_{i+1}.png"
        img.save(img_path)
        
        c.drawImage(img_path, 50, 100, width=500, height=650)
        c.showPage()
        
        os.remove(img_path)
    
    c.save()
    
    packet.seek(0)
    with open(output_path, 'wb') as f:
        f.write(packet.getvalue())
    
    print(f"Created scanned PDF: {output_path}")
    print("This PDF contains only images, no text layer.")


if __name__ == "__main__":
    create_scanned_pdf("test_scanned.pdf")
