import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re

# === 필요한 경우 Tesseract 위치 지정 ===
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf_path = "D:/건대/12~1논문/251216/Docs/123/IEC-61508-Overview.pdf"
output_text_file = "IEC61508_raw_text.txt"

def extract_page_text(page):
    """텍스트 PDF면 text 추출, 아니면 OCR 수행"""
    text = page.get_text("text")

    # 텍스트가 너무 짧으면 OCR로 대체
    if len(text.strip()) < 20:  
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng")
    
    return text


def extract_pdf_all_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        text = extract_page_text(page)

        # 페이지 헤더
        all_text.append(f"\n\n=========== PAGE {page_num} ===========\n")
        all_text.append(text)

    doc.close()
    return "".join(all_text)


# 실행
raw_text = extract_pdf_all_text(pdf_path)

# 저장
with open(output_text_file, "w", encoding="utf-8") as f:
    f.write(raw_text)

print("완료! → 전체 페이지 텍스트 저장됨")
print("파일:", output_text_file)
