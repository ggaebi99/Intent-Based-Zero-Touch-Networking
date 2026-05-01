import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# ✔ 만약 PATH가 자동 등록 안 되면 아래 경로를 직접 지정
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf_path = "D:/건대/12~1논문/251216/Docs/123/ISO-13849-1-2023.pdf"
output_text_file = "ISO13849_raw_OCR_text.txt"

def extract_text_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # PDF 페이지를 이미지로 렌더링
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # OCR 수행
        text = pytesseract.image_to_string(img, lang="eng")

        all_text.append(f"\n\n=========== PAGE {page_num} ===========\n")
        all_text.append(text)

    doc.close()
    return "".join(all_text)


# 실행
raw_text = extract_text_ocr(pdf_path)

with open(output_text_file, "w", encoding="utf-8") as f:
    f.write(raw_text)

print("OCR text extracted →", output_text_file)
