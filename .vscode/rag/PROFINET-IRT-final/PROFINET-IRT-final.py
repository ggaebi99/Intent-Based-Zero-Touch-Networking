import fitz  # PyMuPDF
import re

PDF_PATH = "D:/건대/12~1논문/251216/Docs/PROFINET-IRT-final.pdf"
OUTPUT_TEXT = "PROFINET_IRT_cleaned_text.txt"


############################################################
# 1) 기본 노이즈 패턴 정의
############################################################

NOISE_PATTERNS = [
    r"Imprint\s*PI North America\s*www\.us\.profinet\.com\s*Doc 10-113 v1",
    r"PROFINET System Description\s*\d*",   # 반복 헤더 (왼/오른쪽 모두 제거)
    r"PROFINET System Description",         # 순수 텍스트 등장도 제거
    r"^\s*\d+\s*$",                          # 페이지 하단 숫자 단독 (5, 6, 7...)
]


############################################################
# 2) 노이즈 제거 함수
############################################################

def clean_text(text):
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    
    # 중복 개행 정리
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


############################################################
# 3) PDF → 텍스트 추출
############################################################

def extract_all_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # 페이지 번호 섹션 태그는 유지 (나중에 범위 분석할 때 필요)
        cleaned = clean_text(text)
        full_text.append(f"===== PAGE {page_num} =====\n{cleaned}\n")

    doc.close()
    return "\n".join(full_text)


raw = extract_all_text(PDF_PATH)

with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
    f.write(raw)

print("완료! Cleaned 텍스트 저장됨 →", OUTPUT_TEXT)
