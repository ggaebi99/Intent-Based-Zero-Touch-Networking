import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import json
import tiktoken

############################################################
# 0) Tesseract 경로 (네 환경용)
############################################################

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


############################################################
# 1) 파일 경로
############################################################

pdf_path = "D:/건대/12~1논문/251216/Docs/123/ISO-13849-1-2023.pdf"
raw_ocr_file = "ISO13849_raw_OCR_text.txt"
output_text_file = "ISO13849_selected_text.txt"
output_json_file = "ISO13849_selected_chunks.jsonl"


############################################################
# 2) 사용자가 제공한 정확한 Section PAGE Ranges
############################################################

SECTION_RANGES = {
    "Section3": (12, 22),
    "Section4": (22, 27),
    "Section5": (27, 37),
    "Section6": (37, 57),
    "AnnexA":   (81, 85),
    "AnnexB":   (86, 87),
    "AnnexE":   (98, 101)
}


############################################################
# 3) OCR 수행 (PDF → 이미지 → Tesseract)
############################################################

def extract_text_by_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        text = pytesseract.image_to_string(img, lang="eng")

        all_text.append(f"\n\n=========== PAGE {page_num} ===========\n")
        all_text.append(text)

    doc.close()
    return "".join(all_text)


############################################################
# 4) OCR 실행 & 저장
############################################################

raw_text = extract_text_by_ocr(pdf_path)

with open(raw_ocr_file, "w", encoding="utf-8") as f:
    f.write(raw_text)

print("OCR COMPLETE → saved:", raw_ocr_file)


############################################################
# 5) OCR 결과 불러오기
############################################################

with open(raw_ocr_file, "r", encoding="utf-8") as f:
    full_text = f.read()


############################################################
# 6) 페이지 단위로 Split
############################################################

pages = re.split(r"=+\s*PAGE\s+\d+\s*=+", full_text)
print("Total OCR Pages:", len(pages))


############################################################
# 7) Section 범위 기반 페이지 조각 추출
############################################################

def extract_section_by_range(pages, ranges):
    results = {}
    for sec_name, (start, end) in ranges.items():
        section_text = "\n".join(pages[start:end])
        results[sec_name] = section_text
    return results

raw_sections = extract_section_by_range(pages, SECTION_RANGES)


############################################################
# 8) ISO 특정 노이즈 제거
############################################################

def clean_iso_noise(text):
    # ISO header/footer 제거
    text = re.sub(r"©\s*ISO\s*2023\s*-\s*All rights reserved", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ISO\s*13849-1:2023\(E\)", "", text, flags=re.IGNORECASE)

    # OCR 잡음 제거
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    # 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


############################################################
# 9) 문단 병합
############################################################

def merge_paragraphs(text):
    lines = text.split("\n")
    paras = []
    buf = ""

    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                paras.append(buf)
                buf = ""
        else:
            buf += (" " + line) if buf else line

    if buf:
        paras.append(buf)

    return paras


############################################################
# 10) 문장 분리
############################################################

def split_into_sentences(paragraphs):
    sentences = []
    for para in paragraphs:
        sents = re.split(r'(?<=[.!?])\s+', para)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)
    return sentences


############################################################
# 11) Sliding-window Chunking
############################################################

def sentence_chunk_with_window(sentences, sec_name,
                               chunk_size=300, window_size=100):

    enc = tiktoken.get_encoding("cl100k_base")

    chunks = []
    current = []
    cur_tok = 0

    sent_tokens = [(s, len(enc.encode(s))) for s in sentences]

    i = 0
    while i < len(sent_tokens):
        sent, tok = sent_tokens[i]

        if tok > chunk_size:
            chunks.append(sent)
            i += 1
            continue

        if cur_tok + tok > chunk_size:
            chunks.append(" ".join(current))

            # sliding window overlap 유지
            overlap = []
            overlap_tok = 0

            j = len(current) - 1
            while j >= 0:
                prev = current[j]
                prev_tok = len(enc.encode(prev))
                if overlap_tok + prev_tok > window_size:
                    break
                overlap.insert(0, prev)
                overlap_tok += prev_tok
                j -= 1

            current = overlap
            cur_tok = overlap_tok

        current.append(sent)
        cur_tok += tok
        i += 1

    if current:
        chunks.append(" ".join(current))

    final = []
    for idx, ch in enumerate(chunks):
        final.append({
            "id": f"{sec_name}_{idx}",
            "prefix": f"[ISO 13849 - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "ISO13849-1",
                "section": sec_name,
                "chunk_id": idx
            }
        })
    return final


############################################################
# 12) MAIN PIPELINE — 정제 + Chunking
############################################################

all_chunks = []

with open(output_text_file, "w", encoding="utf-8") as out:

    for sec_name, text in raw_sections.items():

        cleaned = clean_iso_noise(text)
        paras = merge_paragraphs(cleaned)
        sentences = split_into_sentences(paras)

        # text-only 저장
        out.write(f"<<{sec_name}>>\n")
        for s in sentences:
            out.write(s + "\n")
        out.write("\n\n")

        # chunk 생성
        chunks = sentence_chunk_with_window(sentences, sec_name)
        all_chunks.extend(chunks)


############################################################
# 13) JSONL 저장
############################################################

with open(output_json_file, "w", encoding="utf-8") as f:
    for ch in all_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")


print("==== DONE ====")
print("Text saved  →", output_text_file)
print("Chunks saved →", output_json_file)
print("Total chunks:", len(all_chunks))
