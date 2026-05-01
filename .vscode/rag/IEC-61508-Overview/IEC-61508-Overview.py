import fitz  # PyMuPDF
import re
import json
import tiktoken


############################################################
# 1) PDF 전체 추출 범위
#
# ※ 페이지는 0~28 (총 29페이지)
#   전체 사용이 최적 → Section은 1개로 처리
############################################################

SECTION_RANGES = {
    "IEC61508.Overview": (0, 29)
}


############################################################
# 2) 페이지별 텍스트 추출
############################################################

def extract_selected_sections(pdf_path, ranges):
    doc = fitz.open(pdf_path)
    extracted = []

    for sec_name, (start, end) in ranges.items():
        text = ""
        for page_num in range(start, end):
            text += doc[page_num].get_text("text") + "\n"
        extracted.append((sec_name, text))

    doc.close()
    return extracted


############################################################
# 3) 텍스트 정제 (IEC exida 노이즈 제거)
############################################################

def clean_text(text):

    # exida header/footer 제거
    text = re.sub(r"©\s*exida", "", text)
    text = re.sub(r"IEC 61508 Overview Report, Version.*?\d{4}", "", text)

    # "Page X of 29" 제거
    text = re.sub(r"Page\s+\d+\s+of\s+29", "", text)

    # 중복 공백
    text = re.sub(r"[ \t]+", " ", text)

    # 빈 줄 2개 이상 → 1개로
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


############################################################
# 4) 문단 병합
############################################################

def merge_paragraphs(text):
    lines = text.split("\n")
    paras = []
    buf = ""

    for line in lines:
        line = line.strip()
        if line == "":
            if buf:
                paras.append(buf)
                buf = ""
        else:
            buf += " " + line if buf else line

    if buf:
        paras.append(buf)

    return [p for p in paras if p.strip()]


############################################################
# 5) 문장 분해
############################################################

def split_into_sentences(paragraphs):
    sentences = []
    for para in paragraphs:
        sents = re.split(r'(?<=[\.!?])\s+', para)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)
    return sentences


############################################################
# 6) Sliding-window chunking
############################################################

def sentence_chunk_with_window(sentences, sec_name, chunk_size=300, window_size=100):

    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = []
    current_tokens = 0

    sent_tokens = [(s, len(enc.encode(s))) for s in sentences]

    i = 0
    while i < len(sent_tokens):
        sent, tok = sent_tokens[i]

        # 문장 하나가 큰 경우
        if tok > chunk_size:
            chunks.append(sent)
            i += 1
            continue

        # 초과하면 종료 후 슬라이딩 윈도우
        if current_tokens + tok > chunk_size:
            chunks.append(" ".join(current_chunk))

            # overlap 유지
            overlap = []
            overlap_tokens = 0

            j = len(current_chunk) - 1
            while j >= 0:
                prev = current_chunk[j]
                t = len(enc.encode(prev))
                if overlap_tokens + t > window_size:
                    break
                overlap.insert(0, prev)
                overlap_tokens += t
                j -= 1

            current_chunk = overlap.copy()
            current_tokens = overlap_tokens

        # 현재 chunk에 문장 추가
        current_chunk.append(sent)
        current_tokens += tok
        i += 1

    # 마지막 chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # JSON 포맷
    final_chunks = []
    for idx, ch in enumerate(chunks):
        final_chunks.append({
            "id": f"{sec_name}_{idx}",
            "prefix": f"[IEC 61508 Overview - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "IEC61508_Overview",
                "section": sec_name,
                "chunk_id": idx
            }
        })

    return final_chunks


############################################################
# 7) MAIN PIPELINE
############################################################

pdf_path = "D:/건대/12~1논문/251216/Docs/123/IEC-61508-Overview.pdf"

output_file = "IEC61508_selected_chunks.jsonl"
text_output_file = "IEC61508_selected_text.txt"

sections = extract_selected_sections(pdf_path, SECTION_RANGES)

all_chunks = []

with open(text_output_file, "w", encoding="utf-8") as text_f:

    for sec_name, raw_text in sections:

        cleaned = clean_text(raw_text)
        paras = merge_paragraphs(cleaned)
        sentences = split_into_sentences(paras)

        # 검증용 텍스트 파일
        text_f.write(f"<<{sec_name}>>\n")
        for s in sentences:
            text_f.write(s + "\n")
        text_f.write("\n\n")

        # chunk 만들기
        chunks = sentence_chunk_with_window(sentences, sec_name)
        all_chunks.extend(chunks)


# JSONL 저장
with open(output_file, "w", encoding="utf-8") as f:
    for ch in all_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print("Done! Total chunks:", len(all_chunks))
print("Text-only file:", text_output_file)
