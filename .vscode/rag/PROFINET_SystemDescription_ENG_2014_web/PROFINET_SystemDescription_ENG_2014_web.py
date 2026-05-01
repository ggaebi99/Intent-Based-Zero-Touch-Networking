import fitz  # PyMuPDF
import re
import json
import tiktoken


############################################################
# 1) PROFINET PDF 섹션 범위 정의 (너가 원하는 섹션만 넣으면 됨)
############################################################

SECTION_RANGES = {
    "BasicFunctions": (9, 11),      # 예시: 3장
    "IRT": (12, 15),                # 5장
    "OptionalRT": (14, 18),         # 6장
    "Profiles": (18, 18),           # 8장
    "Security": (21, 21)            # 10.4
}
# → 나중에 네가 실제 페이지 숫자만 수정하면 그대로 적용됨


############################################################
# 2) 페이지별 텍스트 추출
############################################################

############################################################
# 2) 페이지별 텍스트 추출
############################################################

def extract_selected_sections(pdf_path, ranges):
    doc = fitz.open(pdf_path)
    extracted = []

    for sec_name, (start, end) in ranges.items():
        text = ""

        # 단일 페이지 구간 처리
        if start == end:
            text += doc[start].get_text("text") + "\n"

        else:
            # 기존 다중 페이지 추출
            for page_num in range(start, end):
                text += doc[page_num].get_text("text") + "\n"

        extracted.append((sec_name, text))

    doc.close()
    return extracted


############################################################
# 3) 텍스트 정제 (PROFINET 전용)
############################################################

def clean_text(text):

    # --- PROFINET 반복 헤더/푸터 제거 ---

    # 형태 1) PROFINET System Description 4
    text = re.sub(r"\bPROFINET System Description\b\s*\d*", "", text)

    # 형태 2) 4 PROFINET System Description
    text = re.sub(r"\d+\s*\bPROFINET System Description\b", "", text)

    # 형태 3) PROFINET System Description
    text = re.sub(r"\bPROFINET System Description\b", "", text)


    # --- 페이지 번호 제거 ---
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)


    # --- 과도한 공백/줄 정리 ---
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Ligature 문자 정리
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

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
        sents = re.split(r'(?<=[.!?])\s+', para)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)
    return sentences


############################################################
# 6) Sentence Chunking
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

        if tok > chunk_size:
            chunks.append(sent)
            i += 1
            continue

        if current_tokens + tok > chunk_size:
            chunks.append(" ".join(current_chunk))

            overlap = []
            overlap_tokens = 0

            j = len(current_chunk) - 1
            while j >= 0:
                prev_sent = current_chunk[j]
                t = len(enc.encode(prev_sent))
                if overlap_tokens + t > window_size:
                    break
                overlap.insert(0, prev_sent)
                overlap_tokens += t
                j -= 1

            current_chunk = overlap.copy()
            current_tokens = overlap_tokens

        current_chunk.append(sent)
        current_tokens += tok
        i += 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    final_chunks = []
    for idx, ch in enumerate(chunks):
        final_chunks.append({
            "id": f"{sec_name}_{idx}",
            "prefix": f"[PROFINET - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "PROFINET",
                "section": sec_name,
                "chunk_id": idx
            }
        })

    return final_chunks


############################################################
# 7) MAIN PIPELINE
############################################################

pdf_path = "D:/건대/12~1논문/251216/Docs/123/PROFINET_SystemDescription_ENG_2014_web.pdf"
output_file = "PROFINET_selected_chunks.jsonl"
text_output_file = "PROFINET_selected_text.txt"

sections = extract_selected_sections(pdf_path, SECTION_RANGES)

all_chunks = []

with open(text_output_file, "w", encoding="utf-8") as text_f:
    for sec_name, raw_text in sections:
        cleaned = clean_text(raw_text)
        paras = merge_paragraphs(cleaned)
        sentences = split_into_sentences(paras)

        # 텍스트 파일 저장
        text_f.write(f"<<{sec_name}>>\n")
        for s in sentences:
            text_f.write(s + "\n")
        text_f.write("\n\n")

        # Chunk 생성
        chunks = sentence_chunk_with_window(sentences, sec_name)
        all_chunks.extend(chunks)

# JSONL 저장
with open(output_file, "w", encoding="utf-8") as f:
    for ch in all_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print("Done! Total chunks:", len(all_chunks))
print("Text-only file:", text_output_file)
