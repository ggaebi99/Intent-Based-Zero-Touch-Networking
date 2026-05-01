import fitz  # PyMuPDF
import re
import json
import tiktoken


############################################################
# 0) 파일 경로 / 출력 파일
############################################################
PDF_PATH = "D:/건대/12~1논문/251216/Docs/EPSG_301_V-1-5-1_DS-c710608e.pdf"

OUTPUT_TEXT = "EPSG301_selected_text.txt"
OUTPUT_JSONL = "EPSG301_chunks.jsonl"


############################################################
# 1) PDF Section 범위 정의 (0-based, end는 exclusive)
############################################################
SECTION_RANGES = {
    # 4.2.4 POWERLINK Cycle (대략)
    "Section4.2.4": (39, 67),

    # 4.6~4.7 프레임/타이밍/에러 처리(대략)
    "Section4.6~4.7": (69, 111),

    # 7.2.1.5 Communication Profile Area - Cycle Timing 관련 오브젝트 정의(대략)
    "Section7.2.1.5": (218, 220),

    # 9.1.5 Security and Routing Type I (대략)
    "Section9.1.5": (313, 316),
}


############################################################
# 2) 페이지별 텍스트 추출
############################################################
def extract_selected_sections(pdf_path, ranges):
    doc = fitz.open(pdf_path)
    extracted = []

    for sec_name, (start, end) in ranges.items():
        # end가 start랑 같거나 작으면 최소 1페이지는 읽도록 보정
        if end <= start:
            end = start + 1

        text_parts = []
        for page_num in range(start, end):
            if 0 <= page_num < len(doc):
                text_parts.append(doc[page_num].get_text("text"))

        extracted.append((sec_name, "\n".join(text_parts)))

    doc.close()
    return extracted


############################################################
# 3) 텍스트 정제 (헤더/푸터/페이지번호 제거)
############################################################
def clean_text(text, remove_fig_table=False):
    # 문서 헤더/버전 제거 (반복 가능)
    text = re.sub(r"(?m)^\s*EPSG\s+DS\s+301\s+V[\d.]+\s*$", "", text)

    # 페이지 표기 제거: "-314-" 같은 형태
    text = re.sub(r"(?m)^\s*-\s*\d+\s*-\s*$", "", text)

    # 페이지 번호/챕터번호처럼 단독 숫자 라인 제거
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # (옵션) Figure/Table 캡션 제거
    if remove_fig_table:
        text = re.sub(r"(?m)^\s*Fig\.\s*\d+.*$", "", text)
        text = re.sub(r"(?m)^\s*Figure\s*\d+.*$", "", text)
        text = re.sub(r"(?m)^\s*Table\s*\d+.*$", "", text)

    # 공백 정리
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


############################################################
# 4) 문단 병합
############################################################
def merge_paragraphs(text):
    lines = text.split("\n")
    paras = []
    buf = []

    for line in lines:
        line = line.strip()
        if line == "":
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(line)

    if buf:
        paras.append(" ".join(buf))

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
# 6) Sliding-window Sentence Chunking
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

        # 너무 긴 문장은 단독 청크
        if tok > chunk_size:
            chunks.append(sent)
            i += 1
            continue

        # 청크 초과 시 저장 + overlap
        if current_tokens + tok > chunk_size and current_chunk:
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
            "prefix": f"[EPSG DS 301 - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "EPSG DS 301 V1.5.1",
                "section": sec_name,
                "chunk_id": idx
            }
        })

    return final_chunks


############################################################
# 7) MAIN PIPELINE
############################################################
def main():
    sections = extract_selected_sections(PDF_PATH, SECTION_RANGES)

    all_chunks = []

    with open(OUTPUT_TEXT, "w", encoding="utf-8") as text_f:
        for sec_name, raw_text in sections:
            cleaned = clean_text(raw_text, remove_fig_table=False)
            paras = merge_paragraphs(cleaned)
            sentences = split_into_sentences(paras)

            # text-only 파일 저장
            text_f.write(f"<<{sec_name}>>\n")
            for s in sentences:
                text_f.write(s + "\n")
            text_f.write("\n\n")

            # chunking & JSON 저장 준비
            chunks = sentence_chunk_with_window(sentences, sec_name)
            all_chunks.extend(chunks)

    # JSONL 저장
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("Done! Total chunks:", len(all_chunks))
    print("Text-only file:", OUTPUT_TEXT)
    print("Output JSONL:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
