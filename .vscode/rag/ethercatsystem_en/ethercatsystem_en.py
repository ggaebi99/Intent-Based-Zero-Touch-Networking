import fitz  # PyMuPDF
import re
import json

# (옵션) 토큰 기반 청킹을 원하면 tiktoken 사용
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def token_len(s: str) -> int:
        return len(_enc.encode(s))

except ImportError:
    # tiktoken 없으면 대충 단어/기호 기준으로 fallback
    def token_len(s: str) -> int:
        return max(1, len(re.findall(r"\w+|[^\w\s]", s)))


############################################################
# 0) 파일 경로 (ethercatsystem_en.pdf 전용)
############################################################
PDF_PATH = "D:/건대/12~1논문/251216/Docs/ethercatsystem_en.pdf"  # 같은 폴더에 두면 이렇게. 경로 다르면 수정.
OUTPUT_TEXT = "ethercatsystem_selected_text.txt"
OUTPUT_JSONL = "ethercatsystem_chunks.jsonl"


############################################################
# 1) RAG에 넣을 섹션 범위 (0-based, end는 exclusive)
#    - 너가 이미 찝은 추천 구간 그대로
############################################################
SECTION_RANGES = {
    # 2 EtherCAT basics: printed 10 ~ 61  => 0-based 9 ~ 60  => (9, 61)
    "Section2": (9, 61),

    # 3.6.7 Standard behavior...: printed 142 ~ 153 => 0-based 141 ~ 152 => (141, 153)
    "Section3.6.7": (141, 153),

    # 3.7 Notes on distributed clocks: printed 154 ~ 164 => 0-based 153 ~ 163 => (153, 164)
    "Section3.7": (153, 164),

    # 4.1~4.2 Diagnostics: printed 165 ~ 194 => 0-based 164 ~ 193 => (164, 194)
    "Section4.1~4.2": (164, 194),
}


############################################################
# 2) 페이지 헤더/푸터 제거 (EtherCAT 문서 포맷 전용)
#    - 각 페이지 상단에 반복되는:
#      [섹션명], EtherCAT, (페이지번호), Version: x.x, (챕터번호)
#    - 본문 내 숫자 라인은 건드리지 않기 위해 "페이지 상단부"만 제거
############################################################
def strip_page_header(page_text: str) -> str:
    lines = page_text.splitlines()

    # 앞쪽 공백 라인 제거
    while lines and not lines[0].strip():
        lines.pop(0)

    # 1) 첫 줄이 "EtherCAT basics", "EtherCAT diagnostics" 같은 섹션 헤더인 경우 제거
    if lines and re.match(r"^EtherCAT\b.*", lines[0].strip()):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    # 2) 다음 줄이 "EtherCAT" 단독이면 제거
    if lines and lines[0].strip() == "EtherCAT":
        lines.pop(0)

    # 3) 페이지 번호(숫자만) 제거
    if lines and re.match(r"^\d+$", lines[0].strip()):
        lines.pop(0)

    # 4) 버전 라인 제거 (예: Version: 5.6 / Version: 5.6.0 등)
    if lines and re.match(r"^Version:\s*\d+(\.\d+)*$", lines[0].strip()):
        lines.pop(0)

    # 5) 챕터 번호(숫자만) 한 줄 더 있는 경우 제거
    if lines and re.match(r"^\d+$", lines[0].strip()):
        lines.pop(0)

    return "\n".join(lines)


############################################################
# 3) 섹션 범위 페이지 텍스트 추출
############################################################
def extract_selected_sections(pdf_path: str, ranges: dict):
    doc = fitz.open(pdf_path)
    extracted = []

    for sec_name, (start, end) in ranges.items():
        # 범위 안전 보정
        start = max(0, start)
        end = min(len(doc), end)
        if end <= start:
            end = min(len(doc), start + 1)

        parts = []
        for page_num in range(start, end):
            raw = doc[page_num].get_text("text")
            cleaned_page = strip_page_header(raw)
            parts.append(cleaned_page)

        extracted.append((sec_name, "\n".join(parts), start, end - 1))

    doc.close()
    return extracted


############################################################
# 4) 전체 텍스트 2차 정제 (공백/개행 위주)
############################################################
def clean_text(text: str) -> str:
    # 과도한 개행 줄이기
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 탭/연속 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


############################################################
# 5) 문단 병합
############################################################
def merge_paragraphs(text: str):
    lines = text.split("\n")
    paras = []
    buf = []

    for ln in lines:
        ln = ln.strip()
        if ln == "":
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln)

    if buf:
        paras.append(" ".join(buf))

    return [p for p in paras if p.strip()]


############################################################
# 6) 문장 분해
############################################################
def split_into_sentences(paragraphs):
    sentences = []
    for para in paragraphs:
        # 너무 공격적으로 쪼개면 약어/표기 깨질 수 있어서 기본 split만 사용
        sents = re.split(r'(?<=[.!?])\s+', para)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)
    return sentences


############################################################
# 7) 슬라이딩 윈도우 Sentence Chunking
############################################################
def sentence_chunk_with_window(sentences, sec_name, chunk_size=300, window_size=100):
    sent_tokens = [(s, token_len(s)) for s in sentences]

    chunks = []
    current = []
    tokens = 0
    i = 0

    while i < len(sent_tokens):
        sent, tok = sent_tokens[i]

        # 문장이 너무 길면 단독 청크로
        if tok > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
                tokens = 0
            chunks.append(sent)
            i += 1
            continue

        # 청크가 찼으면 저장 + overlap 구성
        if tokens + tok > chunk_size and current:
            chunks.append(" ".join(current))

            overlap = []
            overlap_tok = 0
            j = len(current) - 1
            while j >= 0:
                t = token_len(current[j])
                if overlap_tok + t > window_size:
                    break
                overlap.insert(0, current[j])
                overlap_tok += t
                j -= 1

            current = overlap
            tokens = overlap_tok

        current.append(sent)
        tokens += tok
        i += 1

    if current:
        chunks.append(" ".join(current))

    # JSONL 구조 (id/prefix는 TSN 스타일로 통일)
    json_chunks = []
    for idx, ch in enumerate(chunks):
        json_chunks.append({
            "id": f"{sec_name}_{idx}",
            "prefix": f"[EtherCAT - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "EtherCAT System Description",
                "document": "ethercatsystem_en.pdf",
                "section": sec_name,
                "chunk_id": idx
            }
        })

    return json_chunks


############################################################
# 8) MAIN
############################################################
def main():
    sections = extract_selected_sections(PDF_PATH, SECTION_RANGES)

    all_chunks = []
    with open(OUTPUT_TEXT, "w", encoding="utf-8") as tf:
        for sec_name, raw_text, pdf_start, pdf_end in sections:
            cleaned = clean_text(raw_text)
            paras = merge_paragraphs(cleaned)
            sents = split_into_sentences(paras)

            # text-only 저장
            tf.write(f"<<{sec_name}>>\n")
            for s in sents:
                tf.write(s + "\n")
            tf.write("\n\n")

            # jsonl 청킹
            chunks = sentence_chunk_with_window(sents, sec_name, chunk_size=300, window_size=100)

            # 섹션 페이지 메타를 청크에 추가(추적용)
            for c in chunks:
                c["metadata"]["pdf_page_start"] = pdf_start
                c["metadata"]["pdf_page_end"] = pdf_end

            all_chunks.extend(chunks)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as jf:
        for ch in all_chunks:
            jf.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("Done!")
    print("Output text:", OUTPUT_TEXT)
    print("Output jsonl:", OUTPUT_JSONL)
    print("Total chunks:", len(all_chunks))


if __name__ == "__main__":
    main()
