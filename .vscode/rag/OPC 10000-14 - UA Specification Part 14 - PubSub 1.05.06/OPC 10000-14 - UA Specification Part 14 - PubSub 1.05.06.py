import fitz  # PyMuPDF
import re
import json

############################################################
# 0) 파일 경로
############################################################
PDF_PATH = r"D:/건대/12~1논문/251216/Docs/1766412327-ua-part-14-pubsub-1.05.06-2025-10-31/OPC 10000-14 - UA Specification Part 14 - PubSub 1.05.06.pdf"

OUTPUT_TEXT = "OPC10000-14_selected_text.txt"
OUTPUT_JSONL = "OPC10000-14_chunks.jsonl"

############################################################
# 1) (옵션) "실제 페이지" -> PDF 0-based 페이지 인덱스 변환
#    이 PDF는 실제 1페이지가 PDF index 20에서 시작해서 offset = 19로 잡음
############################################################
PDF_PAGE_OFFSET = 19  # 실제 N -> PDF index = N + 19

def printed_to_pdf_index(printed_page: int) -> int:
    return printed_page + PDF_PAGE_OFFSET

############################################################
# 2) 섹션 범위 정의 (0-based, end는 exclusive)
#    - 아래는 임의 값(안맞으면 수정)
############################################################
SECTION_RANGES = {
    "Section5": (24, 44),           # 5 PubSub Concepts
    "Section6": (44, 117),          # 6 PubSub communication parameters
    "Section7": (117, 167),         # 7 PubSub mappings
    "Section8": (167, 185),         # 8 PubSub Security Key Service model
    "AnnexA": (264, 285),           # Annex A (normative)
    "AnnexB": (285, 291),           # Annex B (informative)
}

# "실제 페이지"로 관리하고 싶으면 이 블록을 쓰고 위 SECTION_RANGES 대신 사용:
# SECTION_RANGES_PRINTED = {
#     "Section5": (5, 25),
#     "Section6": (25, 98),
#     "Section7": (98, 148),
#     "Section8": (148, 166),
#     "AnnexA": (245, 266),
#     "AnnexB": (266, 272),
# }
# SECTION_RANGES = {
#     k: (printed_to_pdf_index(s), printed_to_pdf_index(e))
#     for k, (s, e) in SECTION_RANGES_PRINTED.items()
# }

############################################################
# 3) 페이지별 텍스트 추출
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
# 4) 텍스트 정제 (헤더/버전/페이지숫자 제거 등)
############################################################
def clean_text(text, remove_fig_table=False):
    # 상단 헤더 제거
    text = re.sub(r"(?m)^\s*OPC\s*10000-14:\s*PubSub\s*$", "", text)

    # 버전 라인 제거 (예: 1.05.06)
    text = re.sub(r"(?m)^\s*\d+\.\d+\.\d+\s*$", "", text)

    # ===== PAGE N ===== 같은 구분선 제거 (있을 때)
    text = re.sub(r"(?m)^\s*=====\s*PAGE\s*\d+\s*=====\s*$", "", text)

    # 페이지 번호/챕터번호처럼 단독 숫자 라인 제거
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # (옵션) Figure/Table 캡션 제거
    if remove_fig_table:
        text = re.sub(r"(?m)^\s*Figure\s+\d+.*$", "", text)
        text = re.sub(r"(?m)^\s*Table\s+\d+.*$", "", text)

    # 공백 정리
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

############################################################
# 5) 문단 병합
############################################################
def merge_paragraphs(text):
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
        sents = re.split(r'(?<=[.!?])\s+', para)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)
    return sentences

############################################################
# 7) 슬라이딩 윈도우 청킹(JSONL)
############################################################
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def token_len(s: str) -> int:
        return len(_enc.encode(s))
except ImportError:
    # fallback: 단어/기호 기준 대충 토큰 카운트
    def token_len(s: str) -> int:
        return max(1, len(re.findall(r"\w+|[^\w\s]", s)))

STANDARD_TAG = "OPC 10000-14"  # TSN에서 "TSN 802.1AS" 넣는 자리랑 동일

def sentence_chunk_with_window(sentences, sec_name, chunk_size=300, window_size=100):
    sent_tokens = [(s, token_len(s)) for s in sentences]

    chunks = []
    current = []
    tokens = 0
    i = 0

    while i < len(sent_tokens):
        sent, tok = sent_tokens[i]

        # 너무 긴 문장은 단독 청크로
        if tok > chunk_size:
            chunks.append(sent)
            i += 1
            continue

        # 청크 꽉 차면 저장 + overlap 구성
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

    json_chunks = []
    for idx, ch in enumerate(chunks):
        json_chunks.append({
            "id": f"{sec_name}_{idx}",  # 예: Section6_7_0 형태로 맞추려면 sec_name을 그렇게 주면 됨
            "prefix": f"[{STANDARD_TAG} - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": STANDARD_TAG,
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
        for sec_name, raw_text in sections:
            cleaned = clean_text(raw_text, remove_fig_table=False)
            paras = merge_paragraphs(cleaned)
            sents = split_into_sentences(paras)

            # text-only 저장 (가공 포맷)
            tf.write(f"<<{sec_name}>>\n")
            for s in sents:
                tf.write(s + "\n")
            tf.write("\n\n")

            # chunking(JSONL)
            all_chunks.extend(sentence_chunk_with_window(sents, sec_name))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as jf:
        for ch in all_chunks:
            jf.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("Done!")
    print("Output text:", OUTPUT_TEXT)
    print("Output jsonl:", OUTPUT_JSONL)
    print("Sections:", list(SECTION_RANGES.keys()))
    print("Total chunks:", len(all_chunks))

if __name__ == "__main__":
    main()
