import fitz  # PyMuPDF
import re
import json
import tiktoken


############################################################
# 0) 파일 경로 / 출력 파일
############################################################
PDF_PATH = "D:/건대/12~1논문/251216/Docs/123/21916-g20.pdf"          # <-- 너 파일 경로로 변경
OUTPUT_TEXT = "21916-g20_selected_text.txt"
OUTPUT_JSONL = "21916-g20_chunks.jsonl"


############################################################
# 1) PDF Section 범위 정의 (0-based, end는 exclusive)
#    - 아래는 너가 준 TOC 기반으로 "RAG에 유용한 섹션"만 우선 잡아둔 예시야.
#    - 페이지가 안 맞으면 여기 숫자만 수정하면 됨.
#
#    가정: "printed page" == "PDF page" 라고 보고,
#    printed N 페이지 -> 0-based index (N-1) 로 잡음.
############################################################
SECTION_RANGES = {
    # 4 Rel-16 Executive Summary (p.9) + 5 URLLC (p.9~14 초반)
    # 6이 p.14 시작이라, 넉넉히 p.9~p.15 정도로 끊어둠
    "Section4~5": (8, 14),

    # 6 Support of LAN-type services (p.14~16)
    "Section6": (13, 16),

    # 9 Northbound APIs related items (p.29~35)
    "Section9": (28, 35),

    # 16 Slicing (p.72~78)
    "Section16": (71, 78),

    # 18 Other system-wide Features (p.80~89)
    "Section18": (79, 89),

    # 21 Telecom Management (p.140~146)  ※ SLS assurance/trace 포함
    "Section21": (139, 146),

    # (선택) Annex A Work Plan (p.148~156) 필요하면 켜
    # "AnnexA": (147, 156),
}


############################################################
# 2) (핵심) 반복 헤더/푸터 자동 제거 유틸
#    - 각 섹션 범위의 페이지들에서 "자주 반복되는 상단/하단 라인"을 찾고,
#      해당 라인을 페이지 텍스트에서 제거함.
############################################################
def _normalize_line(line: str) -> str:
    # 공백 정리 + 너무 긴 공백 제거
    line = re.sub(r"\s+", " ", line.strip())
    return line

def detect_repeated_header_footer(page_texts, top_k=3, bottom_k=3, threshold_ratio=0.6):
    """
    page_texts: [str, str, ...] 섹션 내 페이지 텍스트들
    top_k/bottom_k: 각 페이지 상단/하단에서 몇 줄을 후보로 볼지
    threshold_ratio: 전체 페이지 중 이 비율 이상 반복되면 header/footer로 간주
    """
    from collections import Counter

    top_counter = Counter()
    bottom_counter = Counter()

    per_page_top = []
    per_page_bottom = []

    for txt in page_texts:
        lines = [_normalize_line(l) for l in txt.splitlines()]
        # 빈줄 제거
        lines = [l for l in lines if l]

        top_lines = lines[:top_k] if lines else []
        bottom_lines = lines[-bottom_k:] if lines else []

        per_page_top.append(top_lines)
        per_page_bottom.append(bottom_lines)

        top_counter.update(top_lines)
        bottom_counter.update(bottom_lines)

    n_pages = max(1, len(page_texts))
    min_count = int(n_pages * threshold_ratio)

    repeated_top = {line for line, cnt in top_counter.items() if cnt >= min_count}
    repeated_bottom = {line for line, cnt in bottom_counter.items() if cnt >= min_count}

    return repeated_top, repeated_bottom

def remove_header_footer_from_page(text: str, repeated_top: set, repeated_bottom: set) -> str:
    lines = [_normalize_line(l) for l in text.splitlines()]
    # 원본의 빈줄은 어느 정도 살리기 위해 ""는 유지하지 않고 나중에 정리
    lines = [l for l in lines if l]

    # 반복 top/bottom 라인 제거(페이지 어디에 있어도 동일 문자열이면 제거)
    cleaned = [l for l in lines if (l not in repeated_top and l not in repeated_bottom)]
    return "\n".join(cleaned)


############################################################
# 3) 페이지별 텍스트 추출 (섹션 단위)
############################################################
def extract_selected_sections(pdf_path, ranges):
    doc = fitz.open(pdf_path)
    extracted = []

    for sec_name, (start, end) in ranges.items():
        # end 보정
        if end <= start:
            end = start + 1

        # 범위 보정
        start = max(0, start)
        end = min(len(doc), end)

        page_texts = []
        for page_num in range(start, end):
            page_texts.append(doc[page_num].get_text("text"))

        # 섹션 내부에서 반복 header/footer 자동 탐지 후 제거
        repeated_top, repeated_bottom = detect_repeated_header_footer(
            page_texts, top_k=3, bottom_k=3, threshold_ratio=0.6
        )

        cleaned_pages = [
            remove_header_footer_from_page(t, repeated_top, repeated_bottom)
            for t in page_texts
        ]

        extracted.append((sec_name, "\n".join(cleaned_pages)))

    doc.close()
    return extracted


############################################################
# 4) 텍스트 정제 (3GPP 문서에 자주 나오는 것들 추가 제거)
############################################################
def clean_text(text, remove_fig_table=False):
    # 3GPP 문서에서 흔한 라인 패턴 제거(있으면 지우고, 없으면 영향 없음)
    patterns = [
        r"(?m)^\s*3GPP\s+TR\s+21\.916.*$",                 # 예: "3GPP TR 21.916 V16.x.x ..."
        r"(?m)^\s*Release\s*16.*$",                        # "Release 16 ..." 라인
        r"(?m)^\s*TSG\s+SA.*$",                            # 문서 상단에 나오는 위원회 라인들
        r"(?m)^\s*Copyright\s+Notification.*$",            # Copyright Notification
        r"(?m)^\s*©\s*3GPP.*$",                            # © 3GPP ...
        r"(?m)^\s*All\s+rights\s+reserved.*$",
    ]
    for p in patterns:
        text = re.sub(p, "", text)

    # 단독 페이지 숫자 라인 제거
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # (옵션) Figure/Table 캡션 제거
    if remove_fig_table:
        text = re.sub(r"(?m)^\s*Figure\s+\d+.*$", "", text)
        text = re.sub(r"(?m)^\s*Table\s+\d+.*$", "", text)

    # 과도한 공백/빈줄 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


############################################################
# 5) 문단 병합
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
# 6) 문단 → 문장 분해
############################################################
def split_into_sentences(paragraphs):
    sentences = []
    for para in paragraphs:
        # 너무 공격적으로 쪼개면 약어/버전/V2X 같은 데서 깨질 수 있어,
        # 기본 마침표/물음표/느낌표 기준만 적용
        sents = re.split(r'(?<=[\.!?])\s+', para)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)
    return sentences


############################################################
# 7) Sliding-window Sentence Chunking
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

        # 청크 꽉 차면 저장 + overlap 구성
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
            "prefix": f"[3GPP TR 21.916 - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "3GPP TR 21.916 (Rel-16 work items)",
                "section": sec_name,
                "chunk_id": idx
            }
        })

    return final_chunks


############################################################
# 8) MAIN PIPELINE
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

            # JSONL chunk 생성
            chunks = sentence_chunk_with_window(sentences, sec_name)
            all_chunks.extend(chunks)

    # JSONL 저장
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("Done! Total chunks:", len(all_chunks))
    print("Text-only file:", OUTPUT_TEXT)
    print("JSONL file:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
