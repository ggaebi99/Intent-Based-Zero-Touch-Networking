import re
import json
import tiktoken

INPUT_TEXT = "PROFINET_selected_text.txt"   # 너가 업로드한 파일
OUTPUT_TEXT = "PROFINET_clean_sentences.txt"
OUTPUT_JSONL = "PROFINET_chunks.jsonl"


############################################################
# 1) 파일 전체 로드
############################################################

with open(INPUT_TEXT, "r", encoding="utf-8") as f:
    full_text = f.read()


############################################################
# 2) 섹션 블록 자동 추출
############################################################
SECTION_MARKERS = [
    "BasicFunctions",
    "IRT",
    "OptionalRT",
    "Profiles",
    "Security"
]

# 정규식으로 <<SectionName>> 기준 분리
pattern = r"<<(" + "|".join(SECTION_MARKERS) + ")>>"
splits = re.split(pattern, full_text)

# re.split 결과 형태:
# ['', 'BasicFunctions', ' ...text... ', 'IRT', ' ...text... ', ...]
sections = {}

for i in range(1, len(splits), 2):
    sec_name = splits[i]
    sec_text = splits[i+1]
    sections[sec_name] = sec_text.strip()


############################################################
# 3) 노이즈 제거
############################################################

def clean_text(text):

    # PROFINET System Description 헤더 제거 (좌/우 페이지 모두 제거)
    text = re.sub(r"PROFINET System Description\s*\d*", "", text)
    text = re.sub(r"PROFINET System Description", "", text)

    # 빈 줄, 중복 공백 정리
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


############################################################
# 4) 문단 병합
############################################################

def merge_paragraphs(text):
    lines = text.split("\n")
    paras = []
    buf = ""

    for ln in lines:
        ln = ln.strip()
        if ln == "":
            if buf:
                paras.append(buf)
                buf = ""
        else:
            buf += " " + ln if buf else ln

    if buf:
        paras.append(buf)

    return paras


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
# 6) 슬라이딩 윈도우 청킹
############################################################

def chunk_sentences(sentences, sec_name, chunk_size=300, window_size=100):

    enc = tiktoken.get_encoding("cl100k_base")
    sent_tokens = [(s, len(enc.encode(s))) for s in sentences]

    chunks = []
    current = []
    tokens = 0

    i = 0
    while i < len(sent_tokens):
        sent, tok = sent_tokens[i]

        if tok > chunk_size:
            chunks.append(sent)
            i += 1
            continue

        if tokens + tok > chunk_size:
            chunks.append(" ".join(current))

            # 슬라이딩 오버랩
            overlap = []
            overlap_tok = 0
            j = len(current) - 1

            while j >= 0:
                t = len(enc.encode(current[j]))
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

    # JSON 변환
    json_chunks = []
    for idx, ch in enumerate(chunks):
        json_chunks.append({
            "id": f"{sec_name}_{idx}",
            "prefix": f"[PROFINET - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "PROFINET",
                "section": sec_name,
                "chunk_id": idx
            }
        })

    return json_chunks


############################################################
# 7) MAIN PIPELINE 실행
############################################################

all_chunks = []

with open(OUTPUT_TEXT, "w", encoding="utf-8") as tfile:

    for sec, text in sections.items():

        cleaned = clean_text(text)
        paras = merge_paragraphs(cleaned)
        sents = split_into_sentences(paras)

        tfile.write(f"<<{sec}>>\n")
        for s in sents:
            tfile.write(s + "\n")
        tfile.write("\n\n")

        chunks = chunk_sentences(sents, sec)
        all_chunks.extend(chunks)


with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for ch in all_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print("Done!")
print("Sections parsed:", list(sections.keys()))
print("Total chunks:", len(all_chunks))
print("Output JSONL:", OUTPUT_JSONL)
