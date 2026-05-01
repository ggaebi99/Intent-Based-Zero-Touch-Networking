import fitz  # PyMuPDF
import re
import json
import tiktoken
import re


############################################################
# 1) PDF Section 범위 정의 (필요한 부분만 추출)
############################################################

SECTION_RANGES = {
    "Section7": (35, 48),
    "Section8": (48, 59),
    "Section10.7": (156, 160),
    "AnnexB": (397, 405)
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
# 3) 텍스트 정제
############################################################

def clean_text(text):

    text = re.sub(r"Authorized licensed use limited to:.*?Restrictions apply\.", "", text, flags=re.DOTALL)
    text = re.sub(r"Copyright ©.*?All rights reserved\.", "", text)

    text = re.sub(
        r"IEEE Standard for Local and Metropolitan Area Networks[\s\S]*?Applications",
        "",
        text
    )

    text = re.sub(r"IEEE Std 802\.1AS-2020.*", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Figure\s+\d+.*", "", text)
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

    paras = [p for p in paras if p.strip()]
    return paras


############################################################
# 5) 문단 → 문장 분해
############################################################

def split_into_sentences(paragraphs):
    sentences = []
    for para in paragraphs:
        sents = re.split(r'(?<=[\.!?])\s+', para)
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
            "prefix": f"[TSN 802.1AS - {sec_name} - Chunk {idx}]",
            "text": ch,
            "metadata": {
                "standard": "802.1AS-2020",
                "section": sec_name,
                "chunk_id": idx
            }
        })

    return final_chunks


############################################################
# 7) MAIN PIPELINE
############################################################

pdf_path = "D:/건대/12~1논문/251216/Docs/123/8021AS-2020.pdf"
output_file = "8021AS_selected_chunks.jsonl"
text_output_file = "8021AS_selected_text.txt"

sections = extract_selected_sections(pdf_path, SECTION_RANGES)

all_chunks = []

with open(text_output_file, "w", encoding="utf-8") as text_f:
    for sec_name, raw_text in sections:
        cleaned = clean_text(raw_text)
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
with open(output_file, "w", encoding="utf-8") as f:
    for ch in all_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print("Done! Total chunks:", len(all_chunks))
print("Text-only file:", text_output_file)
