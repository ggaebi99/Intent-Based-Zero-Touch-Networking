# rag_bm25.py
# BM25 RAG for Industry 5.0 standards chunks (JSONL)
# - Keeps dotted/hyphenated identifiers (802.1Qbv, OPC10000-14)
# - Adds dual-token normalized variants (8021qbv, opc1000014)
# - Indexes prefix + text for better section-title retrieval
#
# Usage:
#   pip install rank-bm25 numpy
#   python rag_bm25.py build
#   python rag_bm25.py query --q "Time synchronization between PLCs must not drift" --k 6
#   python rag_bm25.py query --q "frame replication elimination reliability" --k 6 --kbs KB_TSN_8021CB
#   python rag_bm25.py query --q "Time synchronization drift gPTP" --type operational --subtypes TIME_SYNC --k 6

import os
import re
import json
import glob
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi


# ---------------------------
# 1) File discovery / KB mapping
# ---------------------------

DEFAULT_DATA_DIR = "D:/건대/12~1논문/251216/Codes/json_gather"
INDEX_DIR = "./bm25_index"

# 네 파일명 기준으로 KB ID를 깔끔하게 고정 매핑
# (파일이 있으면 자동으로 포함됨)
KNOWN_FILES_TO_KB = {
    "8021AS_selected_chunks.jsonl": "KB_TSN_8021AS",
    "8021CB_selected_chunks.jsonl": "KB_TSN_8021CB",
    "8021Qbv_selected_chunks.jsonl": "KB_TSN_8021QBV",
    "8021Qbu_selected_chunks.jsonl": "KB_TSN_8021QBU",
    "8021Qci_selected_chunks.jsonl": "KB_TSN_8021QCI",
    "OPC10000-14_chunks.jsonl": "KB_OPCUA_PUBSUB",
    "ethercatsystem_chunks.jsonl": "KB_ETHER_CAT",
    "PROFINET_chunks.jsonl": "KB_PROFINET",
    "PROFINET_chunks copy.jsonl": "KB_PROFINET_EXTRA",
    "EPSG301_chunks.jsonl": "KB_POWERLINK",
    "21916-g20_chunks.jsonl": "KB_3GPP_URLLC",      # (네 파일명 그대로 사용)
    "ISO13849_selected_chunks.jsonl": "KB_ISO_13849",
    "IEC61508_selected_chunks.jsonl": "KB_IEC_61508_OV",
}

# ---------------------------
# 2) Tokenizer (dot/hyphen keep + dual-token normalization)
# ---------------------------

# 핵심: 802.1Qbv / OPC10000-14 같은 표준 식별자를 "한 토큰"으로 유지
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")

# (선택) 너무 흔한 섹션 제목 단어는 노이즈가 될 수 있어 제거 가능
# 일단은 최소만 넣었고, 필요하면 확장 가능
STOPWORDS = {
    "general", "introduction", "overview", "scope", "definitions",
    "clause", "section", "annex", "table", "figure", "time", "times", "timestamp", "network", "packet", "data"
}

def tokenize_with_dual(text: str) -> List[str]:
    """
    BM25용 토큰화:
    1) 점(.)/하이픈(-) 포함 토큰 유지: 802.1Qbv, OPC10000-14, gPTP
    2) 동시에 정규화 토큰 추가(dual-token): 8021qbv, opc1000014
       -> 표기 흔들림 흡수
    """
    text = text.lower()
    base_tokens = TOKEN_RE.findall(text)

    out = []
    for tok in base_tokens:
        if tok in STOPWORDS:
            continue
        out.append(tok)

        # dual-token: . - 제거한 정규화 버전도 추가
        norm = tok.replace(".", "").replace("-", "")
        if norm != tok and len(norm) >= 4 and norm not in STOPWORDS:
            out.append(norm)

    return out


# ---------------------------
# 3) JSONL loader
# ---------------------------

def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ---------------------------
# 4) BM25 index build / save / load
# ---------------------------

@dataclass
class ChunkMeta:
    kb: str
    chunk_id: str
    prefix: str
    text: str
    metadata: Dict[str, Any]

def index_path(kb_id: str) -> str:
    return os.path.join(INDEX_DIR, f"{kb_id}.pkl")

def build_bm25_for_kb(kb_id: str, jsonl_path: str, overwrite: bool = False) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    out_path = index_path(kb_id)

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {kb_id} already indexed: {out_path}")
        return

    chunks = load_jsonl(jsonl_path)

    tokenized_docs = []
    metas: List[ChunkMeta] = []

    for c in chunks:
        prefix = (c.get("prefix") or "").strip()
        text = (c.get("text") or "").strip()

        # ✅ 헤더/섹션 정보를 검색에 포함시키기 위해 prefix + text로 인덱싱
        full = (prefix + "\n" + text).strip() if prefix else text

        tokens = tokenize_with_dual(full)
        tokenized_docs.append(tokens)

        metas.append(ChunkMeta(
            kb=kb_id,
            chunk_id=str(c.get("id", "")),
            prefix=prefix,
            text=text,
            metadata=c.get("metadata", {}) or {}
        ))

    bm25 = BM25Okapi(tokenized_docs)

    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "metas": metas, "source": jsonl_path}, f)

    print(f"[OK] Built {kb_id}: chunks={len(metas)} -> {out_path}")

def load_bm25_index(kb_id: str):
    path = index_path(kb_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing index for {kb_id}. Run `python rag_bm25.py build` first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------
# 5) Retrieval
# ---------------------------

def retrieve_kb(kb_id: str, query: str, top_k: int = 5):
    data = load_bm25_index(kb_id)
    bm25: BM25Okapi = data["bm25"]
    metas: List[ChunkMeta] = data["metas"]

    q = tokenize_with_dual(query)
    scores = bm25.get_scores(q)

    idx = np.argsort(-scores)[:top_k]
    results = []
    for i in idx:
        m = metas[int(i)]
        results.append({
            "kb": kb_id,
            "score": float(scores[int(i)]),
            "chunk_id": m.chunk_id,
            "prefix": m.prefix,
            "metadata": m.metadata,
            "text": m.text,
        })
    return results

def retrieve_multi(kb_ids: List[str], query: str, top_k_per_kb: int = 5, final_top_k: int = 8):
    all_hits = []
    for kb in kb_ids:
        hits = retrieve_kb(kb, query, top_k=top_k_per_kb)
        all_hits.extend(hits)

    # BM25 점수는 KB마다 스케일이 다를 수 있음.
    # 여기서는 "KB별 top_k 먼저 뽑고 합친 뒤" 단순 재정렬(검증용으로 충분)
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:final_top_k]


# ---------------------------
# 6) (Optional) Router: type/subtypes -> KB selection
# ---------------------------

# 네 전체 문서/표준 기반으로 기본 라우팅(우선순위) 제공
SUBTYPE_TO_KB_PRIORITY = {
    "TIME_SYNC": ["KB_TSN_8021AS"],
    "DETERMINISTIC_SCHEDULING": ["KB_TSN_8021QBV"],
    "FRAME_PREEMPTION": ["KB_TSN_8021QBU"],
    "RELIABILITY_FRER": ["KB_TSN_8021CB"],
    "STREAM_FILTERING_POLICING": ["KB_TSN_8021QCI"],
    "OPCUA_PUBSUB": ["KB_OPCUA_PUBSUB"],
    "WIRELESS_URLLC": ["KB_3GPP_URLLC"],
    "INDUSTRIAL_MOTION_ETHERNET": ["KB_ETHER_CAT", "KB_PROFINET", "KB_POWERLINK"],
    "CONTROL_LOOP_REALTIME": ["KB_TSN_8021QBV", "KB_TSN_8021QBU", "KB_ETHER_CAT", "KB_PROFINET", "KB_POWERLINK"],
    "SAFETY": ["KB_ISO_13849", "KB_IEC_61508_OV"],
}

TYPE_BOOST = {
    "performance": ["KB_TSN_8021CB", "KB_TSN_8021QBV", "KB_TSN_8021QBU"],
    "operational": ["KB_TSN_8021AS", "KB_TSN_8021QBV"],
    "application": ["KB_TSN_8021QBV", "KB_OPCUA_PUBSUB", "KB_3GPP_URLLC"],
    "policy": ["KB_TSN_8021QCI", "KB_TSN_8021QBV"],
}

def route_kbs(intent_type: str, subtypes: List[str], k: int = 3) -> List[str]:
    score: Dict[str, float] = {}

    for st in subtypes:
        for rank, kb in enumerate(SUBTYPE_TO_KB_PRIORITY.get(st, [])):
            score[kb] = score.get(kb, 0.0) + (10.0 - rank)  # subtype 신호가 더 강하게

    for kb in TYPE_BOOST.get(intent_type, []):
        score[kb] = score.get(kb, 0.0) + 3.0              # type 부스팅은 보조

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    kbs = [kb for kb, _ in ranked[:k]]

    # fallback: 아무것도 없으면 전체 검색 대신 TSN 핵심 1~2개만
    if not kbs:
        kbs = ["KB_TSN_8021AS", "KB_TSN_8021QBV"]
    return kbs


# ---------------------------
# 7) Utilities: discover KB files, ensure indexes exist
# ---------------------------

def discover_sources(data_dir: str) -> Dict[str, str]:
    """
    /mnt/data 내 jsonl 파일을 스캔해서, KNOWN_FILES_TO_KB에 해당하는 것만 자동 연결.
    """
    paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
    found = {}
    for p in paths:
        base = os.path.basename(p)
        if base in KNOWN_FILES_TO_KB:
            kb_id = KNOWN_FILES_TO_KB[base]
            found[kb_id] = p
    return found

def build_all(data_dir: str, overwrite: bool = False):
    sources = discover_sources(data_dir)
    if not sources:
        raise RuntimeError(f"No known JSONL files found in {data_dir}. Check filenames/paths.")

    print("[INFO] Discovered KB sources:")
    for kb, p in sorted(sources.items()):
        print(f"  - {kb}: {p}")

    for kb, p in sorted(sources.items()):
        build_bm25_for_kb(kb, p, overwrite=overwrite)

def list_indexed_kbs() -> List[str]:
    if not os.path.exists(INDEX_DIR):
        return []
    files = glob.glob(os.path.join(INDEX_DIR, "*.pkl"))
    return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])


# ---------------------------
# 8) CLI
# ---------------------------

def cmd_build(args):
    build_all(args.data_dir, overwrite=args.overwrite)

def cmd_query(args):
    # if indexes missing -> build first?
    indexed = set(list_indexed_kbs())
    if not indexed:
        raise RuntimeError("No BM25 indexes found. Run: python rag_bm25.py build")

    query = args.q.strip()
    top_k = args.k

    # 선택 1) KB를 직접 지정했으면 그걸로 검색
    if args.kbs:
        kbs = args.kbs
    # 선택 2) type/subtypes 라우팅을 쓰면 라우팅된 KB만 검색
    elif args.type and args.subtypes:
        kbs = route_kbs(args.type, args.subtypes, k=args.kbs_top)
    # 선택 3) 아무것도 안 주면 전체 indexed KB에서 검색(디버깅에 좋음)
    else:
        kbs = sorted(list(indexed))

    # 실제로 인덱스가 있는 KB만 대상으로
    kbs = [kb for kb in kbs if kb in indexed]
    if not kbs:
        raise RuntimeError("No valid KBs to search (indexes missing for given KB IDs).")

    hits = retrieve_multi(
        kb_ids=kbs,
        query=query,
        top_k_per_kb=args.top_k_per_kb,
        final_top_k=top_k
    )

    print("\n==== QUERY ====")
    print(query)
    if args.type:
        print(f"type={args.type}")
    if args.subtypes:
        print(f"subtypes={args.subtypes}")
    print(f"searched_kbs={kbs}")

    print("\n==== TOP HITS ====")
    for rank, h in enumerate(hits, 1):
        prefix = h["prefix"].replace("\n", " ").strip()
        snippet = h["text"].replace("\n", " ").strip()
        snippet = snippet[:240] + (" ..." if len(snippet) > 240 else "")

        print(f"\n[{rank}] kb={h['kb']} score={h['score']:.3f} chunk_id={h['chunk_id']}")
        if prefix:
            print(f"  prefix: {prefix}")
        print(f"  text:   {snippet}")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p_build.add_argument("--overwrite", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query")
    p_query.add_argument("--q", required=True, help="query / intent sentence")
    p_query.add_argument("--k", type=int, default=8, help="final top-k results to print")
    p_query.add_argument("--top-k-per-kb", type=int, default=6, help="retrieve top-k per KB before merge")
    p_query.add_argument("--kbs", nargs="*", help="explicit KB IDs to search (skip routing)")
    p_query.add_argument("--type", choices=["performance","operational","application","policy"])
    p_query.add_argument("--subtypes", nargs="*", help="subtypes for routing (e.g., TIME_SYNC RELIABILITY_FRER)")
    p_query.add_argument("--kbs-top", type=int, default=3, help="how many KBs to route into")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
