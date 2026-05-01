# layer1_pipeline.py
# Industry5.0 Layer1: Intent -> (type/subtypes) -> RAG(BM25) -> QoS JSON (+target/hard) -> Layer2 scalar
#
# Requirements:
#   pip install rank-bm25 numpy pyyaml
# (OpenAI 사용 시)
#   pip install openai
#
# Build:
#   python layer1_pipeline.py build --config config.yaml
# Run (mock):
#   python layer1_pipeline.py run --config config.yaml --q "..." --k 8
# Run (OpenAI):
#   python layer1_pipeline.py run --config config.yaml --q "..." --k 8 --llm openai --model gpt-5-nano
#
# Notes:
# - Tokenizer keeps dotted/hyphenated identifiers and adds normalized dual-token.
# - Evidence includes prefix + text in index for section-title retrieval.
# - This script DOES NOT read any environment variables. Only YAML + CLI.

import os
import re
import json
import glob
import time
import pickle
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi
import yaml


# =========================
# 0) HARD DEFAULTS (YAML/CLI로만 override)
# =========================
LLM_MODE = "mock"  # mock | openai
OPENAI_MODEL = "gpt-5-nano"
OPENAI_API_KEY = ""  # 반드시 config.yaml에서 채울 것

OPENAI_TEMPERATURE = 0.0  # 일부 모델은 미지원일 수 있어 실제 호출에선 사용 안함
OPENAI_MAX_OUTPUT_TOKENS = 600  # responses.max_output_tokens / chat.max_completion_tokens

INDEX_DIR = "./bm25_index"
DEFAULT_DATA_DIR = "/mnt/data"

# =========================
# 0.1) Guardrail knobs (Layer1 품질 보호)
# =========================
# loss=0 금지: 최소 eps
LOSS_EPS = 1e-12

# enable_autofix=True면 QoS 프로필 자체를 보정(일관성 확보)
# warn_only=True면 QoS 프로필은 그대로 두고 issues만 추가 (Layer2 scalar는 안정화 가능)
ENABLE_AUTOFIX = True
WARN_ONLY = False

# 우선순위/클래스 sanity rules
VIDEO_HINT_WORDS = ["video", "image", "camera", "vision", "frame", "stream"]
SAFETY_HINT_WORDS = ["safety", "e-stop", "estop", "interlock", "failsafe", "human", "emergency"]
PUBSUB_HINT_WORDS = ["pubsub", "opc", "opcua", "telemetry", "sensor", "dataset", "metadata"]
# AMR map update 안정화용 힌트
MAP_HINT_WORDS = ["map", "mapping", "slam", "localization", "revision", "revisions", "revise", "update", "updates"]

# 어떤 traffic_class에 critical을 허용할지
CRITICAL_ALLOWED_CLASSES = {"control_loop", "safety_io"}
DEFAULT_PRIORITY_FOR_CLASS = {
    "control_loop": "critical",
    "safety_io": "critical",
    "pubsub_telemetry": "high",
    "video_stream": "high",
    "best_effort_misc": "medium",
}

DEFAULT_FEASIBILITY_BASIS = "hard"  # LLM이 비우면 기본


# =========================
# 1) Paths / KB mapping
# =========================
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
    "21916-g20_chunks.jsonl": "KB_3GPP_URLLC",
    "ISO13849_selected_chunks.jsonl": "KB_ISO_13849",
    "IEC61508_selected_chunks.jsonl": "KB_IEC_61508_OV",
}

ALL_TYPES = ["performance", "operational", "application", "policy"]

ALL_SUBTYPES = [
    "SAFETY",
    "TIME_SYNC",
    "DETERMINISTIC_SCHEDULING",
    "FRAME_PREEMPTION",
    "RELIABILITY_FRER",
    "STREAM_FILTERING_POLICING",
    "INDUSTRIAL_MOTION_ETHERNET",
    "CONTROL_LOOP_REALTIME",
    "OPCUA_PUBSUB",
    "WIRELESS_URLLC",
]

ALL_TRAFFIC_CLASSES = [
    "control_loop",
    "safety_io",
    "pubsub_telemetry",
    "video_stream",
    "best_effort_misc",
]

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


# =========================
# 1.1) Config (YAML only)
# =========================
def load_config_yaml(path: Optional[str]) -> dict:
    if not path:
        raise RuntimeError("--config is required (you said YAML-only).")
    if not os.path.exists(path):
        raise RuntimeError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(cfg: dict) -> None:
    global LLM_MODE, OPENAI_MODEL, OPENAI_API_KEY, OPENAI_TEMPERATURE, OPENAI_MAX_OUTPUT_TOKENS
    global INDEX_DIR, DEFAULT_DATA_DIR
    global ENABLE_AUTOFIX, WARN_ONLY, LOSS_EPS, DEFAULT_FEASIBILITY_BASIS

    llm = cfg.get("llm") or {}
    mode = llm.get("mode")
    if mode in ["mock", "openai"]:
        LLM_MODE = mode

    openai_cfg = llm.get("openai") or {}
    if openai_cfg.get("model"):
        OPENAI_MODEL = str(openai_cfg["model"])
    if openai_cfg.get("api_key") is not None:
        OPENAI_API_KEY = str(openai_cfg["api_key"]).strip()
    if openai_cfg.get("temperature") is not None:
        OPENAI_TEMPERATURE = float(openai_cfg["temperature"])
    if openai_cfg.get("max_output_tokens") is not None:
        OPENAI_MAX_OUTPUT_TOKENS = int(openai_cfg["max_output_tokens"])

    paths = cfg.get("paths") or {}
    if paths.get("index_dir"):
        INDEX_DIR = str(paths["index_dir"])
    if paths.get("data_dir"):
        DEFAULT_DATA_DIR = str(paths["data_dir"])

    guard = cfg.get("guardrails") or {}
    if guard.get("loss_eps") is not None:
        LOSS_EPS = float(guard["loss_eps"])
    if guard.get("enable_autofix") is not None:
        ENABLE_AUTOFIX = bool(guard["enable_autofix"])
    if guard.get("warn_only") is not None:
        WARN_ONLY = bool(guard["warn_only"])
    if guard.get("default_feasibility_basis") in ["hard", "target"]:
        DEFAULT_FEASIBILITY_BASIS = guard["default_feasibility_basis"]


# =========================
# 2) Tokenizer (dot/hyphen keep + dual-token)
# =========================
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")

STOPWORDS = {
    "general", "introduction", "overview", "scope", "definitions",
    "clause", "section", "annex", "table", "figure",
    "time", "timestamp", "network", "packet", "data",
}


def tokenize_with_dual(text: str) -> List[str]:
    text = (text or "").lower()
    base_tokens = TOKEN_RE.findall(text)

    out: List[str] = []
    for tok in base_tokens:
        if tok in STOPWORDS:
            continue
        out.append(tok)

        norm = tok.replace(".", "").replace("-", "")
        if norm != tok and len(norm) >= 4 and norm not in STOPWORDS:
            out.append(norm)

    return out


# =========================
# 3) JSONL loader / BM25 index
# =========================
def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


@dataclass
class ChunkMeta:
    kb: str
    chunk_id: str
    prefix: str
    text: str
    metadata: Dict[str, Any]


def index_path(kb_id: str) -> str:
    return os.path.join(INDEX_DIR, f"{kb_id}.pkl")


def discover_sources(data_dir: str) -> Dict[str, str]:
    print("[DEBUG] scanning data_dir:", os.path.abspath(data_dir))
    paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
    print("[DEBUG] found jsonl files:", len(paths))

    found: Dict[str, str] = {}
    for p in paths:
        base = os.path.basename(p)
        if base in KNOWN_FILES_TO_KB:
            kb_id = KNOWN_FILES_TO_KB[base]
            found[kb_id] = p
    return found


def build_bm25_for_kb(kb_id: str, jsonl_path: str, overwrite: bool = False) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    out_path = index_path(kb_id)

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {kb_id} already indexed: {out_path}")
        return

    chunks = load_jsonl(jsonl_path)
    tokenized_docs: List[List[str]] = []
    metas: List[ChunkMeta] = []

    for c in chunks:
        prefix = (c.get("prefix") or "").strip()
        text = (c.get("text") or "").strip()
        full = (prefix + "\n" + text).strip() if prefix else text

        tokenized_docs.append(tokenize_with_dual(full))
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


def build_all(data_dir: str, overwrite: bool = False):
    sources = discover_sources(data_dir)
    if not sources:
        raise RuntimeError(f"No known JSONL files found in {data_dir}. Check filenames.")

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


def load_bm25_index(kb_id: str):
    path = index_path(kb_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing index for {kb_id}. Run build first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================
# 4) Retrieval / Routing
# =========================
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


def retrieve_multi(kb_ids: List[str], query: str, top_k_per_kb: int = 6, final_top_k: int = 8):
    all_hits = []
    for kb in kb_ids:
        all_hits.extend(retrieve_kb(kb, query, top_k=top_k_per_kb))
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:final_top_k]


def _contains_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def route_kbs(intent_type: str, subtypes: List[str], intent: str, k: int = 2) -> List[str]:
    score: Dict[str, float] = {}

    # subtype 우선
    for st in subtypes:
        for rank, kb in enumerate(SUBTYPE_TO_KB_PRIORITY.get(st, [])):
            score[kb] = score.get(kb, 0.0) + (10.0 - rank)

    # type 보조
    for kb in TYPE_BOOST.get(intent_type, []):
        score[kb] = score.get(kb, 0.0) + 2.0

    # ✅ AMR map update 류는 OPC UA PubSub KB를 살려주는게 안정적
    if _contains_any(intent, MAP_HINT_WORDS) and not _contains_any(intent, VIDEO_HINT_WORDS):
        score["KB_OPCUA_PUBSUB"] = score.get("KB_OPCUA_PUBSUB", 0.0) + 8.0

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    kbs = [kb for kb, _ in ranked[:k]]
    return kbs or ["KB_TSN_8021AS", "KB_TSN_8021QBV"]


# =========================
# 5) LLM schemas (+ additionalProperties fix)
# =========================
_EVIDENCE_REF_PATTERN = r"^ref=[A-Z0-9_]+:[A-Za-z0-9_.\-]+$"

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent_type": {"type": "string", "enum": ALL_TYPES},
        "subtypes": {
            "type": "array",
            "items": {"type": "string", "enum": ALL_SUBTYPES},
            "minItems": 1,
            "maxItems": 4,
        },
        "rationale_short": {"type": "string"}
    },
    "required": ["intent_type", "subtypes", "rationale_short"],
}

QOS_SCHEMA = {
    "type": "object",
    "properties": {
        "traffic_class": {"type": "string", "enum": ALL_TRAFFIC_CLASSES},
        "feasibility_basis": {"type": "string", "enum": ["target", "hard"]},

        "latency_ms_target": {"type": "number"},
        "latency_ms_hard": {"type": "number"},
        "jitter_ms_target": {"type": "number"},
        "jitter_ms_hard": {"type": "number"},
        "loss_rate_target": {"type": "number"},
        "loss_rate_hard": {"type": "number"},
        "bandwidth_mbps_min": {"type": "number"},

        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "qos_rationale_short": {"type": "string"},

        "operational_constraints": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string", "pattern": _EVIDENCE_REF_PATTERN},
            "maxItems": 8
        },
    },
    "required": [
        "traffic_class",
        "feasibility_basis",
        "latency_ms_target", "latency_ms_hard",
        "jitter_ms_target", "jitter_ms_hard",
        "loss_rate_target", "loss_rate_hard",
        "bandwidth_mbps_min",
        "priority", "qos_rationale_short",
        "operational_constraints", "evidence_refs"
    ],
}


def _force_additional_properties_false(schema: dict) -> dict:
    s = deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            t = node.get("type")
            if t == "object":
                node["additionalProperties"] = False
                props = node.get("properties") or {}
                for _, v in props.items():
                    walk(v)

            if "items" in node:
                walk(node["items"])
            for k in ["anyOf", "oneOf", "allOf"]:
                if k in node and isinstance(node[k], list):
                    for it in node[k]:
                        walk(it)

        elif isinstance(node, list):
            for it in node:
                walk(it)

    walk(s)
    return s


# =========================
# 5.1) LLM call
# =========================
_SUBTYPES_LINE_RE = re.compile(r"Predicted subtypes:\s*(\[[^\]]*\])", re.IGNORECASE)


def _parse_predicted_subtypes_from_prompt(user_prompt: str) -> List[str]:
    if not user_prompt:
        return []
    m = _SUBTYPES_LINE_RE.search(user_prompt)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(1))
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, str)]
    except Exception:
        return []
    return []


def _mock_pick_traffic_class(intent_text: str, subtypes: List[str]) -> str:
    t = (intent_text or "").lower()
    if any(w in t for w in VIDEO_HINT_WORDS):
        return "video_stream"
    if any(w in t for w in PUBSUB_HINT_WORDS) or any(w in t for w in MAP_HINT_WORDS):
        return "pubsub_telemetry"
    if any(w in t for w in SAFETY_HINT_WORDS):
        return "safety_io"
    if "control" in t or "loop" in t or "latency" in t or "robot" in t:
        return "control_loop"
    if "SAFETY" in subtypes:
        return "safety_io"
    if "OPCUA_PUBSUB" in subtypes:
        return "pubsub_telemetry"
    return "best_effort_misc"


def call_llm(system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
    global LLM_MODE, OPENAI_MODEL, OPENAI_API_KEY, OPENAI_MAX_OUTPUT_TOKENS

    if LLM_MODE == "mock":
        text = (user_prompt or "").lower()
        subtypes: List[str] = []
        intent_type = "performance"

        if any(k in text for k in ["synchronization", "sync", "clock", "gptp", "drift", "grandmaster"]):
            subtypes.append("TIME_SYNC")
            intent_type = "operational"
        if any(k in text for k in ["deterministic", "cycle", "gate", "scheduled", "time-aware", "tas"]):
            subtypes.append("DETERMINISTIC_SCHEDULING")
            if intent_type != "policy":
                intent_type = "operational"
        if any(k in text for k in ["preemption", "guard band"]):
            subtypes.append("FRAME_PREEMPTION")
        if any(k in text for k in ["no interruption", "without interruption", "redundancy", "replication",
                                   "elimination", "no loss", "retransmission"]):
            subtypes.append("RELIABILITY_FRER")
        if any(k in text for k in ["isolate", "blocked", "restricted", "authorized", "violation", "police", "filter"]):
            subtypes.append("STREAM_FILTERING_POLICING")
            intent_type = "policy"
        if any(k in text for k in ["safety", "emergency", "e-stop", "interlock", "human-detection", "failsafe", "human"]):
            subtypes.append("SAFETY")
            intent_type = "operational"
        if any(k in text for k in MAP_HINT_WORDS + PUBSUB_HINT_WORDS):
            subtypes.append("OPCUA_PUBSUB")
            intent_type = "operational"

        if not subtypes:
            subtypes = ["CONTROL_LOOP_REALTIME"]
            intent_type = "application"

        if "###task:classify###" in text:
            return {
                "intent_type": intent_type,
                "subtypes": subtypes[:4],
                "rationale_short": "mock classification based on keywords"
            }

        if "###task:generate_qos###" in text:
            predicted = _parse_predicted_subtypes_from_prompt(user_prompt)
            predicted = [s for s in predicted if s in ALL_SUBTYPES]
            if predicted:
                subtypes = predicted

            traffic_class = _mock_pick_traffic_class(user_prompt, subtypes)
            return {
                "traffic_class": traffic_class,
                "feasibility_basis": "hard",
                "latency_ms_target": 10.0,
                "latency_ms_hard": 30.0,
                "jitter_ms_target": 1.0,
                "jitter_ms_hard": 3.0,
                "loss_rate_target": 1e-6,
                "loss_rate_hard": 1e-4,
                "bandwidth_mbps_min": 1.0,
                "priority": DEFAULT_PRIORITY_FOR_CLASS.get(traffic_class, "high"),
                "qos_rationale_short": "mock: conservative defaults",
                "operational_constraints": [],
                "evidence_refs": ["ref=KB_TSN_8021QBV:MOCK"]
            }

        return {
            "intent_type": intent_type,
            "subtypes": subtypes[:4],
            "rationale_short": "mock fallback"
        }

    if LLM_MODE == "openai":
        api_key = (OPENAI_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty. Put it in config.yaml: llm.openai.api_key")

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Install openai first: pip install openai") from e

        client = OpenAI(api_key=api_key)
        schema_fixed = _force_additional_properties_false(json_schema)

        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "layer1_schema",
                        "strict": True,
                        "schema": schema_fixed,
                    }
                },
                max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                store=False,
            )
            raw = getattr(resp, "output_text", None) or ""
            return json.loads(raw.strip())

        except Exception as e:
            msg = str(e).lower()
            if ("insufficient_quota" in msg) or ("exceeded your current quota" in msg) or ("error code: 429" in msg):
                raise RuntimeError(
                    "OpenAI 429 insufficient_quota: 크레딧 소진/월 사용한도(프로젝트 버짓 포함) 초과입니다. "
                    "Billing/Usage/Limits 확인 후 다시 실행하세요."
                ) from e

            print("[DEBUG] responses.create failed:", repr(e))
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                )
                return json.loads((resp.choices[0].message.content or "").strip())
            except Exception as e2:
                raise RuntimeError(
                    f"OpenAI call failed (Responses + fallback). err1={e} err2={e2}"
                ) from e2

    raise RuntimeError(f"Unknown LLM_MODE={LLM_MODE}")


# =========================
# 5.2) Classifier + postprocess (AMR 안정화)
# =========================
def _postprocess_subtypes(intent: str, subs: List[str]) -> List[str]:
    """
    ✅ AMR map update 안정화:
    - map/revision/update 힌트인데 video 힌트가 없으면 OPCUA_PUBSUB를 보강(가능하면)
    - 최대 3개 유지
    """
    out = [s for s in subs if s in ALL_SUBTYPES]
    t = (intent or "").lower()

    is_mapish = _contains_any(t, MAP_HINT_WORDS) and not _contains_any(t, VIDEO_HINT_WORDS)
    is_pubsubish = _contains_any(t, PUBSUB_HINT_WORDS)

    if (is_mapish or is_pubsubish) and ("OPCUA_PUBSUB" not in out):
        out.append("OPCUA_PUBSUB")

    # 최대 3개 유지(뒤에 붙은 OPCUA_PUBSUB가 잘리면 의미 없음 → 우선순위 조정)
    # control/safety처럼 강한 subtype이 있으면 유지, 나머지 중복 제거
    preferred = ["SAFETY", "CONTROL_LOOP_REALTIME", "DETERMINISTIC_SCHEDULING", "OPCUA_PUBSUB",
                 "WIRELESS_URLLC", "RELIABILITY_FRER", "FRAME_PREEMPTION",
                 "INDUSTRIAL_MOTION_ETHERNET", "TIME_SYNC", "STREAM_FILTERING_POLICING"]
    dedup = []
    for p in preferred:
        if p in out and p not in dedup:
            dedup.append(p)
    for s in out:
        if s not in dedup:
            dedup.append(s)

    return dedup[:3] if dedup else ["CONTROL_LOOP_REALTIME"]


def classify_intent(intent: str) -> dict:
    system_prompt = (
        "You are an intent classifier for Industry 5.0 smart factory networking.\n"
        "Classify the intent into one main type and 1-4 subtypes.\n"
        "Return ONLY JSON.\n"
    )
    user_prompt = (
        "###TASK:CLASSIFY###\n"
        f"Intent: {intent}\n"
        f"Allowed types: {ALL_TYPES}\n"
        f"Allowed subtypes: {ALL_SUBTYPES}\n"
        "Output JSON fields: intent_type, subtypes, rationale_short\n"
    )

    out = call_llm(system_prompt, user_prompt, CLASSIFIER_SCHEMA)

    if "rationale_short" not in out or not isinstance(out["rationale_short"], str):
        out["rationale_short"] = ""

    if out.get("intent_type") not in ALL_TYPES:
        out["intent_type"] = "performance"

    subs = out.get("subtypes") or []
    subs = [s for s in subs if s in ALL_SUBTYPES]
    if not subs:
        subs = ["CONTROL_LOOP_REALTIME"]

    subs = _postprocess_subtypes(intent, subs)
    out["subtypes"] = subs
    return out


def build_evidence_block(hits: List[dict]) -> Tuple[str, List[str]]:
    blocks = []
    refs: List[str] = []
    for i, h in enumerate(hits, 1):
        ref_id = f"{h['kb']}:{h['chunk_id']}"
        refs.append(f"ref={ref_id}")

        prefix = (h.get("prefix") or "").strip()
        text = (h.get("text") or "").strip()
        snippet = text[:900]

        blocks.append(f"[E{i}] ref={ref_id}\nprefix: {prefix}\ntext: {snippet}\n")
    return "\n".join(blocks), refs


def infer_traffic_class_hint(intent: str, subtypes: List[str]) -> Tuple[Optional[str], str]:
    """
    ✅ 프롬프트에 주입할 traffic_class 힌트(강제는 아님).
    - map/update/revision 이고 video 힌트 없으면 pubsub_telemetry 우선
    - safety -> safety_io
    - control loop -> control_loop
    - image/video -> video_stream
    """
    t = (intent or "").lower()

    if _contains_any(t, SAFETY_HINT_WORDS) or ("SAFETY" in subtypes):
        return "safety_io", "safety/interlock intent"
    if _contains_any(t, VIDEO_HINT_WORDS):
        return "video_stream", "image/video intent"
    if (_contains_any(t, MAP_HINT_WORDS) and not _contains_any(t, VIDEO_HINT_WORDS)) or ("OPCUA_PUBSUB" in subtypes):
        return "pubsub_telemetry", "map/metadata/pubsub intent"
    if ("CONTROL_LOOP_REALTIME" in subtypes) or any(k in t for k in ["control", "loop", "servo", "latency"]):
        return "control_loop", "control-loop intent"
    return None, "no strong hint"


def generate_qos(intent: str, intent_type: str, subtypes: List[str], hits: List[dict]) -> dict:
    evidence_text, refs = build_evidence_block(hits)
    hint_class, hint_reason = infer_traffic_class_hint(intent, subtypes)

    system_prompt = (
        "You translate Industry 5.0 intents into quantitative QoS requirements.\n"
        "IMPORTANT:\n"
        "1) Decide traffic_class.\n"
        "2) Output both 'target' and 'hard' bounds (hard >= target).\n"
        "3) Choose feasibility_basis: 'hard' if strict guarantees are required; otherwise 'target'.\n"
        "Use evidence excerpts for grounding.\n"
        "If evidence does not contain explicit numeric QoS, choose conservative-but-plausible values.\n"
        "Return ONLY valid JSON.\n"
    )

    # ✅ AMR 안정화를 위해 traffic_class_hint를 강하게 제시
    traffic_hint_line = ""
    if hint_class:
        traffic_hint_line = (
            f"TRAFFIC_CLASS_HINT: Prefer '{hint_class}' ({hint_reason}). "
            f"Only choose a different class if the intent explicitly indicates it.\n"
        )

    user_prompt = (
        "###TASK:GENERATE_QOS###\n"
        f"Intent: {intent}\n"
        f"Predicted type: {intent_type}\n"
        f"Predicted subtypes: {subtypes}\n\n"
        f"{traffic_hint_line}\n"
        "Choose exactly ONE traffic_class from:\n"
        f"{ALL_TRAFFIC_CLASSES}\n\n"
        "Traffic class guidance (soft reference):\n"
        "- control_loop: very low latency/jitter; low bandwidth.\n"
        "- safety_io: reliability + bounded latency; safety interlocks/e-stop.\n"
        "- pubsub_telemetry: moderate latency/jitter; periodic updates; metadata/revision updates.\n"
        "- video_stream: high bandwidth; moderate latency; jitter matters.\n"
        "- best_effort_misc: relaxed.\n\n"
        "Evidence:\n"
        f"{evidence_text}\n\n"
        "STRICT FORMAT RULES:\n"
        "- evidence_refs MUST be an array of strings in the form: \"ref=KB_ID:CHUNK_ID\".\n"
        "- Use ONLY refs that appear in Evidence blocks above.\n"
        "- Ensure: hard >= target for latency/jitter/loss. (hard is looser, target tighter)\n"
        "- loss_rate must be > 0 (use a small positive value, not 0).\n\n"
        "Output JSON with fields:\n"
        "- traffic_class\n"
        "- feasibility_basis (target|hard)\n"
        "- latency_ms_target, latency_ms_hard\n"
        "- jitter_ms_target, jitter_ms_hard\n"
        "- loss_rate_target, loss_rate_hard\n"
        "- bandwidth_mbps_min\n"
        "- priority (low|medium|high|critical)\n"
        "- qos_rationale_short (<= 30 words)\n"
        "- operational_constraints (string list)\n"
        "- evidence_refs\n"
    )

    out = call_llm(system_prompt, user_prompt, QOS_SCHEMA)

    if "evidence_refs" not in out or not isinstance(out["evidence_refs"], list) or len(out["evidence_refs"]) == 0:
        out["evidence_refs"] = refs[: min(5, len(refs))]

    if "operational_constraints" not in out or not isinstance(out["operational_constraints"], list):
        out["operational_constraints"] = []

    if "qos_rationale_short" not in out or not isinstance(out["qos_rationale_short"], str):
        out["qos_rationale_short"] = ""

    if out.get("feasibility_basis") not in ["target", "hard"]:
        out["feasibility_basis"] = DEFAULT_FEASIBILITY_BASIS

    return out


# =========================
# 6) Quality / Guardrail validation (+ optional autofix)
# =========================
def _should_autofix() -> bool:
    if WARN_ONLY:
        return False
    return bool(ENABLE_AUTOFIX)


def _clamp_loss_positive(v: float) -> float:
    return LOSS_EPS if v <= 0.0 else v


def _ensure_hard_ge_target(qos: dict, key_target: str, key_hard: str, issues: List[str]) -> None:
    try:
        t = float(qos.get(key_target, 0.0))
        h = float(qos.get(key_hard, 0.0))
    except Exception:
        issues.append(f"{key_target}/{key_hard} not numeric.")
        return

    if h < t:
        issues.append(f"{key_hard} < {key_target} (hard should be >= target).")
        if _should_autofix():
            qos[key_hard] = t


def _default_priority_for_class(tc: str) -> str:
    return DEFAULT_PRIORITY_FOR_CLASS.get(tc, "high")


def infer_expected_class(intent: str, subtypes: List[str]) -> Optional[str]:
    t = (intent or "").lower()

    # safety > video > pubsub(map) > control
    if _contains_any(t, SAFETY_HINT_WORDS) or ("SAFETY" in subtypes):
        return "safety_io"
    if _contains_any(t, VIDEO_HINT_WORDS):
        return "video_stream"
    if (_contains_any(t, MAP_HINT_WORDS) and not _contains_any(t, VIDEO_HINT_WORDS)) or ("OPCUA_PUBSUB" in subtypes):
        return "pubsub_telemetry"
    if ("CONTROL_LOOP_REALTIME" in subtypes) or any(k in t for k in ["control", "loop", "servo", "latency"]):
        return "control_loop"
    return None


def validate_profile(intent: str, intent_type: str, subtypes: List[str], qos: dict) -> List[str]:
    issues: List[str] = []

    # snapshot (버그 방지: "got"은 이 값을 기준으로 찍음)
    original_tc = qos.get("traffic_class")
    original_pr = qos.get("priority")

    # ---- 1) loss_rate 0 금지: 항상 경고, 필요시 보정
    for k in ["loss_rate_target", "loss_rate_hard"]:
        try:
            v = float(qos.get(k, LOSS_EPS))
        except Exception:
            issues.append(f"{k} not numeric.")
            continue

        if v <= 0.0:
            issues.append(f"{k} is 0 or negative (invalid); clamping to eps in Layer2 input.")
            if _should_autofix():
                qos[k] = LOSS_EPS

    # ---- 2) hard >= target
    _ensure_hard_ge_target(qos, "latency_ms_target", "latency_ms_hard", issues)
    _ensure_hard_ge_target(qos, "jitter_ms_target", "jitter_ms_hard", issues)
    _ensure_hard_ge_target(qos, "loss_rate_target", "loss_rate_hard", issues)

    # ---- 3) traffic_class sanity (+ AMR map update 안정화)
    expected = infer_expected_class(intent, subtypes)
    current_tc = qos.get("traffic_class")

    if expected and current_tc and expected != current_tc:
        # ✅ expected/got 역전 방지: "got"은 current_tc(보정 전)로 고정
        issues.append(f"traffic_class mismatch: expected '{expected}' but got '{current_tc}'.")
        if _should_autofix():
            qos["traffic_class"] = expected
            issues.append(f"autofixed traffic_class -> '{expected}'.")

    # ---- 4) priority sanity
    tc = qos.get("traffic_class")
    pr = qos.get("priority")

    if tc in ALL_TRAFFIC_CLASSES:
        default_p = _default_priority_for_class(tc)

        if pr not in ["low", "medium", "high", "critical"]:
            issues.append("priority invalid enum.")
            if _should_autofix():
                qos["priority"] = default_p
        else:
            if (tc not in CRITICAL_ALLOWED_CLASSES) and (pr == "critical"):
                issues.append(f"priority 'critical' unusual for traffic_class='{tc}'.")
                if _should_autofix():
                    qos["priority"] = default_p
                    issues.append(f"autofixed priority -> '{default_p}'.")

    # ---- 5) subtype-specific sanity (기존)
    try:
        jit_t = float(qos.get("jitter_ms_target", 1e9))
        loss_t = float(qos.get("loss_rate_target", 1.0))
    except Exception:
        jit_t, loss_t = 1e9, 1.0

    if "TIME_SYNC" in subtypes:
        if jit_t > 1.0:
            issues.append("TIME_SYNC jitter_target too high (>1ms).")
        if loss_t > 1e-3:
            issues.append("TIME_SYNC loss_target too high (>1e-3).")

    if intent_type == "policy":
        try:
            lat_t = float(qos.get("latency_ms_target", 1e9))
        except Exception:
            lat_t = 1e9
        if lat_t < 1.0 and "SAFETY" not in subtypes:
            issues.append("Policy with extremely low latency target (<1ms) looks suspicious.")

    # ---- 6) feasibility_basis 유효성
    fb = qos.get("feasibility_basis")
    if fb not in ["target", "hard"]:
        issues.append("feasibility_basis invalid; defaulted.")
        if _should_autofix():
            qos["feasibility_basis"] = DEFAULT_FEASIBILITY_BASIS

    return issues


# =========================
# 7) Layer2 handoff: 단일 스칼라 추출
# =========================
def extract_layer2_scalar(qos: dict) -> dict:
    basis = qos.get("feasibility_basis")
    if basis not in ["target", "hard"]:
        basis = DEFAULT_FEASIBILITY_BASIS

    suffix = "hard" if basis == "hard" else "target"

    def pick(name: str, default: float) -> float:
        key = f"{name}_{suffix}"
        try:
            return float(qos.get(key, default))
        except Exception:
            return float(default)

    lat = pick("latency_ms", 1e9)
    jit = pick("jitter_ms", 1e9)

    # ✅ Layer2 입력은 항상 loss>0으로 clamp (warn_only든 아니든)
    loss_raw = pick("loss_rate", 1.0)
    loss = _clamp_loss_positive(loss_raw)

    bw = float(qos.get("bandwidth_mbps_min", 0.0) or 0.0)

    return {
        "basis": basis,
        "traffic_class": qos.get("traffic_class", "best_effort_misc"),
        "priority": qos.get("priority", "high"),
        "latency_ms": lat,
        "jitter_ms": jit,
        "loss_rate": loss,
        "bandwidth_mbps": bw,
    }


# =========================
# 8) Main pipeline
# =========================
def layer1(intent: str, final_top_k: int = 8, top_k_per_kb: int = 6, routed_k: int = 2):
    indexed = set(list_indexed_kbs())
    if not indexed:
        raise RuntimeError("No BM25 indexes found. Run build first.")

    t0 = time.time()

    cls = classify_intent(intent)
    intent_type = cls["intent_type"]
    subtypes = cls["subtypes"]

    kbs = route_kbs(intent_type, subtypes, intent=intent, k=routed_k)
    kbs = [kb for kb in kbs if kb in indexed] or [
        kb for kb in ["KB_TSN_8021AS", "KB_TSN_8021QBV"] if kb in indexed
    ]

    hits = retrieve_multi(kbs, intent, top_k_per_kb=top_k_per_kb, final_top_k=final_top_k)
    qos = generate_qos(intent, intent_type, subtypes, hits)

    issues = validate_profile(intent, intent_type, subtypes, qos)
    layer2 = extract_layer2_scalar(qos)

    t1 = time.time()

    return {
        "intent": intent,
        "classification": cls,
        "searched_kbs": kbs,
        "evidence": hits,
        "qos_profile": qos,
        "validation_issues": issues,
        "layer2_scalar": layer2,
        "latency_sec": round(t1 - t0, 3),
        "llm_mode": LLM_MODE,
        "openai_model": OPENAI_MODEL if LLM_MODE == "openai" else None,
    }


# =========================
# CLI
# =========================
def cmd_build(args):
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    build_all(data_dir, overwrite=args.overwrite)


def cmd_run(args):
    global LLM_MODE, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_OUTPUT_TOKENS

    if args.llm is not None:
        LLM_MODE = args.llm
    if args.model is not None:
        OPENAI_MODEL = args.model
    if args.temperature is not None:
        OPENAI_TEMPERATURE = args.temperature
    if args.max_output_tokens is not None:
        OPENAI_MAX_OUTPUT_TOKENS = args.max_output_tokens

    result = layer1(
        intent=args.q,
        final_top_k=args.k,
        top_k_per_kb=args.top_k_per_kb,
        routed_k=args.routed_k
    )

    print("\n==== LAYER1 RESULT ====")
    print("intent:", result["intent"])
    print("llm_mode:", result["llm_mode"])
    if result["openai_model"]:
        print("openai_model:", result["openai_model"])
    print("classification:", result["classification"])
    print("searched_kbs:", result["searched_kbs"])
    print("validation_issues:", result["validation_issues"])
    print("qos_profile:", json.dumps(result["qos_profile"], indent=2, ensure_ascii=False))
    print("layer2_scalar:", json.dumps(result["layer2_scalar"], indent=2, ensure_ascii=False))

    print("\n==== EVIDENCE (top hits) ====")
    for i, h in enumerate(result["evidence"], 1):
        prefix = (h.get("prefix") or "").replace("\n", " ").strip()
        snippet = (h.get("text") or "").replace("\n", " ").strip()
        snippet = snippet[:220] + (" ..." if len(snippet) > 220 else "")
        print(f"[{i}] {h['kb']} score={h['score']:.3f} chunk_id={h['chunk_id']}")
        if prefix:
            print("    prefix:", prefix)
        print("    text:", snippet)


def main():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default=None)

    parser = argparse.ArgumentParser(parents=[parent])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", parents=[parent])
    p_build.add_argument("--data-dir", default=None)
    p_build.add_argument("--overwrite", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_run = sub.add_parser("run", parents=[parent])
    p_run.add_argument("--q", required=True)
    p_run.add_argument("--k", type=int, default=8)
    p_run.add_argument("--top-k-per-kb", type=int, default=6)
    p_run.add_argument("--routed-k", type=int, default=2)

    p_run.add_argument("--llm", choices=["mock", "openai"], default=None)
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--max-output-tokens", type=int, default=None)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required. e.g. --config config.yaml")

    cfg = load_config_yaml(args.config)
    apply_config(cfg)

    args.func(args)


if __name__ == "__main__":
    main()
