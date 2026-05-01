# edge_profiles.py
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

# edge_attr = [bw, lat, jit, loss, queue, interf, tsn, frer]
EDGE_DIM = 8

@dataclass
class ScenarioParams:
    congestion_level: float = 0.0      # 0~1
    interference_level: float = 0.0    # 0~1
    failure_mode: str = "none"         # "none" | "disable_frer" | "disable_tsn"

def _u(a: float, b: float) -> float:
    return random.uniform(a, b)

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# (src_type, dst_type) 무방향
EDGE_PROFILES: Dict[Tuple[str, str], Dict] = {
    ("tsn", "tsn"):   dict(bw=(200,1000), lat=(0.03,0.15), jit=(0.003,0.03), loss=(1e-8,1e-6), tsn=1, frer=1),
    ("plc", "tsn"):   dict(bw=(1,10),     lat=(0.05,0.3),  jit=(0.005,0.08), loss=(1e-7,1e-5), tsn=1, frer=1),
    ("robot", "tsn"): dict(bw=(5,100),    lat=(0.05,0.5),  jit=(0.01,0.15),  loss=(1e-7,1e-5), tsn=1, frer=1),
    ("vision","tsn"): dict(bw=(50,500),   lat=(0.1,1.0),   jit=(0.05,0.4),   loss=(1e-6,1e-4), tsn=1, frer=0),
    ("ap", "tsn"):    dict(bw=(50,500),   lat=(0.2,2.0),   jit=(0.05,1.0),   loss=(1e-5,1e-3), tsn=0, frer=0),
    ("agv", "ap"):    dict(bw=(5,80),     lat=(1.0,10.0),  jit=(0.5,5.0),    loss=(1e-4,1e-2), tsn=0, frer=0),
    ("robot","ap"):   dict(bw=(5,80),     lat=(1.0,10.0),  jit=(0.5,5.0),    loss=(1e-4,1e-2), tsn=0, frer=0),
    ("vision","ap"):  dict(bw=(80,300),   lat=(1.0,6.0),   jit=(0.2,2.0),    loss=(1e-4,1e-2), tsn=0, frer=0),
}

def sample_edge_attr(src_type: str, dst_type: str, link_kind: str, scenario: ScenarioParams) -> List[float]:
    key = (src_type, dst_type)
    if key not in EDGE_PROFILES:
        key = (dst_type, src_type)

    prof = EDGE_PROFILES.get(key, dict(
        bw=(10,100), lat=(0.5,3.0), jit=(0.1,1.0), loss=(1e-6,1e-4), tsn=0, frer=0
    ))

    bw   = _u(*prof["bw"])
    lat  = _u(*prof["lat"])
    jit  = _u(*prof["jit"])
    loss = _u(*prof["loss"])

    # congestion: 모두에 영향
    queue = _clip(_u(0.0, 0.3) + scenario.congestion_level * _u(0.2, 1.0), 0.0, 1.0)
    lat *= (1.0 + scenario.congestion_level * _u(0.2, 1.0))
    jit *= (1.0 + scenario.congestion_level * _u(0.3, 1.5))

    # interference: wireless에서 크게
    if link_kind == "wireless":
        interf = _clip(_u(0.1,0.6) + scenario.interference_level * _u(0.5,1.5), 0.0, 1.0)
        loss *= (1.0 + scenario.interference_level * _u(1.0, 5.0))
        jit  *= (1.0 + scenario.interference_level * _u(0.5, 3.0))
        lat  *= (1.0 + scenario.interference_level * _u(0.1, 1.0))
    else:
        interf = _clip(_u(0.0,0.1) + scenario.interference_level * _u(0.0,0.3), 0.0, 1.0)

    tsn_cap  = float(prof["tsn"])
    frer_cap = float(prof["frer"])

    if scenario.failure_mode == "disable_frer":
        frer_cap = 0.0
    elif scenario.failure_mode == "disable_tsn":
        tsn_cap = 0.0

    return [bw, lat, jit, loss, queue, interf, tsn_cap, frer_cap]