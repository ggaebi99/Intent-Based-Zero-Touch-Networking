# topology_templates.py
from dataclasses import dataclass
from typing import List, Tuple, Set

NODE_TYPES = ["robot", "agv", "plc", "vision", "ap", "tsn"]  # node feature one-hot 기준

@dataclass(frozen=True)
class TopologyTemplate:
    name: str
    node_types: List[str]                 # len=N
    edges: List[Tuple[int, int, str]]     # (u,v,kind) kind in {"wired","wireless"}


def _add_edge(edges: List[Tuple[int,int,str]], seen: Set[Tuple[int,int,str]], u: int, v: int, kind: str):
    a, b = (u, v) if u < v else (v, u)
    key = (a, b, kind)
    if key in seen:
        return
    seen.add(key)
    edges.append((a, b, kind))


def build_T1(n: int = 8) -> TopologyTemplate:
    # T1: cell-level TSN
    assert 6 <= n <= 8
    base = ["tsn", "tsn", "plc", "robot", "robot", "robot", "vision", "robot"]
    node_types = base[:n]

    edges, seen = [], set()

    # core-backbone
    _add_edge(edges, seen, 0, 1, "wired")

    # access(tsn=1)에 말단 연결
    for i in range(2, n):
        _add_edge(edges, seen, 1, i, "wired")

    robots = [i for i, t in enumerate(node_types) if t == "robot"]
    if len(robots) >= 2:
        _add_edge(edges, seen, robots[0], robots[1], "wired")
    if len(robots) >= 3:
        _add_edge(edges, seen, robots[1], robots[2], "wired")

    # 우회 링크(대체 경로)
    visions = [i for i, t in enumerate(node_types) if t == "vision"]
    if visions:
        _add_edge(edges, seen, 0, visions[0], "wired")
    if robots:
        _add_edge(edges, seen, 0, robots[-1], "wired")

    return TopologyTemplate("T1_cell_tsn", node_types, edges)


def build_T2(n: int = 12) -> TopologyTemplate:
    # T2: TSN + AP + AGV
    assert 10 <= n <= 12
    base = ["tsn","tsn","tsn","plc","robot","robot","robot","vision","ap","agv","agv","robot"]
    node_types = base[:n]

    edges, seen = [], set()
    _add_edge(edges, seen, 0, 1, "wired")
    _add_edge(edges, seen, 1, 2, "wired")

    # devices to tsn access(2)
    for i in range(3, min(n, 8)):
        _add_edge(edges, seen, 2, i, "wired")

    ap_idx = node_types.index("ap")
    _add_edge(edges, seen, 1, ap_idx, "wired")

    # AGV wireless to AP
    for i, t in enumerate(node_types):
        if t == "agv":
            _add_edge(edges, seen, i, ap_idx, "wireless")

    if n > 11 and node_types[11] == "robot":
        _add_edge(edges, seen, 2, 11, "wired")

    # 우회 링크
    _add_edge(edges, seen, 2, ap_idx, "wired")  # AP dual-homing
    if "vision" in node_types:
        vision_idx = node_types.index("vision")
        _add_edge(edges, seen, 1, vision_idx, "wired")

    return TopologyTemplate("T2_hybrid_wifi", node_types, edges)


def build_T3(n: int = 24, cells: int = 3) -> TopologyTemplate:
    # T3: zone backbone + multiple cells
    assert 20 <= n <= 30
    node_types = ["tsn","tsn","ap","vision"]  # 0,1 core, 2 ap, 3 vision
    edges, seen = [], set()

    _add_edge(edges, seen, 0, 1, "wired")
    _add_edge(edges, seen, 1, 2, "wired")
    _add_edge(edges, seen, 1, 3, "wired")

    idx = len(node_types)
    cell_access = []

    for _ in range(cells):
        if idx + 5 > n:
            break
        tsn_a = idx
        plc = idx + 1
        r1  = idx + 2
        r2  = idx + 3
        agv = idx + 4
        node_types += ["tsn","plc","robot","robot","agv"]

        _add_edge(edges, seen, 1, tsn_a, "wired")
        _add_edge(edges, seen, tsn_a, plc, "wired")
        _add_edge(edges, seen, tsn_a, r1, "wired")
        _add_edge(edges, seen, tsn_a, r2, "wired")
        _add_edge(edges, seen, agv, 2, "wireless")  # agv->zone ap

        cell_access.append(tsn_a)
        idx += 5

    # fill remain nodes (간단 연결)
    while len(node_types) < n:
        node_types.append("robot" if (len(node_types) % 2 == 0) else "vision")
        _add_edge(edges, seen, 1, len(node_types)-1, "wired")

    # 우회 링크
    if len(cell_access) >= 2:
        _add_edge(edges, seen, cell_access[0], cell_access[1], "wired")
    if len(cell_access) >= 3:
        _add_edge(edges, seen, cell_access[1], cell_access[2], "wired")
    if len(cell_access) >= 1:
        _add_edge(edges, seen, 0, cell_access[0], "wired")
    if len(cell_access) >= 2:
        _add_edge(edges, seen, 0, cell_access[-1], "wired")
    _add_edge(edges, seen, 0, 2, "wired")  # core<->ap 우회

    return TopologyTemplate("T3_zone_backbone", node_types, edges)


def build_T4(n: int = 48, zones: int = 4) -> TopologyTemplate:
    # T4: multi-zone + edge cloud
    assert 40 <= n <= 60
    node_types = ["tsn","tsn","tsn","vision","vision","plc"]  # 0,1,2 core, 3,4 vision, 5 plc
    edges, seen = [], set()

    _add_edge(edges, seen, 0, 1, "wired")
    _add_edge(edges, seen, 1, 2, "wired")
    _add_edge(edges, seen, 0, 2, "wired")
    _add_edge(edges, seen, 1, 3, "wired")
    _add_edge(edges, seen, 2, 4, "wired")
    _add_edge(edges, seen, 1, 5, "wired")

    idx = len(node_types)
    zone_a, zone_b, zone_ap = [], [], []

    for _ in range(zones):
        if idx + 10 > n:
            break
        tsn_a = idx
        tsn_b = idx + 1
        ap    = idx + 2
        plc   = idx + 3
        robots = [idx+4, idx+5, idx+6, idx+7]
        agvs   = [idx+8, idx+9]
        node_types += ["tsn","tsn","ap","plc","robot","robot","robot","robot","agv","agv"]

        _add_edge(edges, seen, 0, tsn_a, "wired")
        _add_edge(edges, seen, 1, tsn_b, "wired")
        _add_edge(edges, seen, tsn_a, tsn_b, "wired")
        _add_edge(edges, seen, tsn_a, ap, "wired")

        _add_edge(edges, seen, tsn_b, plc, "wired")
        for r in robots:
            _add_edge(edges, seen, tsn_b, r, "wired")
        for a in agvs:
            _add_edge(edges, seen, a, ap, "wireless")

        zone_a.append(tsn_a); zone_b.append(tsn_b); zone_ap.append(ap)
        idx += 10

    # fill remain nodes
    while len(node_types) < n:
        t = ["robot","vision","agv"][len(node_types) % 3]
        node_types.append(t)
        new_i = len(node_types)-1
        if t == "agv" and zone_ap:
            _add_edge(edges, seen, new_i, zone_ap[-1], "wireless")
        else:
            _add_edge(edges, seen, 1, new_i, "wired")

    # 우회 링크(dual uplink + inter-zone)
    for a in zone_a:
        _add_edge(edges, seen, 2, a, "wired")
    for b in zone_b:
        _add_edge(edges, seen, 2, b, "wired")
    if len(zone_a) >= 2:
        _add_edge(edges, seen, zone_a[0], zone_a[1], "wired")
    if len(zone_a) >= 3:
        _add_edge(edges, seen, zone_a[1], zone_a[2], "wired")
    if len(zone_ap) >= 2:
        _add_edge(edges, seen, zone_ap[0], zone_ap[1], "wired")

    return TopologyTemplate("T4_multizone_edgecloud", node_types, edges)