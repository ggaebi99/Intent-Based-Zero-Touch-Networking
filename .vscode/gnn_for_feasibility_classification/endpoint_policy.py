# endpoint_policy.py
import random
from typing import List, Tuple
import torch

# build_layer2_dataset.py의 tclass_id 매핑을 그대로 가정:
# 0 control_loop, 1 safety_io, 2 pubsub_telemetry, 3 video_stream, 4 best_effort_misc

def _nodes_of_type(ntype: torch.Tensor, type_id: int) -> List[int]:
    return [i for i, t in enumerate(ntype.view(-1).tolist()) if int(t) == int(type_id)]

def sample_pairs_from_tclass(
    ntype: torch.Tensor,
    tcid: int,
    k: int,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """
    Intent에는 src/dst가 없으므로, traffic_class(tcid) 기반으로
    endpoint 후보 집합을 만들고 pair를 샘플링한다.
    """
    rng = random.Random(seed)

    # NODE_TYPES = ["robot","agv","plc","vision","ap","tsn"] 라고 했으니 type_id는:
    ROBOT, AGV, PLC, VISION, AP, TSN = 0, 1, 2, 3, 4, 5

    robots = _nodes_of_type(ntype, ROBOT)
    agvs   = _nodes_of_type(ntype, AGV)
    plcs   = _nodes_of_type(ntype, PLC)
    visions= _nodes_of_type(ntype, VISION)
    aps    = _nodes_of_type(ntype, AP)
    tsns   = _nodes_of_type(ntype, TSN)

    # core TSN(0,1)이 있으면 선호
    core_tsns = [i for i in tsns if i in (0, 1)]
    if not core_tsns and tsns:
        core_tsns = [tsns[0]]

    # traffic_class별 후보 정의 (ID 없이 "역할 기반")
    if tcid in (0, 1):  # control_loop, safety_io
        src_pool = (robots + agvs) or robots or agvs
        dst_pool = plcs or core_tsns or tsns
    elif tcid == 2:  # telemetry
        src_pool = (robots + agvs + plcs) or robots or agvs or plcs
        dst_pool = core_tsns or tsns or aps
    elif tcid == 3:  # video_stream
        src_pool = visions or robots
        dst_pool = core_tsns or tsns or aps
    else:  # best_effort_misc
        src_pool = aps or robots or tsns
        dst_pool = core_tsns or tsns or aps

    if not src_pool or not dst_pool:
        # 그래프에 타입이 부족하면 fallback: 아무 노드나
        n = int(ntype.numel())
        src_pool = list(range(n))
        dst_pool = list(range(n))

    pairs = []
    seen = set()
    max_try = k * 50
    for _ in range(max_try):
        s = rng.choice(src_pool)
        t = rng.choice(dst_pool)
        if s == t:
            continue
        key = (s, t)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((s, t))
        if len(pairs) >= k:
            break

    # k개가 안 채워졌으면 중복 허용
    while len(pairs) < k:
        s = rng.choice(src_pool)
        t = rng.choice(dst_pool)
        if s != t:
            pairs.append((s, t))

    return pairs