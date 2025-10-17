# client/net_sim.py
import time, random, math
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

# --- 기존 단일홉 시뮬 (서버로 HTTP 전송 등 가벼운 전송에 사용) ---
class NetworkSimulator:
    def __init__(self, latency_ms=50, loss_pct=1.0, bandwidth_mbps=100.0, jitter_ms=10):
        self.latency_ms = latency_ms
        self.loss_pct = loss_pct
        self.bandwidth_mbps = bandwidth_mbps
        self.jitter_ms = jitter_ms

    def transmit(self, payload_bytes: int) -> bool:
        base = self.latency_ms/1000.0
        jitter = random.uniform(0, self.jitter_ms)/1000.0
        bw_delay = (payload_bytes*8) / max(self.bandwidth_mbps*1e6, 1.0)
        time.sleep(base + bw_delay + jitter)
        if random.random()*100 < self.loss_pct:
            return False
        return True

# --- 에지 전처리 ---
class EdgePreproc:
    @staticmethod
    def apply(mode: str, B: int, *, topk: float = 0.1) -> int:
        mode = (mode or "none").lower()
        if mode == "none":
            return int(B)
        if mode == "hist-only":
            return 512  # 통계치/다이제스트만 보낸다고 가정
        if mode == "quant8":
            return int(max(256, B * 0.25))
        if mode == "topk":
            k = max(0.01, min(1.0, topk))
            return int(max(256, B * k))
        return int(B)

# --- 캐시 매니저 (노드별 LRU) ---
class CacheManager:
    def __init__(self, capacity_per_node: int = 128):
        self.capacity = capacity_per_node
        self.store: Dict[str, OrderedDict] = defaultdict(OrderedDict)

    def _touch(self, node: str, key: str):
        od = self.store[node]
        if key in od:
            od.move_to_end(key)
        else:
            od[key] = True
            if len(od) > self.capacity:
                od.popitem(last=False)

    def check(self, node: str, key: str) -> bool:
        od = self.store[node]
        if key in od:
            od.move_to_end(key)
            return True
        return False

    def put(self, node: str, key: str):
        self._touch(node, key)

# --- 다계층 DAG 네트워크 ---
class MultiTierNetwork:
    """
    DAG 위에서 경로 비용 최소화: argmin_P sum_e w(e,B)
    w(e,B) = α*lat_ms + β*loss_pct + γ*(B/bw_Mbps_sec) + δ*queue
    - B는 에지전처리/캐시 적용 후 바이트
    - 캐시 히트가 일어나면 그 이후 구간은 B_delta(작게)로 전송
    """
    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str,str,Dict]],
        *,
        src="edge",
        dst="core",
        alpha=1.0, beta=2.0, gamma=1.0, delta=0.0,
        cache: Optional[CacheManager]=None,
        edge_mode: str = "quant8",
        topk: float = 0.1,
        rng_seed: int = 0
    ):
        self.nodes = nodes
        self.edges = edges
        self.src, self.dst = src, dst
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.graph = defaultdict(list)
        for u,v,w in edges:
            self.graph[u].append((v,w))
        self.cache = cache or CacheManager()
        self.edge_mode = edge_mode
        self.topk = topk
        random.seed(rng_seed)

    @staticmethod
    def _topo(nodes, edges):
        indeg = {u:0 for u in nodes}
        for u,v,_ in edges: indeg[v]+=1
        Q = [u for u in nodes if indeg[u]==0]
        topo=[]
        while Q:
            u = Q.pop()
            topo.append(u)
            for v,_w in []:
                pass
        # 간단한 DAG 가정: 노드 수 작으므로 기본 순서를 그대로 반환
        return nodes

    def _edge_cost(self, w: Dict, B: int) -> float:
        lat = float(w.get("latency_ms", 50.0))
        loss = float(w.get("loss_pct", 1.0))
        bw = float(w.get("bandwidth_mbps", 100.0))
        queue = float(w.get("queue", 0.0))
        xfer_ms = (B*8) / max(bw*1e6,1.0) * 1000.0
        return self.alpha*lat + self.beta*loss + self.gamma*xfer_ms + self.delta*queue

    def _best_path(self, B: int) -> Tuple[List[str], float]:
        # 간단: 모든 경로를 DFS로 열거 (작은 DAG 전제), 비용 최소 경로 선택
        paths = []
        def dfs(u, acc, acc_cost):
            if u == self.dst:
                paths.append((acc[:], acc_cost))
                return
            for v, w in self.graph.get(u, []):
                c = self._edge_cost(w, B)
                acc.append(v)
                dfs(v, acc, acc_cost + c)
                acc.pop()
        dfs(self.src, [self.src], 0.0)
        if not paths:
            return [self.src, self.dst], float("inf")
        best = min(paths, key=lambda x: x[1])
        return best

    def transmit(self, cache_key: str, payload_bytes: int) -> Dict:
        # 1) 에지 전처리
        B0 = payload_bytes
        B1 = EdgePreproc.apply(self.edge_mode, B0, topk=self.topk)

        # 2) 캐시 확인: region 캐시 히트 시 이후 구간은 B_delta
        cache_nodes = [n for n in self.nodes if n.startswith("region")]
        cache_hit_at = None
        for cn in cache_nodes:
            if self.cache.check(cn, cache_key):
                cache_hit_at = cn
                break

        path, cost = self._best_path(B1)
        # RTT 근사: 경로 상 각 간선의 latency + xfer + jitter 합
        total_ms = 0.0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            w = next(w for nxt,w in self.graph[u] if nxt==v)
            lat = float(w.get("latency_ms", 50.0))
            bw  = float(w.get("bandwidth_mbps", 100.0))
            jit = float(w.get("jitter_ms", 10.0))
            # 캐시 히트 지점 이후 구간은 거의 메타만 전송
            use_bytes = 128 if (cache_hit_at is not None and (u==cache_hit_at or path.index(u) > path.index(cache_hit_at))) else B1
            xfer_ms = (use_bytes*8) / max(bw*1e6,1.0) * 1000.0
            total_ms += lat + xfer_ms + random.uniform(0, jit)
        # 캐시에 넣기(업데이트 배포됨)
        for cn in cache_nodes:
            self.cache.put(cn, cache_key)

        # 슬립(시뮬) — 실제 전송이 아니라 비용만 재현
        time.sleep(min(total_ms/1000.0, 0.2))  # 대시보드 반응성 위해 상한

        return {
            "ok": True,
            "rtt_ms": total_ms,
            "cost": cost,
            "path": path,
            "cache_hit_at": cache_hit_at,
            "bytes_after_preproc": B1,
            "edge_mode": self.edge_mode
        }

# --- 기본 DAG 팩토리 ---
def make_default_multitier(net_profile: Dict, edge_mode="quant8") -> MultiTierNetwork:
    """
    세 노드(edge→region→core), 대체 경로(edge→core) 제공.
    client별 net_profile은 edge→region 구간에 반영.
    """
    nodes = ["edge", "region-a", "core"]
    e2r = {
        "latency_ms": net_profile.get("latency_ms", 80.0),
        "loss_pct":   net_profile.get("loss_pct", 1.0),
        "bandwidth_mbps": net_profile.get("bandwidth_mbps", 50.0),
        "jitter_ms":  net_profile.get("jitter_ms", 10.0),
    }
    r2c = {"latency_ms": 60.0, "loss_pct": 0.5, "bandwidth_mbps": 150.0, "jitter_ms": 8.0}
    e2c_direct = {"latency_ms": 140.0, "loss_pct": 1.5, "bandwidth_mbps": 80.0, "jitter_ms": 15.0}
    edges = [
        ("edge", "region-a", e2r),
        ("region-a", "core", r2c),
        ("edge", "core", e2c_direct),
    ]
    return MultiTierNetwork(nodes, edges, cache=CacheManager(), edge_mode=edge_mode)
