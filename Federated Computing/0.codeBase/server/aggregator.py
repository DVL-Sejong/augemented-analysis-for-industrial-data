# server/aggregator.py
import json
from collections import Counter, defaultdict
from typing import Dict, Any

class SimpleFederatedAggregator:
    """
    Placeholder for 'KIST 담당 기술' 인터페이스.
    로컬 분석 payload_json(예: {"label_hist": {"0":123, "1":77}})들을
    R 가중치로 합산해 전역 분포/요약을 만든다.
    """
    def __init__(self):
        self._round_buckets = defaultdict(list)

    def add_local(self, round_id: int, client_id: str, payload_json: str, R: float = 1.0):
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
        self._round_buckets[round_id].append((client_id, payload, R))

    def summarize_round(self, round_id: int) -> Dict[str, Any]:
        items = self._round_buckets.get(round_id, [])
        total = Counter()
        weight_sum = 0.0
        for cid, payload, R in items:
            hist = payload.get("label_hist", {})
            # 가중치 합산
            for k,v in hist.items():
                total[k] += R * float(v)
            weight_sum += R
        # 정규화 분포
        denom = sum(total.values()) or 1.0
        dist = {k: (v/denom) for k,v in total.items()}
        return {"round": round_id, "weighted_label_dist": dist, "contributors": len(items)}
