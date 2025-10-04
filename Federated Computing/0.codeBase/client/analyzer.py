# client/analyzer.py
import math
from collections import Counter
from typing import Dict, List

def entropy_from_hist(hist: Dict[int,int], base="e") -> float:
    total = sum(hist.values()) or 1
    H = 0.0
    for _, c in hist.items():
        p = c/total
        if p > 0:
            if base == "2":
                H -= p * math.log2(p)
            else:
                H -= p * math.log(p)
    return H

def label_hist(labels: List[int]) -> Dict[int,int]:
    c = Counter(labels)
    return dict(c)
