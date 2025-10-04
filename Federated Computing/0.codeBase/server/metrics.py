# server/metrics.py
import math
from typing import Dict, List

def zscores(values: List[float]):
    if not values:
        return []
    m = sum(values)/len(values)
    v = sum((x-m)**2 for x in values)/max(len(values)-1,1)
    s = math.sqrt(v) if v>0 else 1e-12
    return [(x - m)/s for x in values]

def detect_outliers_by_z(values: List[float], thresh=3.0):
    zs = zscores(values)
    return [i for i,z in enumerate(zs) if abs(z) >= thresh]
