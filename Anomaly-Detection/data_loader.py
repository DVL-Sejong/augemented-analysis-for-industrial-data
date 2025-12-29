import csv
from typing import List
from analyzer import Flow, _FLOW_FIELDS


class FlowDataLoader:
    """네트워크 플로우 데이터 로더"""
    @staticmethod
