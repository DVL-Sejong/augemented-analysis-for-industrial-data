import csv
from typing import List
from analyzer import Flow, _FLOW_FIELDS


class FlowDataLoader:
    """네트워크 플로우 데이터 로더"""
    @staticmethod
    def load_from_csv(csv_file: str) -> List[Flow]:
        """CSV 파일에서 플로우 데이터 로드"""
        flows = []
