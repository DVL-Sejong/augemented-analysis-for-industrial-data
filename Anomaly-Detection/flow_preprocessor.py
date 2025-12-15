import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional


class FlowPreprocessor:
    """네트워크 플로우 데이터 전처리 클래스"""
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

    def normalize_ports(self, ports: np.ndarray) -> np.ndarray:
        """포트 번호를 로그 변환하여 정규화"""
        return np.log(ports + 1)
