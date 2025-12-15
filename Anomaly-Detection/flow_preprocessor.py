import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional


class FlowPreprocessor:
    """네트워크 플로우 데이터 전처리 클래스"""
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
