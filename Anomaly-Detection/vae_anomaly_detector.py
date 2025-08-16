import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import Flow, _FLOW_FIELDS
import ipaddr
import datetime


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEAnomalyDetector:
    def __init__(self, hidden_dim=128, latent_dim=20, learning_rate=1e-3):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.threshold = None
        
    def preprocess_flow_data(self, flows):
        """네트워크 플로우 데이터를 VAE 입력용으로 전처리"""
        features = []
        
        for flow in flows:
            feature = []
            
            # 시간 특성 추출
            hour = flow.ts.hour
            day_of_week = flow.ts.weekday()
            feature.extend([hour, day_of_week])
            
            # IP 프로토콜 인코딩
            if 'ip_protocol' not in self.label_encoders:
                self.label_encoders['ip_protocol'] = LabelEncoder()
            
            # 상태 인코딩
            if 'state' not in self.label_encoders:
                self.label_encoders['state'] = LabelEncoder()
            
            # 포트 번호 (정규화를 위해 로그 변환)
            src_port_log = np.log(flow.src_port + 1)
            dst_port_log = np.log(flow.dst_port + 1)
            
            # 전송 바이트 (로그 변환)
            src_tx_log = np.log(flow.src_tx + 1)
            dst_tx_log = np.log(flow.dst_tx + 1)
            
            # IP 주소 특성 (private/public)
            src_is_private = 1 if flow.src_ip.is_private else 0
            dst_is_private = 1 if flow.dst_ip.is_private else 0
            
            feature.extend([
                src_port_log, dst_port_log,
                src_tx_log, dst_tx_log,
                src_is_private, dst_is_private
            ])
            
            features.append(feature)
        
