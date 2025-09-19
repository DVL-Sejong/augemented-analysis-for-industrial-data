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
        
        # 범주형 변수들을 따로 처리
        protocols = [flow.ip_protocol for flow in flows]
        states = [flow.state for flow in flows]
        
        # 라벨 인코딩
        if hasattr(self.label_encoders['ip_protocol'], 'classes_'):
            encoded_protocols = self.label_encoders['ip_protocol'].transform(protocols)
        else:
            encoded_protocols = self.label_encoders['ip_protocol'].fit_transform(protocols)
            
        if hasattr(self.label_encoders['state'], 'classes_'):
            encoded_states = self.label_encoders['state'].transform(states)
        else:
            encoded_states = self.label_encoders['state'].fit_transform(states)
        
        # 최종 특성 벡터 생성
        features = np.array(features)
        encoded_protocols = encoded_protocols.reshape(-1, 1)
        encoded_states = encoded_states.reshape(-1, 1)
        
        final_features = np.concatenate([features, encoded_protocols, encoded_states], axis=1)
        
        return final_features
    
    def load_data_from_csv(self, csv_file):
        """CSV 파일로부터 플로우 데이터 로드"""
        flows = []
        with open(csv_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split(',')
                    if len(parts) == len(_FLOW_FIELDS):
                        flow = Flow.from_csv(parts)
                        flows.append(flow)
                except Exception as e:
                    continue
        return flows
    
    def vae_loss(self, recon_x, x, mu, logvar):
        """VAE 손실 함수"""
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def fit(self, flows, epochs=100, batch_size=32):
        """VAE 모델 훈련"""
        # 데이터 전처리
        X = self.preprocess_flow_data(flows)
        X_scaled = self.scaler.fit_transform(X)
        
        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 모델 초기화
        input_dim = X_scaled.shape[1]
        self.model = VAE(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 훈련
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = self.model(data)
                loss = self.vae_loss(recon_batch, data, mu, logvar)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader.dataset)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
        # 임계값 설정 (정상 데이터의 재구성 오차 기반)
        self.model.eval()
        with torch.no_grad():
            recon_errors = []
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                recon_batch, _, _ = self.model(data)
                error = nn.functional.mse_loss(recon_batch, data, reduction='none')
                recon_errors.extend(error.mean(dim=1).cpu().numpy())

        recon_errors = np.array(recon_errors)
        self.threshold = np.percentile(recon_errors, 95)
        print(f'Threshold set to: {self.threshold:.4f}')
        return losses

    def predict(self, flows):
        """이상 탐지 수행"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")


