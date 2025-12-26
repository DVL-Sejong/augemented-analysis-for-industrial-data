"""VAE Anomaly Detector Configuration"""


class VAEConfig:
    """VAE 모델 설정"""
    HIDDEN_DIM = 128
    LATENT_DIM = 20
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    BATCH_SIZE = 32
    ANOMALY_THRESHOLD_PERCENTILE = 95
