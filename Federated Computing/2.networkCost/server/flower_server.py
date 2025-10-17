# server/flower_server.py
import flwr as fl
from strategy import make_strategy

if __name__ == "__main__":
    strategy = make_strategy()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),  # 필요시 조정
    )
