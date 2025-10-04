
# server/flower_server.py
import os
import flwr as fl
from .strategy import make_strategy

if __name__ == "__main__":
    strategy = make_strategy()
    num_rounds = int(os.getenv("ROUNDS", "5"))
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )
