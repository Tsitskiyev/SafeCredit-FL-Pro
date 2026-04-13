from __future__ import annotations

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from app.task import evaluate, get_weights, load_data, make_model, set_weights, train


class BankClient(NumPyClient):
    def __init__(self, partition_id: int, num_partitions: int, batch_size: int, local_epochs: int, lr: float):
        self.model = make_model()
        self.trainloader, self.valloader = load_data(partition_id, num_partitions, batch_size)
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_weights(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        set_weights(self.model, parameters)
        train(
            self.model,
            self.trainloader,
            local_epochs=self.local_epochs,
            lr=self.lr,
            device=self.device,
        )
        return get_weights(self.model), len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        set_weights(self.model, parameters)
        loss, auc = evaluate(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"auc": float(auc)}


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    lr = float(context.run_config["learning-rate"])

    return BankClient(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
    ).to_client()


# SecAgg+ мод включен
app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
