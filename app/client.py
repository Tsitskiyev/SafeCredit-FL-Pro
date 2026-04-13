import argparse
from typing import Dict

import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.data import DataGenConfig, generate_bank_dataset
from app.model import CreditMLP, get_model_parameters, set_model_parameters
from app.train_eval import evaluate_local, train_local
from app.utils import set_global_seed


class BankClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        num_banks: int,
        n_samples: int,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.cid = cid
        self.device = torch.device(device)
        self.model = CreditMLP(input_dim=10).to(self.device)

        cfg = DataGenConfig(n_samples=n_samples, n_banks=num_banks, seed=seed)
        X_train, y_train, X_val, y_val = generate_bank_dataset(bank_id=cid, cfg=cfg)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        self.train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    def get_parameters(self, config: Dict[str, str]):
        return get_model_parameters(self.model)

    def fit(self, parameters, config: Dict[str, str]):
        set_model_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 1e-3))
        batch_size = int(config.get("batch_size", 128))

        # rebuild loader if batch size changed by server
        train_ds = self.train_loader.dataset
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        _ = train_local(
            model=self.model,
            loader=self.train_loader,
            local_epochs=local_epochs,
            lr=lr,
            device=self.device,
        )

        train_metrics = evaluate_local(self.model, self.train_loader, self.device)

        metrics = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "train_auc": float(train_metrics["auc"]),
            "train_ks": float(train_metrics["ks"]),
        }

        return get_model_parameters(self.model), len(train_ds), metrics

    def evaluate(self, parameters, config: Dict[str, str]):
        set_model_parameters(self.model, parameters)

        batch_size = int(config.get("batch_size", 512))
        val_ds = self.val_loader.dataset
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        val_metrics = evaluate_local(self.model, self.val_loader, self.device)

        return float(val_metrics["loss"]), len(val_ds), {
            "val_acc": float(val_metrics["acc"]),
            "val_auc": float(val_metrics["auc"]),
            "val_ks": float(val_metrics["ks"]),
        }


def main() -> None:
    import argparse
    import flwr as fl

    parser = argparse.ArgumentParser(description="SafeCredit-FL-Pro client (bank node)")
    parser.add_argument("--cid", type=int, required=True, help="Bank ID (0..N-1)")
    parser.add_argument("--num-banks", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--device", type=str, default="cpu")

    # ВАЖНО: сначала parse_args, потом использование args
    args = parser.parse_args()

    # если у тебя есть set_global_seed — оставь
    try:
        from app.utils import set_global_seed
        set_global_seed(args.seed + args.cid)
    except Exception:
        pass

    client = BankClient(
        cid=args.cid,
        num_banks=args.num_banks,
        n_samples=args.samples,
        seed=args.seed,
        device=args.device,
    )

    # Современный запуск NumPyClient
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
