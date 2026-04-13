import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common import Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from app.utils import append_metrics_row, init_metrics_csv, set_global_seed


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Weighted average by number of examples."""
    if not metrics:
        return {}

    aggregated: Dict[str, Scalar] = {}
    keys = set()
    for _, m in metrics:
        keys.update(m.keys())

    for k in keys:
        weighted_sum = 0.0
        total = 0
        for n, m in metrics:
            if k in m and isinstance(m[k], (int, float)):
                weighted_sum += n * float(m[k])
                total += n
        if total > 0:
            aggregated[k] = weighted_sum / total

    return aggregated


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, metrics_csv: str, ckpt_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_csv = metrics_csv
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        init_metrics_csv(metrics_csv)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures,
    ):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        parameters, metrics = aggregated

        append_metrics_row(
            self.metrics_csv,
            {
                "round": server_round,
                "phase": "fit",
                "loss": float(metrics.get("train_loss", "")) if "train_loss" in metrics else "",
                "acc": float(metrics.get("train_acc", "")) if "train_acc" in metrics else "",
                "auc": float(metrics.get("train_auc", "")) if "train_auc" in metrics else "",
                "ks": float(metrics.get("train_ks", "")) if "train_ks" in metrics else "",
                "num_clients": len(results),
            },
        )

        # Save global model checkpoint
        ndarrays = parameters_to_ndarrays(parameters)
        np.savez(self.ckpt_dir / f"global_round_{server_round:03d}.npz", *ndarrays)

        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures,
    ):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            return None

        loss, metrics = aggregated

        append_metrics_row(
            self.metrics_csv,
            {
                "round": server_round,
                "phase": "evaluate",
                "loss": float(loss) if loss is not None else "",
                "acc": float(metrics.get("val_acc", "")) if "val_acc" in metrics else "",
                "auc": float(metrics.get("val_auc", "")) if "val_auc" in metrics else "",
                "ks": float(metrics.get("val_ks", "")) if "val_ks" in metrics else "",
                "num_clients": len(results),
            },
        )

        return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeCredit-FL-Pro server")
    parser.add_argument("--address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--min-clients", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-csv", type=str, default="reports/tables/round_metrics.csv")
    parser.add_argument("--ckpt-dir", type=str, default="reports/checkpoints")
    args = parser.parse_args()

    set_global_seed(args.seed)

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "server_round": server_round,
            "local_epochs": args.local_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }

    def eval_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "server_round": server_round,
            "batch_size": 512,
        }

    strategy = LoggingFedAvg(
        metrics_csv=args.metrics_csv,
        ckpt_dir=args.ckpt_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        accept_failures=False,
    )

    print(f"[SERVER] Starting at {args.address}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
