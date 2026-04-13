from __future__ import annotations

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from app.task import get_weights, make_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Агрегируем AUC по числу примеров
    weighted_auc = [num_examples * float(m["auc"]) for num_examples, m in metrics if "auc" in m]
    examples = [num_examples for num_examples, m in metrics if "auc" in m]
    if not examples:
        return {"auc": 0.0}
    return {"auc": sum(weighted_auc) / sum(examples)}


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    min_fit_clients = int(context.run_config["min-fit-clients"])
    min_eval_clients = int(context.run_config["min-evaluate-clients"])
    min_available_clients = int(context.run_config["min-available-clients"])

    ndarrays = get_weights(make_model())
    initial_parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    fit_workflow = SecAggPlusWorkflow(
        num_shares=float(context.run_config["num-shares"]),
        reconstruction_threshold=float(context.run_config["reconstruction-threshold"]),
        max_weight=float(context.run_config["max-weight"]),
        timeout=float(context.run_config["timeout"]),
    )

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, legacy_context)
