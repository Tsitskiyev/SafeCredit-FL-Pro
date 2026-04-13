from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def make_model(input_dim: int = 7) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_partition(
    partition_id: int,
    num_partitions: int,
    n_total: int = 25000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Синтетические данные банка (неидентичные распределения между банками)."""
    rng = np.random.default_rng(seed + partition_id)
    n = max(1000, n_total // num_partitions)

    # Признаки: доход, стаж кредитной истории, возраст, DTI, просрочки, utilization, сумма займа
    income = rng.normal(9000 + 700 * partition_id, 2800, n).clip(500, 40000)
    credit_hist_years = rng.normal(6 + 0.6 * partition_id, 3.2, n).clip(0, 35)
    age = rng.normal(35 + partition_id, 10, n).clip(18, 75)
    dti = rng.beta(2.0 + 0.15 * partition_id, 5.0, n)  # debt-to-income, 0..1
    delinq_12m = rng.poisson(0.25 + 0.06 * partition_id, n).clip(0, 10)
    utilization = rng.beta(2.5, 2.0, n)  # 0..1
    loan_amount = rng.normal(18000, 8000, n).clip(1000, 90000)

    X = np.column_stack(
        [income, credit_hist_years, age, dti, delinq_12m, utilization, loan_amount]
    ).astype(np.float32)

    # Скрытая функция риска (y=1 -> default / bad)
    z = (
        -0.00008 * income
        -0.12 * credit_hist_years
        -0.015 * age
        + 2.6 * dti
        + 0.42 * delinq_12m
        + 1.25 * utilization
        + 0.000018 * loan_amount
        + 0.2 * partition_id / max(1, num_partitions - 1)
        + rng.normal(0.0, 0.35, n)
    )
    p_default = _sigmoid(z)
    y = rng.binomial(1, p_default).astype(np.float32)

    # Локальная стандартизация (данные не покидают банк)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mu) / sigma

    return X, y


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    X, y = generate_partition(partition_id, num_partitions)
    n = len(X)
    idx = np.arange(n)
    np.random.default_rng(1000 + partition_id).shuffle(idx)

    split = int(0.8 * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32).view(-1, 1)
    Xva = torch.tensor(X[va_idx], dtype=torch.float32)
    yva = torch.tensor(y[va_idx], dtype=torch.float32).view(-1, 1)

    trainloader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    valloader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    return trainloader, valloader


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def train(
    model: nn.Module,
    trainloader: DataLoader,
    local_epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(local_epochs):
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def evaluate(
    model: nn.Module,
    valloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    y_true = []
    y_score = []

    with torch.no_grad():
        for xb, yb in valloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            y_score.append(probs)
            y_true.append(yb.cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    auc = float(roc_auc_score(y_true, y_score))
    return float(np.mean(losses)), auc
