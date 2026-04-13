from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _safe_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def train_local(
    model: torch.nn.Module,
    loader: DataLoader,
    local_epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    steps = 0

    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1

    return running_loss / max(steps, 1)


@torch.no_grad()
def evaluate_local(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_true = []
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = yb.detach().cpu().numpy()

        batch_n = len(y_true)
        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

        all_probs.append(probs)
        all_true.append(y_true)

    y_prob = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    if len(y_true) == 0:
        return {"loss": 0.0, "acc": 0.0, "auc": 0.5, "ks": 0.0}

    return {
        "loss": float(total_loss / max(total_n, 1)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": _safe_auc(y_true, y_prob),
        "ks": _safe_ks(y_true, y_prob),
    }
