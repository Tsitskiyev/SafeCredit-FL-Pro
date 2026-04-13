from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.data import DataGenConfig, generate_bank_dataset
from app.model import CreditMLP
from app.train_eval import evaluate_local, train_local
from app.utils import ensure_dir, set_global_seed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    set_global_seed(42)

    out_dir = Path("reports/tables")
    ensure_dir(str(out_dir))

    cfg = DataGenConfig(n_samples=8000, n_banks=5, seed=42)

    # Simulate "centralized" training by pooling all banks' data
    Xtr_list, ytr_list, Xv_list, yv_list = [], [], [], []
    for bank_id in range(cfg.n_banks):
        X_train, y_train, X_val, y_val = generate_bank_dataset(bank_id, cfg)
        Xtr_list.append(X_train)
        ytr_list.append(y_train)
        Xv_list.append(X_val)
        yv_list.append(y_val)

    X_train = np.vstack(Xtr_list).astype(np.float32)
    y_train = np.concatenate(ytr_list).astype(np.float32)
    X_val = np.vstack(Xv_list).astype(np.float32)
    y_val = np.concatenate(yv_list).astype(np.float32)

    device = torch.device("cpu")
    model = CreditMLP(input_dim=X_train.shape[1]).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=512,
        shuffle=False,
    )

    rows = []
    epochs = 10
    lr = 1e-3

    for epoch in range(1, epochs + 1):
        loss = train_local(model, train_loader, local_epochs=1, lr=lr, device=device)
        metrics_train = evaluate_local(model, train_loader, device)
        metrics_val = evaluate_local(model, val_loader, device)

        rows.append(
            {
                "epoch": epoch,
                "train_loss": metrics_train["loss"],
                "train_acc": metrics_train["acc"],
                "train_auc": metrics_train["auc"],
                "train_ks": metrics_train["ks"],
                "val_loss": metrics_val["loss"],
                "val_acc": metrics_val["acc"],
                "val_auc": metrics_val["auc"],
                "val_ks": metrics_val["ks"],
            }
        )

        print(
            f"[CENTRAL] epoch={epoch:02d} "
            f"val_auc={metrics_val['auc']:.4f} val_ks={metrics_val['ks']:.4f} val_loss={metrics_val['loss']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "centralized_metrics.csv", index=False)
    print(f"Saved: {out_dir / 'centralized_metrics.csv'}")


if __name__ == "__main__":
    main()
