import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


CSV_HEADER: List[str] = [
    "round",
    "phase",
    "loss",
    "acc",
    "auc",
    "ks",
    "num_clients",
    "timestamp_utc",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def init_metrics_csv(csv_path: str) -> None:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()


def append_metrics_row(csv_path: str, row: Dict[str, object]) -> None:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    base = {k: "" for k in CSV_HEADER}
    base.update(row)
    base["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerow(base)
