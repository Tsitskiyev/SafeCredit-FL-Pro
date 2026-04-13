from pathlib import Path
import numpy as np
import pandas as pd
import torch

from app.data import DataGenConfig, generate_bank_dataset
from app.model import CreditMLP, set_model_parameters
from app.train_eval import _safe_auc, _safe_ks  # already in your code


def load_latest_global_model(ckpt_dir: Path) -> CreditMLP:
    ckpts = sorted(ckpt_dir.glob("global_round_*.npz"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    latest = ckpts[-1]
    npz = np.load(latest)
    arrays = [npz[k] for k in npz.files]  # arr_0, arr_1, ...

    model = CreditMLP(input_dim=10)
    set_model_parameters(model, arrays)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {latest}")
    return model


@torch.no_grad()
def predict_prob(model: CreditMLP, x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x.astype(np.float32))
    logits = model(t).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_latest_global_model(Path("reports/checkpoints"))

    cfg = DataGenConfig(n_samples=8000, n_banks=5, seed=42)

    bands = [
        (18, 25, "18-25"),
        (26, 35, "26-35"),
        (36, 45, "36-45"),
        (46, 55, "46-55"),
        (56, 70, "56-70"),
    ]

    rows = []
    for bank_id in range(cfg.n_banks):
        _, _, X_val, y_val = generate_bank_dataset(bank_id=bank_id, cfg=cfg)
        p = predict_prob(model, X_val)

        age = X_val[:, 2] * 52.0 + 18.0  # back-transform from normalized age

        for lo, hi, label in bands:
            mask = (age >= lo) & (age <= hi)
            yb = y_val[mask]
            pb = p[mask]

            if len(yb) == 0:
                continue

            pred = (pb >= 0.5).astype(np.float32)
            acc = float((pred == yb).mean())
            auc = float(_safe_auc(yb, pb))
            ks = float(_safe_ks(yb, pb))
            default_rate = float(yb.mean())
            approve_rate = float((pred == 0).mean())

            rows.append(
                {
                    "bank_id": bank_id,
                    "age_band": label,
                    "n": int(len(yb)),
                    "default_rate": default_rate,
                    "approve_rate_at_0.5": approve_rate,
                    "acc": acc,
                    "auc": auc,
                    "ks": ks,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "fairness_by_age_band.csv"
    df.to_csv(csv_path, index=False)

    # Simple summary: max metric gaps between age bands (overall)
    grp = df.groupby("age_band").agg(
        n=("n", "sum"),
        acc=("acc", "mean"),
        auc=("auc", "mean"),
        ks=("ks", "mean"),
        default_rate=("default_rate", "mean"),
        approve_rate=("approve_rate_at_0.5", "mean"),
    ).reset_index()

    gap_acc = grp["acc"].max() - grp["acc"].min()
    gap_auc = grp["auc"].max() - grp["auc"].min()
    gap_approve = grp["approve_rate"].max() - grp["approve_rate"].min()

    md = f"""# Fairness Report (Age Bands)

## Aggregate gaps across age bands
- ACC gap: **{gap_acc:.4f}**
- AUC gap: **{gap_auc:.4f}**
- Approval-rate gap (threshold=0.5): **{gap_approve:.4f}**

## Notes
- This is a diagnostic fairness report for model-risk governance.
- If gaps exceed policy thresholds, apply mitigation (threshold calibration, reweighting, segment-specific monitoring).
"""

    md_path = out_dir / "fairness_summary.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
