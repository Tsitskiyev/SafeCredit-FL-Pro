from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    csv_path = Path("reports/tables/round_metrics.csv")
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fit_df = df[df["phase"] == "fit"].copy()
    eval_df = df[df["phase"] == "evaluate"].copy()

    if fit_df.empty and eval_df.empty:
        print("No metrics to plot.")
        return

    # Loss curve
    plt.figure(figsize=(8, 5))
    if not fit_df.empty:
        plt.plot(fit_df["round"], fit_df["loss"], marker="o", label="train_loss")
    if not eval_df.empty:
        plt.plot(eval_df["round"], eval_df["loss"], marker="o", label="val_loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("SafeCredit-FL Pro: Loss by Round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fl_loss_curve.png", dpi=150)
    plt.close()

    # AUC curve
    plt.figure(figsize=(8, 5))
    if not fit_df.empty:
        plt.plot(fit_df["round"], fit_df["auc"], marker="o", label="train_auc")
    if not eval_df.empty:
        plt.plot(eval_df["round"], eval_df["auc"], marker="o", label="val_auc")
    plt.xlabel("Round")
    plt.ylabel("AUC")
    plt.title("SafeCredit-FL Pro: AUC by Round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fl_auc_curve.png", dpi=150)
    plt.close()

    print("Saved:")
    print(f" - {out_dir / 'fl_loss_curve.png'}")
    print(f" - {out_dir / 'fl_auc_curve.png'}")


if __name__ == "__main__":
    main()
