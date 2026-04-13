from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    fl_csv = Path("reports/tables/round_metrics.csv")
    cen_csv = Path("reports/tables/centralized_metrics.csv")
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not fl_csv.exists():
        print(f"Missing: {fl_csv} (run federated first)")
        return
    if not cen_csv.exists():
        print(f"Missing: {cen_csv} (run centralized first)")
        return

    fl_df = pd.read_csv(fl_csv)
    cen_df = pd.read_csv(cen_csv)

    fl_eval = fl_df[fl_df["phase"] == "evaluate"].copy()
    fl_eval = fl_eval.sort_values("round")

    # Plot AUC: FL rounds vs Central epochs (two x-axes in one figure is messy; keep simple)
    plt.figure(figsize=(8, 5))
    plt.plot(fl_eval["round"], fl_eval["auc"], marker="o", label="Federated (val AUC)")
    plt.plot(cen_df["epoch"], cen_df["val_auc"], marker="o", label="Centralized (val AUC)")
    plt.xlabel("Round / Epoch")
    plt.ylabel("AUC")
    plt.title("Centralized vs Federated: Validation AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_auc.png", dpi=150)
    plt.close()

    # Plot KS
    plt.figure(figsize=(8, 5))
    plt.plot(fl_eval["round"], fl_eval["ks"], marker="o", label="Federated (val KS)")
    plt.plot(cen_df["epoch"], cen_df["val_ks"], marker="o", label="Centralized (val KS)")
    plt.xlabel("Round / Epoch")
    plt.ylabel("KS")
    plt.title("Centralized vs Federated: Validation KS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_ks.png", dpi=150)
    plt.close()

    print("Saved:")
    print(f" - {fig_dir / 'compare_auc.png'}")
    print(f" - {fig_dir / 'compare_ks.png'}")


if __name__ == "__main__":
    main()
