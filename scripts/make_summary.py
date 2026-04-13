from pathlib import Path
import pandas as pd

def main() -> None:
    fl_csv = Path("reports/tables/round_metrics.csv")
    cen_csv = Path("reports/tables/centralized_metrics.csv")
    out_md = Path("reports/tables/experiment_summary.md")

    if not fl_csv.exists():
        print("Missing FL metrics:", fl_csv)
        return
    if not cen_csv.exists():
        print("Missing centralized metrics:", cen_csv)
        return

    fl = pd.read_csv(fl_csv)
    fl_eval = fl[fl["phase"] == "evaluate"].copy().sort_values("round")
    cen = pd.read_csv(cen_csv).copy().sort_values("epoch")

    # best by AUC
    fl_best = fl_eval.loc[fl_eval["auc"].idxmax()]
    cen_best = cen.loc[cen["val_auc"].idxmax()]

    md = f"""# SafeCredit-FL Pro — Experiment Summary

## Federated (best validation round)
- Round: **{int(fl_best['round'])}**
- Val Loss: **{fl_best['loss']:.4f}**
- Val AUC: **{fl_best['auc']:.4f}**
- Val KS: **{fl_best['ks']:.4f}**
- Val ACC: **{fl_best['acc']:.4f}**

## Centralized (best validation epoch)
- Epoch: **{int(cen_best['epoch'])}**
- Val Loss: **{cen_best['val_loss']:.4f}**
- Val AUC: **{cen_best['val_auc']:.4f}**
- Val KS: **{cen_best['val_ks']:.4f}**
- Val ACC: **{cen_best['val_acc']:.4f}**

## Interpretation
- Centralized is an upper-bound reference (requires raw data pooling).
- Federated preserves data locality and can approach centralized quality with proper tuning.
"""

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print("Saved:", out_md)

if __name__ == "__main__":
    main()
