# SafeCredit-FL Pro — Experiment Summary

## Federated (best validation round)
- Round: **12**
- Val Loss: **0.3634**
- Val AUC: **0.6510**
- Val KS: **0.2465**
- Val ACC: **0.8750**

## Centralized (best validation epoch)
- Epoch: **10**
- Val Loss: **0.3559**
- Val AUC: **0.6733**
- Val KS: **0.2529**
- Val ACC: **0.8750**

## Interpretation
- Centralized is an upper-bound reference (requires raw data pooling).
- Federated preserves data locality and can approach centralized quality with proper tuning.
