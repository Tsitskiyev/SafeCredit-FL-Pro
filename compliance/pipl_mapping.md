# PIPL Mapping — SafeCredit-FL Pro

## Principle: Data minimization
- Control: raw personal data stays at bank nodes.
- Evidence: only model parameters/metrics are transmitted.

## Principle: Security safeguards
- Control: encrypted transport (TLS planned), role-based access, audit logs.
- Evidence: server/client communication architecture and logging.

## Principle: Automated decision transparency
- Control: model performance and fairness reports (AUC/KS/ACC by segments).
- Evidence: experiment tables and explainability notes.

## Principle: Impact assessment
- Control: PIPIA-style risk review before production deployment.
- Evidence: threat model + mitigation checklist.
