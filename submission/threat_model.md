# Threat Model — SafeCredit-FL Pro

## Assets
- Client raw data (PII/financial)
- Model updates
- Global model checkpoints

## Threats
1. Model update interception
2. Malicious client poisoning
3. Membership inference / gradient leakage
4. Unauthorized access to server artifacts

## Mitigations
- TLS transport
- Secure aggregation (next phase)
- Client authentication and allowlist
- Gradient clipping / anomaly checks
- Audit logs and key rotation
