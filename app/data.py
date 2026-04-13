from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "income_log_norm",
    "credit_history_norm",
    "age_norm",
    "debt_to_income",
    "delinquencies_norm",
    "credit_utilization",
    "loan_to_income_norm",
    "employment_years_norm",
    "recent_inquiries_norm",
    "savings_to_income_norm",
]


@dataclass
class DataGenConfig:
    n_samples: int = 8000
    n_banks: int = 5
    test_size: float = 0.2
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_bank_dataset(
    bank_id: int,
    cfg: DataGenConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic non-IID credit-scoring data for a specific bank.
    Returns: X_train, y_train, X_val, y_val (float32)
    """
    rng = np.random.default_rng(cfg.seed + 131 * bank_id)
    n = cfg.n_samples

    # Raw synthetic features (bank-specific shifts => non-IID)
    income = rng.lognormal(mean=10.1 + 0.06 * bank_id, sigma=0.45, size=n)  # CNY/month
    credit_history_years = np.clip(rng.normal(8 + 0.8 * bank_id, 4.8, size=n), 0, 40)
    age = np.clip(rng.normal(35 + bank_id, 10, size=n), 18, 70)
    dti = np.clip(rng.beta(2.0 + 0.20 * bank_id, 5.0, size=n), 0, 1)  # debt-to-income
    delinq = np.clip(rng.poisson(0.7 + 0.25 * bank_id, size=n), 0, 12)
    util = np.clip(rng.beta(2.4, 2.8, size=n), 0, 1)  # credit utilization
    loan_amount = income * rng.uniform(3.0, 15.0, size=n)
    employment_years = np.clip(rng.normal(6 + 0.45 * bank_id, 4.0, size=n), 0, 45)
    inquiries = np.clip(rng.poisson(2.0 + 0.3 * bank_id, size=n), 0, 20)
    savings = income * rng.uniform(0.2, 8.0, size=n)

    # Normalized features
    x0 = np.log1p(income) / 12.0
    x1 = credit_history_years / 40.0
    x2 = (age - 18.0) / 52.0
    x3 = dti
    x4 = delinq / 12.0
    x5 = util
    x6 = np.clip(loan_amount / (12.0 * income + 1e-6), 0, 3) / 3.0
    x7 = employment_years / 45.0
    x8 = inquiries / 20.0
    x9 = np.clip(savings / (income + 1e-6), 0, 10) / 10.0

    X = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]).astype(np.float32)

    # Latent default-risk score
    bank_risk_shift = (bank_id - (cfg.n_banks - 1) / 2.0) * 0.35
    noise = rng.normal(0, 0.18, size=n)

    logit = (
        -2.2
        + bank_risk_shift
        - 1.30 * x0
        - 0.70 * x1
        - 0.20 * x2
        + 2.30 * x3
        + 1.00 * x4
        + 1.50 * x5
        + 0.90 * x6
        - 0.60 * x7
        + 0.50 * x8
        - 0.90 * x9
        + noise
    )

    p_default = _sigmoid(logit)
    y = rng.binomial(1, p_default).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed + bank_id,
        stratify=y,
    )

    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_val.astype(np.float32),
        y_val.astype(np.float32),
    )
