from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

# --- Чтобы импорты app.* работали при streamlit run web/app.py ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model import CreditMLP, set_model_parameters  # noqa: E402


# =========================
# Model loading + scoring
# =========================
def _arr_key(k: str):
    # np.savez сохраняет как arr_0, arr_1...
    if k.startswith("arr_"):
        try:
            return int(k.split("_")[1])
        except Exception:
            return k
    return k


@st.cache_resource
def load_latest_checkpoint() -> Tuple[CreditMLP | None, str | None]:
    ckpt_dir = PROJECT_ROOT / "reports" / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("global_round_*.npz"))
    if not ckpts:
        return None, None

    latest = ckpts[-1]
    npz = np.load(latest, allow_pickle=False)
    arrays = [npz[k] for k in sorted(npz.files, key=_arr_key)]

    model = CreditMLP(input_dim=10)
    set_model_parameters(model, arrays)
    model.eval()
    return model, latest.name


def preprocess_single(
    income: float,
    credit_history_years: float,
    age: float,
    dti: float,
    delinq: float,
    utilization: float,
    loan_amount: float,
    employment_years: float,
    inquiries: float,
    savings: float,
) -> np.ndarray:
    # Те же преобразования, что в обучении
    x0 = np.log1p(income) / 12.0
    x1 = credit_history_years / 40.0
    x2 = (age - 18.0) / 52.0
    x3 = np.clip(dti, 0, 1)
    x4 = np.clip(delinq, 0, 12) / 12.0
    x5 = np.clip(utilization, 0, 1)
    x6 = np.clip(loan_amount / (12.0 * income + 1e-6), 0, 3) / 3.0
    x7 = np.clip(employment_years, 0, 45) / 45.0
    x8 = np.clip(inquiries, 0, 20) / 20.0
    x9 = np.clip(savings / (income + 1e-6), 0, 10) / 10.0
    return np.array([[x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]], dtype=np.float32)


def preprocess_df(df: pd.DataFrame) -> np.ndarray:
    x0 = np.log1p(df["income"].values) / 12.0
    x1 = df["credit_history_years"].values / 40.0
    x2 = (df["age"].values - 18.0) / 52.0
    x3 = np.clip(df["dti"].values, 0, 1)
    x4 = np.clip(df["delinq"].values, 0, 12) / 12.0
    x5 = np.clip(df["utilization"].values, 0, 1)
    x6 = np.clip(df["loan_amount"].values / (12.0 * df["income"].values + 1e-6), 0, 3) / 3.0
    x7 = np.clip(df["employment_years"].values, 0, 45) / 45.0
    x8 = np.clip(df["inquiries"].values, 0, 20) / 20.0
    x9 = np.clip(df["savings"].values / (df["income"].values + 1e-6), 0, 10) / 10.0
    X = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]).astype(np.float32)
    return X


@torch.no_grad()
def predict_pd(model: CreditMLP, X: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(X)
    logits = model(t)
    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return probs


# =========================
# Business logic
# =========================
def decision_rule(pd_hat: float, t_approve: float, t_review: float) -> str:
    if pd_hat < t_approve:
        return "APPROVE"
    if pd_hat < t_review:
        return "MANUAL_REVIEW"
    return "REJECT"


def recommend_apr(base_apr: float, pd_hat: float, dti: float, delinq: float) -> float:
    apr = base_apr + 25.0 * pd_hat + 6.0 * dti + 0.6 * delinq
    return float(np.clip(apr, 6.0, 48.0))


def recommend_limit(
    requested_loan: float,
    income: float,
    dti: float,
    pd_hat: float,
    savings: float,
) -> float:
    # Консервативная эвристика для демо
    capacity = max(1000.0, income * 8.0 * (1.0 - 0.5 * dti))
    liquidity_boost = 1.0 + min(savings / (income + 1e-6), 3.0) * 0.05
    risk_factor = float(np.clip(1.0 - 1.4 * pd_hat, 0.2, 1.0))
    lim = min(requested_loan, capacity * liquidity_boost) * risk_factor
    return float(max(1000.0, lim))


def expected_profit_one(
    pd_hat: float,
    amount: float,
    apr: float,
    term_months: int,
    lgd: float,
    funding_cost: float,
    opex_rate: float,
) -> Tuple[float, float]:
    # EL = PD * LGD * Amount
    el = float(pd_hat * lgd * amount)

    # Нетто-маржа за срок кредита
    net_rate_annual = (apr - funding_cost - opex_rate) / 100.0
    net_margin_term = net_rate_annual * (term_months / 12.0)

    # Ожидаемая прибыль
    # (1-PD)*маржа*Amount - PD*LGD*Amount
    profit = float((1.0 - pd_hat) * net_margin_term * amount - pd_hat * lgd * amount)
    return el, profit


def risk_flags(
    age: float, dti: float, delinq: float, utilization: float, inquiries: float, savings: float, income: float
):
    flags = []
    if dti > 0.55:
        flags.append("Высокий DTI")
    if delinq >= 2:
        flags.append("Просрочки за 12м")
    if utilization > 0.8:
        flags.append("Высокая загрузка кредитных линий")
    if inquiries >= 6:
        flags.append("Много недавних запросов")
    if savings < 0.5 * income:
        flags.append("Низкая финансовая подушка")
    if age < 21:
        flags.append("Очень короткий жизненный/кредитный профиль")
    return flags


# =========================
# Portfolio simulator data
# =========================
def generate_portfolio(n: int, bank_id: int, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17 * bank_id)

    income = rng.lognormal(mean=10.0 + 0.06 * bank_id, sigma=0.45, size=n)
    credit_history_years = np.clip(rng.normal(7 + 0.8 * bank_id, 4.8, size=n), 0, 40)
    age = np.clip(rng.normal(34 + bank_id, 10, size=n), 18, 70)
    dti = np.clip(rng.beta(2.0 + 0.2 * bank_id, 5.0, size=n), 0, 1)
    delinq = np.clip(rng.poisson(0.7 + 0.2 * bank_id, size=n), 0, 12)
    utilization = np.clip(rng.beta(2.3, 2.9, size=n), 0, 1)
    loan_amount = income * rng.uniform(3.0, 15.0, size=n)
    employment_years = np.clip(rng.normal(6 + 0.4 * bank_id, 4.0, size=n), 0, 45)
    inquiries = np.clip(rng.poisson(2.0 + 0.3 * bank_id, size=n), 0, 20)
    savings = income * rng.uniform(0.2, 8.0, size=n)

    df = pd.DataFrame(
        {
            "income": income,
            "credit_history_years": credit_history_years,
            "age": age,
            "dti": dti,
            "delinq": delinq,
            "utilization": utilization,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "inquiries": inquiries,
            "savings": savings,
        }
    )

    # Synthetic "ground truth" default (для KPI bad rate в симуляторе)
    x0 = np.log1p(df["income"].values) / 12.0
    x1 = df["credit_history_years"].values / 40.0
    x2 = (df["age"].values - 18.0) / 52.0
    x3 = df["dti"].values
    x4 = df["delinq"].values / 12.0
    x5 = df["utilization"].values
    x6 = np.clip(df["loan_amount"].values / (12.0 * df["income"].values + 1e-6), 0, 3) / 3.0
    x7 = df["employment_years"].values / 45.0
    x8 = df["inquiries"].values / 20.0
    x9 = np.clip(df["savings"].values / (df["income"].values + 1e-6), 0, 10) / 10.0

    shift = (bank_id - 2) * 0.30
    noise = rng.normal(0, 0.18, size=n)
    logit = (
        -2.2
        + shift
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
    p_true = 1 / (1 + np.exp(-logit))
    y_true = rng.binomial(1, p_true)

    df["y_true_default"] = y_true
    return df


# =========================
# UI
# =========================
st.set_page_config(page_title="SafeCredit-FL Pro", layout="wide")
st.title("SafeCredit-FL Pro")
st.caption("Privacy-First Federated Credit Scoring with Decision & Profit Engine")

model, ckpt_name = load_latest_checkpoint()
if model is None:
    st.error("Не найден глобальный FL checkpoint. Сначала запусти федеративное обучение.")
    st.stop()
else:
    st.success(f"Модель загружена: {ckpt_name}")

with st.sidebar:
    st.subheader("Policy (глобально)")
    t_approve = st.slider("PD threshold: Auto-Approve", 0.01, 0.60, 0.15, 0.01)
    t_review = st.slider("PD threshold: Manual-Review upper bound", 0.05, 0.80, 0.30, 0.01)

    if t_review <= t_approve:
        st.warning("t_review должен быть больше t_approve. Автокоррекция +0.05")
        t_review = min(0.95, t_approve + 0.05)

    review_pass_rate = st.slider("Manual Review approval rate", 0.0, 1.0, 0.35, 0.05)

    st.markdown("---")
    st.subheader("Economics")
    base_apr = st.slider("Base APR (%)", 6.0, 30.0, 12.0, 0.5)
    funding_cost = st.slider("Funding cost (% annual)", 0.0, 20.0, 6.0, 0.5)
    opex_rate = st.slider("OPEX (% annual)", 0.0, 10.0, 2.0, 0.1)
    lgd = st.slider("LGD (0..1)", 0.1, 1.0, 0.45, 0.05)
    term_months = st.slider("Loan term (months)", 6, 60, 24, 6)

tab1, tab2, tab3 = st.tabs(
    ["1) Underwriter Decision", "2) Portfolio Profit Simulator", "3) FL Monitoring"]
)

# -------------------------
# TAB 1: Underwriter
# -------------------------
with tab1:
    st.subheader("Анкета заемщика")

    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Monthly income (CNY)", 500.0, 300000.0, 10000.0, 100.0)
        credit_history_years = st.number_input("Credit history (years)", 0.0, 40.0, 5.0, 0.5)
        age = st.number_input("Age", 18.0, 75.0, 30.0, 1.0)
        dti = st.number_input("Debt-to-income (0..1)", 0.0, 1.0, 0.35, 0.01)
        delinq = st.number_input("Delinquencies (12m)", 0.0, 12.0, 0.0, 1.0)

    with c2:
        utilization = st.number_input("Credit utilization (0..1)", 0.0, 1.0, 0.40, 0.01)
        loan_amount = st.number_input("Requested loan amount (CNY)", 1000.0, 3000000.0, 80000.0, 500.0)
        employment_years = st.number_input("Employment years", 0.0, 45.0, 6.0, 0.5)
        inquiries = st.number_input("Recent inquiries", 0.0, 20.0, 2.0, 1.0)
        savings = st.number_input("Savings (CNY)", 0.0, 3000000.0, 20000.0, 500.0)

    if st.button("Score Applicant", type="primary"):
        X = preprocess_single(
            income=income,
            credit_history_years=credit_history_years,
            age=age,
            dti=dti,
            delinq=delinq,
            utilization=utilization,
            loan_amount=loan_amount,
            employment_years=employment_years,
            inquiries=inquiries,
            savings=savings,
        )
        pd_hat = float(predict_pd(model, X)[0])

        dec = decision_rule(pd_hat, t_approve, t_review)
        apr = recommend_apr(base_apr, pd_hat, dti, delinq)
        rec_limit = recommend_limit(loan_amount, income, dti, pd_hat, savings)
        el, exp_profit = expected_profit_one(
            pd_hat=pd_hat,
            amount=rec_limit,
            apr=apr,
            term_months=term_months,
            lgd=lgd,
            funding_cost=funding_cost,
            opex_rate=opex_rate,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PD (default probability)", f"{pd_hat:.2%}")
        m2.metric("Decision", dec)
        m3.metric("Recommended limit (CNY)", f"{rec_limit:,.0f}")
        m4.metric("Risk-based APR", f"{apr:.2f}%")

        m5, m6 = st.columns(2)
        m5.metric("Expected Loss (CNY)", f"{el:,.0f}")
        m6.metric("Expected Profit over term (CNY)", f"{exp_profit:,.0f}")

        flags = risk_flags(age, dti, delinq, utilization, inquiries, savings, income)
        if flags:
            st.warning("Ключевые risk flags: " + ", ".join(flags))
        else:
            st.success("Явных risk flags по базовым правилам не выявлено.")

        st.info(
            "Это decision-support инструмент: финальное решение принимает кредитная политика банка и андеррайтер."
        )

# -------------------------
# TAB 2: Portfolio Simulator
# -------------------------
with tab2:
    st.subheader("Портфельная симуляция: прибыль/риск при выбранной политике")

    s1, s2, s3 = st.columns(3)
    with s1:
        bank_id = st.selectbox("Simulated bank profile", [0, 1, 2, 3, 4], index=2)
    with s2:
        n_apps = st.slider("Portfolio size (applications)", 1000, 50000, 10000, 1000)
    with s3:
        sim_seed = st.number_input("Simulation seed", 1, 99999, 42, 1)

    run_sim = st.button("Run Portfolio Simulation")

    if run_sim:
        df = generate_portfolio(n=n_apps, bank_id=bank_id, seed=int(sim_seed))
        Xp = preprocess_df(df)
        pd_hat = predict_pd(model, Xp)
        df["pd_hat"] = pd_hat

        # Decision with review stochastic pass
        rng = np.random.default_rng(int(sim_seed) + 999)
        dec = np.full(len(df), "REJECT", dtype=object)
        mask_a = df["pd_hat"].values < t_approve
        mask_r = (df["pd_hat"].values >= t_approve) & (df["pd_hat"].values < t_review)

        dec[mask_a] = "APPROVE"
        review_pass = rng.random(mask_r.sum()) < review_pass_rate
        idx_r = np.where(mask_r)[0]
        dec[idx_r[review_pass]] = "APPROVE"
        dec[idx_r[~review_pass]] = "MANUAL_REVIEW"

        df["decision"] = dec
        approved = df["decision"] == "APPROVE"

        # Pricing + limits + economics
        apr = np.clip(base_apr + 25 * df["pd_hat"].values + 6 * df["dti"].values + 0.6 * df["delinq"].values, 6, 48)
        capacity = np.maximum(1000.0, df["income"].values * 8.0 * (1.0 - 0.5 * df["dti"].values))
        liq_boost = 1.0 + np.minimum(df["savings"].values / (df["income"].values + 1e-6), 3.0) * 0.05
        risk_factor = np.clip(1.0 - 1.4 * df["pd_hat"].values, 0.2, 1.0)
        rec_limit = np.maximum(1000.0, np.minimum(df["loan_amount"].values, capacity * liq_boost) * risk_factor)

        net_rate_annual = (apr - funding_cost - opex_rate) / 100.0
        net_margin_term = net_rate_annual * (term_months / 12.0)

        el = df["pd_hat"].values * lgd * rec_limit
        profit = (1.0 - df["pd_hat"].values) * net_margin_term * rec_limit - df["pd_hat"].values * lgd * rec_limit

        df["apr"] = apr
        df["rec_limit"] = rec_limit
        df["el"] = el
        df["exp_profit"] = profit

        # KPI
        approval_rate = float(approved.mean())
        approved_count = int(approved.sum())

        if approved_count > 0:
            bad_rate_true = float(df.loc[approved, "y_true_default"].mean())
            avg_pd_approved = float(df.loc[approved, "pd_hat"].mean())
            total_el = float(df.loc[approved, "el"].sum())
            total_profit = float(df.loc[approved, "exp_profit"].sum())
            avg_profit_per_approved = float(df.loc[approved, "exp_profit"].mean())
        else:
            bad_rate_true = 0.0
            avg_pd_approved = 0.0
            total_el = 0.0
            total_profit = 0.0
            avg_profit_per_approved = 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Approval rate", f"{approval_rate:.2%}")
        k2.metric("Approved count", f"{approved_count:,}")
        k3.metric("Observed bad rate (approved)", f"{bad_rate_true:.2%}")
        k4.metric("Avg predicted PD (approved)", f"{avg_pd_approved:.2%}")

        k5, k6 = st.columns(2)
        k5.metric("Total expected loss (approved, CNY)", f"{total_el:,.0f}")
        k6.metric("Total expected profit (approved, CNY)", f"{total_profit:,.0f}")

        st.metric("Expected profit per approved loan (CNY)", f"{avg_profit_per_approved:,.0f}")

        # Threshold sweep chart: auto-approve threshold vs profit/risk
        t_grid = np.linspace(0.05, min(0.6, t_review - 0.01), 20)
        rows = []
        pd_all = df["pd_hat"].values
        y_true = df["y_true_default"].values
        for t in t_grid:
            appr = pd_all < t
            appr_rate = appr.mean()

            if appr.sum() > 0:
                bad = y_true[appr].mean()
                # reuse economics for approved only
                p = profit[appr].sum()
            else:
                bad = 0.0
                p = 0.0

            rows.append(
                {
                    "t_auto_approve": t,
                    "approval_rate": appr_rate,
                    "bad_rate_approved": bad,
                    "expected_profit": p,
                }
            )

        sweep = pd.DataFrame(rows)
        st.markdown("### Policy frontier (threshold sweep)")
        st.line_chart(sweep.set_index("t_auto_approve")[["approval_rate", "bad_rate_approved"]])
        st.line_chart(sweep.set_index("t_auto_approve")[["expected_profit"]])

        with st.expander("Показать 20 примеров решений"):
            show_cols = [
                "income", "age", "dti", "delinq", "utilization", "loan_amount",
                "pd_hat", "decision", "apr", "rec_limit", "el", "exp_profit", "y_true_default"
            ]
            st.dataframe(df[show_cols].head(20), use_container_width=True)

# -------------------------
# TAB 3: Monitoring
# -------------------------
with tab3:
    st.subheader("Federated training monitoring")

    metrics_csv = PROJECT_ROOT / "reports" / "tables" / "round_metrics.csv"
    cen_csv = PROJECT_ROOT / "reports" / "tables" / "centralized_metrics.csv"

    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)
        fit_df = mdf[mdf["phase"] == "fit"].copy().sort_values("round")
        eval_df = mdf[mdf["phase"] == "evaluate"].copy().sort_values("round")

        c1, c2, c3 = st.columns(3)
        if not eval_df.empty:
            c1.metric("Latest FL Val AUC", f"{eval_df['auc'].iloc[-1]:.4f}")
            c2.metric("Latest FL Val KS", f"{eval_df['ks'].iloc[-1]:.4f}")
            c3.metric("Latest FL Val Loss", f"{eval_df['loss'].iloc[-1]:.4f}")

        st.markdown("#### FL curves")
        if not fit_df.empty:
            st.line_chart(fit_df.set_index("round")[["loss", "auc", "ks"]])
        if not eval_df.empty:
            st.line_chart(eval_df.set_index("round")[["loss", "auc", "ks"]])
    else:
        st.warning("Не найден reports/tables/round_metrics.csv — сначала прогони FL-обучение.")

    if cen_csv.exists():
        cdf = pd.read_csv(cen_csv)
        st.markdown("#### Centralized baseline")
        st.line_chart(cdf.set_index("epoch")[["val_loss", "val_auc", "val_ks"]])
        st.metric("Best centralized val AUC", f"{cdf['val_auc'].max():.4f}")
    else:
        st.info("centralized_metrics.csv пока нет (запусти scripts.run_centralized)")
