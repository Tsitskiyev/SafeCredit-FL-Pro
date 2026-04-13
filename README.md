# SafeCredit-FL Pro
### Privacy-First Federated Credit Scoring for Banking Consortia (PIPL-Aligned)

> **SafeCredit-FL Pro** is an end-to-end prototype of a privacy-preserving federated credit scoring system.  
> Multiple banks collaboratively train a global model **without sharing raw customer data**.  
> The project includes:
> - Federated training (FedAvg),
> - Banking decision engine (Approve / Review / Reject),
> - Portfolio profit simulator,
> - FL monitoring and compliance artifacts.


🚀 Why this project matters

Traditional consortium scoring requires data centralization (high legal and security risk).  
**SafeCredit-FL Pro** keeps personal financial records inside each bank and exchanges only model updates.

**Business value for banks:**
1. Better risk models than single-bank training (cross-institution learning).
2. Lower privacy/legal risk via data locality.
3. Practical decision support: PD, limits, pricing, expected loss/profit.


## ✨ Key Features

- **Federated Learning (FL)** across multiple simulated bank nodes.
- **FedAvg aggregation** on a central coordinator.
- **Synthetic banking dataset generator** with heterogeneous bank profiles.
- **Underwriter UI**:
  - Probability of Default (PD),
  - Decision class (APPROVE / MANUAL_REVIEW / REJECT),
  - Recommended credit limit,
  - Risk-based APR,
  - Expected loss & expected profit.
- **Portfolio simulator**:
  - Approval rate,
  - Observed bad rate,
  - Profit-vs-threshold frontier.
- **Monitoring & analytics**:
  - FL curves (loss/AUC/KS),
  - Centralized baseline comparison,
  - Fairness-by-age diagnostics.
- **Compliance docs**:
  - PIPL mapping,
  - Threat model.

---

## 🧠 System Architecture

```text
+--------------------+       model updates        +----------------------+
|  Bank Node #0      | -------------------------> |                      |
|  Local training    |                            |                      |
|  Local customer DB | <------------------------- |  Central Aggregator  |
+--------------------+      global model          |  (FedAvg server)     |
                                                   |                      |
+--------------------+       model updates        |                      |
|  Bank Node #1      | -------------------------> |                      |
|  Local training    |                            +----------------------+
|  Local customer DB | <-------------------------     ^
+--------------------+      global model             |
                                                     |
...                                                  |
                                                     |
+--------------------+       model updates           |
|  Bank Node #N      | ------------------------------+
|  Local training    |
|  Local customer DB |
+--------------------+
Privacy principle: raw customer data never leaves local bank nodes.

🔢 FedAvg (Mathematical Formulation)
At round t, At round k trains locally and returns parameters w k (t+1)​
Server computes weighted average by local sample size nk: 

formula.png(in the folder)

where: 
K - number of participating banks,
nk - number of local training samples at bank K
w(t+1)- new global model parameters.


🛠 Tech Stack
1)Python 3.11
2)Flower (flwr) for federated orchestration
3)PyTorch for model training/inference
4)NumPy / Pandas / scikit-learn for data + metrics
5)Streamlit for decision dashboard
6)Matplotlib for reporting figures



📁 Project Structure
SafeCredit-FL-Pro/
├─ app/
│  ├─ __init__.py
│  ├─ client.py
│  ├─ server.py
│  ├─ data.py
│  ├─ model.py
│  ├─ train_eval.py
│  └─ utils.py
├─ scripts/
│  ├─ __init__.py
│  ├─ plot_metrics.py
│  ├─ run_centralized.py
│  ├─ compare_centralized_vs_fl.py
│  ├─ make_summary.py
│  └─ fairness_report.py
├─ web/
│  └─ dashboard.py
├─ reports/
│  ├─ checkpoints/
│  ├─ figures/
│  └─ tables/
├─ compliance/
│  ├─ pipl_mapping.md
│  └─ threat_model.md
├─ requirements.txt
└─ README.md


⚙️ Installation
# From project root
        python -m venv .venv
# Windows PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUse
        .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


▶️ Run Federated Training (Local, Multi-Terminal)
Terminal 1: Server

python -m app.server --address 127.0.0.1:8080 --rounds 12 --min-clients 5

Terminals 2–6: Clients (one per terminal)

python -m app.client --cid 0 --num-banks 5 --server-address 127.0.0.1:8080
python -m app.client --cid 1 --num-banks 5 --server-address 127.0.0.1:8080
python -m app.client --cid 2 --num-banks 5 --server-address 127.0.0.1:8080
python -m app.client --cid 3 --num-banks 5 --server-address 127.0.0.1:8080
python -m app.client --cid 4 --num-banks 5 --server-address 127.0.0.1:8080

Training will stop automatically after the configured number of rounds.

📊 Generate Reports
python -m scripts.plot_metrics
python -m scripts.run_centralized
python -m scripts.compare_centralized_vs_fl
python -m scripts.make_summary
python -m scripts.fairness_report
Expected outputs:

reports/figures/fl_loss_curve.png
reports/figures/fl_auc_curve.png
reports/figures/compare_auc.png
reports/figures/compare_ks.png
reports/tables/experiment_summary.md
reports/tables/fairness_by_age_band.csv
reports/tables/fairness_summary.md


🌐 Run Banking Dashboard

streamlit run .\web\appaskhab.py

Dashboard tabs:

3)Underwriter Decision — PD + decision + limit + APR + expected P/L
2)Portfolio Profit Simulator — policy thresholds vs approval/risk/profit
1)FL Monitoring — federated metrics + centralized baseline


🧾 PIPL Alignment (Design-Level)
SafeCredit-FL Pro is built around privacy-first principles aligned with PIPL-oriented controls:

1) Data minimization: no raw personal data transfer across institutions.
2) Purpose limitation: model collaboration for credit risk estimation only.
3) Security safeguards: isolated local training, controlled aggregation, audit-ready artifacts.
4) Governance readiness: fairness/risk reporting and threat-model documentation.

See:

compliance/pipl_mapping.md
compliance/threat_model.md


🛡 Security & Risk Notes
Current prototype includes:

1) Local data confinement per bank,
2) Aggregated model update flow,
3) Operational reporting for governance.

Recommended production upgrades:

1) Mutual TLS between nodes and aggregator,
2) Secure aggregation (e.g., SecAgg+),
3) Robust aggregation / poisoning defenses,
4) Model registry + signed artifacts + continuous drift monitoring.


📈 Evaluation Philosophy
This project is evaluated in both ML and banking terms:

1. Model quality: AUC, KS, loss
2. Decision quality: approval rate, bad rate, expected loss/profit
3. Governance: fairness diagnostics, compliance traceability
4. Privacy posture: no raw data centralization

🔬 Reproducibility

Deterministic seeds are used in synthetic data generation.
Commands above reproduce all figures and summary tables.
All outputs are stored under reports/.



🗺 Roadmap

1. TLS-enabled FL communication

2. SecAgg+ integration

3. Adversarial client simulation + robust defenses

4. Calibration + reject inference

5. Explainability (feature attribution at decision time)

6. Containerized deployment (Docker Compose/K8s)


👤 Author
Tsitskiyev Askhab Magamet-Salievich
Applicant portfolio project: Privacy-Preserving AI for Fintech

📄 License
For educational and research demonstration purposes.
