# Vireo Audio — Support Operations & Defect Analytics

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Engine](https://img.shields.io/badge/Analytics-Pandas%20%7C%20Statsmodels-150458.svg)](https://statsmodels.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Run Cost](https://img.shields.io/badge/Run%20Cost-₹0%20%2F%20Zero%20API%20Calls-success.svg)](#zero-cost-architecture)

A high-performance, cost-free analytics pipeline and interactive dashboard for customer support intelligence, hardware defect early-warning, and bias-corrected agent performance evaluation.

---

## 📌 Executive Summary

When customer satisfaction (CSAT) plummeted and product replacement rates surged, initial raw metrics suggested frontline customer service underperformance. This analytics suite establishes that the crisis was **not an agent capability issue**, but rather an isolated hardware manufacturing breakdown:

- **Root Cause Isolated:** Identified an acute charging fault concentrated in three consecutive production batches of **Pulse 2 earbuds (Lots 2510, 2511, 2512)**, exhibiting **41.7 to 47.0 replacements per 100 units** (vs. 2 to 9 for normal lots).
- **Financial Impact Identified:** Quantified **₹14.5 lakh** (₹1,451,124) in total excess replacement costs, of which **₹10.9 lakh** (₹1,087,175) was directly avoidable via automated inventory quarantine.
- **Fair Agent Attribution:** Frontline chat agents handling 60–65% hardware fault tickets absorbed the brunt of customer frustration. Using OLS regression and Empirical Bayes shrinkage, queue bias was eliminated—proving that apparent agent deficits were largely structural.
- **Zero Ongoing Cost:** Built with pure Python (`pandas`, `statsmodels`, `streamlit`). Zero external LLM calls, zero cloud dependencies, and ₹0 marginal run cost.

---

## 🎯 Key Findings & Business Levers

| Dimension | Raw Metric View | Adjusted Analytics Reality | Business Impact |
|---|---|---|---|
| **Pulse 2 Replacement Rate** | 16.1% overall across export | **45.3% on Lots 2510–2512** vs **7.6% on healthy lots** | Problem isolated to 3 production months, not an ongoing failure |
| **Avoidable Defect Cost** | Unmonitored | **₹10.9 lakh avoidable** (₹14.5L total excess, ₹13.0–14.5L range) | Early lot alerts fire within 27–34 days of first sale |
| **Agent Performance** | Bottom 10 agents slated for retraining | **Only 3 agents sit marginally below average** (0.20–0.29 gap) | 7 of the raw bottom 10 are explained by queue & courier delay |
| **Statistical Certainty** | Assumed persistent underperformance | **Permutation tests show 0.95 null false flags** (95th pct = 3) | 3 flagged agents are suggestive for coaching, not punitive action |
| **Resolution Timestamps** | 68% negative handle times in legacy data | **0% negative handle times** after +5.5h UTC shift fix | Data hygiene established across 11,750 helpdesk tickets |

---

## 🛠️ System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & HYGIENE                       │
│  • Timezone Normalization (+5.5h UTC shift on legacy_fd records)      │
│  • Fallback Order-Lot Join (merge_asof with 94.9% accuracy)           │
│  • SLA Breach & Financial Policy Calculations (₹350 / ₹1,820 formula) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│   DEFECT CONTROL ENGINE         │   │   AGENT PERFORMANCE NORMALIZER   │
│ • Cumulative 3-Sigma CUSUM      │   │ • OLS Multi-Factor Regression    │
│ • Lot-Month Group Tracking      │   │ • Empirical Bayes Shrinkage      │
│ • 108 Groups Tested, 0 False    │   │ • Split-Half Reliability (300x)  │
│   Alarms on 105 Healthy Lots    │   │ • Permutation Null Testing (500x)│
└────────────────┬────────────────┘   └─────────────────┬────────────────┘
                 │                                      │
                 └──────────────────┬───────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      INTERACTIVE STREAMLIT UI                          │
│  Tab 1: Raw Agent View    |  Tab 2: Bias-Adjusted Agent View           │
│  Tab 3: Defect Lot Alerts |  Tab 4: Data Quality & System Telemetry    │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Cumulative Defect Early Warning (CUSUM Control Chart)
Monitors cumulative replacement frequency per $(SKU, \text{Production Month})$ against a 3-sigma upper control limit ($n \ge 30$):
$$\text{UCL} = p_0 + 3 \sqrt{\frac{p_0(1 - p_0)}{n}}$$
- **Lot 2510:** Flagged on **19 Nov 2025** (Day 27 post-launch) $\rightarrow$ ₹4.90L avoidable.
- **Lot 2511:** Flagged on **29 Dec 2025** (Day 34 post-launch) $\rightarrow$ ₹3.72L avoidable.
- **Lot 2512:** Flagged on **25 Jan 2026** (Day 32 post-launch) $\rightarrow$ ₹2.25L avoidable.
- **False Alarm Specificity:** Tested across 108 lot groups ($\ge 30$ tickets); **0 false alarms** occurred across the 105 healthy lot groups (**0.0% false-alarm rate**).

### 2. Fair Agent Evaluation (OLS + James–Stein Shrinkage)
Eliminates structural queue sorting by estimating:
$$\text{CSAT}_i = \beta_0 + \sum \beta_c \text{Category}_i + \sum \beta_h \text{Channel}_i + \sum \beta_p \text{Product}_i + \sum \beta_m \text{Month}_i + \sum \beta_t \text{Team}_i + \epsilon_i$$
- Individual residuals are stabilized using Empirical Bayes shrinkage toward the mean:
  $$\hat{\alpha}_j^{\text{shrunk}} = \mu + \bar{e}_j \left(\frac{\tau^2}{\tau^2 + \sigma_j^2}\right)$$
- Accounts for low adjusted split-half reliability ($r = 0.25$ over 300 iterations) and demonstrates via 500-iteration permutation tests that chance alone produces up to 3 flags under the null hypothesis.

---

## 📂 Repository Structure

```
vireo-support/
├── app.py                  # Full interactive 4-tab Streamlit dashboard
├── validate.py             # Statistical validation & test suite
├── requirements.txt        # Pinned runtime dependencies
├── README.md               # Architecture documentation
├── vireo/
│   ├── __init__.py         # Package entrypoint
│   └── pipeline.py         # Data loading, cleaning, OLS, lot alerts, and metrics
└── data/                   # Place raw input CSVs here (excluded from version control)
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 (x64 recommended)
- Git

### 1. Clone & Setup Environment

```powershell
# Clone the repository
git clone https://github.com/Mohithmcu/vireo-support.git
cd vireo-support

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # On Windows PowerShell
# source .venv/bin/activate  # On macOS / Linux

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add Data Files
Place the 5 operational data CSVs in the `./data/` folder:
- `tickets.csv` (or export containing helpdesk tickets)
- `agents.csv`
- `orders.csv`
- `products.csv`
- `customers.csv`

*(Note: Data files are automatically excluded from git tracking via `.gitignore` to preserve privacy).*

### 3. Run Pipeline Check (CLI)
Verify data integrity and compute all core figures in ~1.5 seconds:
```powershell
python -m vireo.pipeline
```

### 4. Run Statistical Validation Suite
Execute the independent verification suite (evaluates stratified sample, OLS fit, permutation tests, split-half reliability, and lot false-alarm rates):
```powershell
python validate.py
```

### 5. Launch Interactive Dashboard
```powershell
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🖥️ Dashboard Interface

| Tab | Focus | Core Capability |
|---|---|---|
| **Tab 1: Raw Agent View** | Baseline Audit | Ranks agents by raw CSAT and handle time with team comparisons and clear routing warnings. |
| **Tab 2: Adjusted View** | Fair Performance | OLS-adjusted CSAT with 95% confidence intervals and James–Stein shrinkage to avoid over-penalizing high-variance agents. |
| **Tab 3: Defect Alerts** | Hardware Quality | Interactive cumulative control charts pinpointing defective production lots and tracking post-fix return to baseline. |
| **Tab 4: Data Quality** | Data Hygiene & SLAs | Quantifies timezone offset corrections, order join coverage (94.9%), and SLA credit exposure (1,064 breaches, ₹3.72L). |

---

## ⚡ Zero-Cost Architecture

- **Paid APIs:** ₹0.00
- **Cloud LLM Calls:** 0
- **Inference Latency:** Instantaneous local execution (< 2 seconds)
- **Portability:** Operates completely offline on standard commodity hardware.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
