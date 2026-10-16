# 🚂 UK Train Delay Analytics

> **Production-style data pipeline** — Official ORR data · AWS S3 · AWS Lambda · Python · Machine Learning · Streamlit Dashboard

**🔗 Live demo:** https://uk-train-delay-analytics-abu.streamlit.app/

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![AWS S3](https://img.shields.io/badge/AWS-S3-orange?logo=amazons3)](https://aws.amazon.com/s3/)
[![AWS Lambda](https://img.shields.io/badge/AWS-Lambda-orange?logo=awslambda)](https://aws.amazon.com/lambda/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io)
[![Data Source](https://img.shields.io/badge/Data-ORR%20Official-green)](https://dataportal.orr.gov.uk)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Project Overview

End-to-end data engineering pipeline that ingests **official UK rail performance data** from the Office of Rail and Road (ORR), processes it automatically via **AWS Lambda**, stores it in **Amazon S3**, and serves it through an interactive **Streamlit dashboard** with a built-in **Machine Learning** predictor.

### Key questions answered
- Which operators accumulate the most delay minutes — and why?
- How did COVID-19 impact UK rail punctuality?
- Are delays caused by Network Rail or Train Operating Companies?
- Which UK stations have the worst punctuality and cancellation rates?
- Can ML predict whether an operator will underperform next quarter?

---

## 🏗 Architecture (V1)

```
 ORR Data Portal
       │
       ▼ manual download (future: automated scraper)
 ┌─────────────────────────────────────────────────┐
 │              Amazon S3                          │
 │  s3://uk-train-delay-abu/                       │
 │    ├── raw/        ← ORR Excel/ODS files        │
 │    ├── processed/  ← cleaned CSVs               │
 │    └── models/     ← trained ML model           │
 └──────────────┬──────────────────────────────────┘
                │ S3 PutObject trigger
                ▼
 ┌──────────────────────────────┐
 │       AWS Lambda             │
 │  uk-train-delay-pipeline     │
 │  • Detects which table       │
 │  • Runs cleaner in-memory    │
 │  • Writes CSV → processed/   │
 │  • Logs to CloudWatch        │
 └──────────────────────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │     Local SQLite / RDS*      │  ← *RDS in V2
 │     delays.db                │
 └──────────────────────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │    Streamlit Dashboard       │
 │    6 interactive pages       │
 │    Live ML predictor         │
 └──────────────────────────────┘
```

### Roadmap
| Version | Status | Stack addition |
|---------|--------|---------------|
| **V1** | ✅ Live | AWS S3 + Lambda auto-pipeline |
| V2 | 🔜 Next | Amazon RDS PostgreSQL |
| V3 | 🔜 Planned | Docker + AWS ECS (always-live dashboard) |
| V4 | 🔜 Planned | AWS Glue + Athena (serverless data lake) |

---

## 📊 Data Sources

Free official data from [dataportal.orr.gov.uk](https://dataportal.orr.gov.uk):

| Table | Description | Rows |
|-------|-------------|------|
| **3184** | Delay minutes by operator and cause (periodic) | 2,140 |
| **3133** | Quarterly punctuality by operator (2014–2025) | 1,095 |
| **3130** | Time-to-3 and cancellations by station | 101,205 |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud storage | **Amazon S3** |
| Serverless compute | **AWS Lambda** (auto-triggered on upload) |
| Monitoring | **AWS CloudWatch** |
| Data cleaning | Python · pandas · openpyxl · odfpy |
| Database | SQLite (V1) → Amazon RDS (V2) |
| Analysis | pandas · matplotlib · seaborn |
| Machine Learning | scikit-learn · logistic regression · time-based validation |
| Dashboard | Streamlit · Plotly |

---

## 🗂 Project Structure

```
uk-train-delay-analytics/
├── config.py                   ← Central config (bucket name, paths)
├── scripts/
│   ├── s3_utils.py             ← S3 upload/download helpers (boto3)
│   ├── upload_to_s3.py         ← One-time S3 seeding script
│   ├── clean_data.py           ← Local cleaning pipeline
│   ├── load_to_sql.py          ← Loads CSVs → SQLite
│   ├── analysis.py             ← Generates 6 charts
│   ├── features.py             ← Shared feature builder (training + dashboard)
│   ├── insights.py             ← All dashboard calculations and sentences (unit-tested)
│   └── ml_model.py             ← Trains and evaluates the risk model
├── lambda/
│   ├── lambda_function.py      ← AWS Lambda handler
│   └── deploy_lambda.sh        ← One-command deploy script
├── app/
│   └── streamlit_app.py        ← Interactive dashboard (6 pages)
├── data/                       ← Raw + processed files (gitignored, stored in S3)
├── database/delays.db          ← SQLite DB (committed so the dashboard runs out of the box)
├── models/                     ← risk_model.joblib + metrics.json
├── visuals/                    ← 7 generated charts
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- AWS account (Free Tier is sufficient)
- AWS CLI configured (`~/.aws/credentials`)

### 1. Clone & install
```bash
git clone https://github.com/AbubakarShaikh00/uk-train-delay-analytics.git
cd uk-train-delay-analytics

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure AWS
Edit `config.py` and set your S3 bucket name, or set the environment variable:
```bash
export S3_BUCKET=your-bucket-name
```

### 3. Download ORR data
Download Tables 3184, 3133, 3130 from [dataportal.orr.gov.uk](https://dataportal.orr.gov.uk) into `data/raw/`.

### 4. Run locally
```bash
python scripts/clean_data.py    # Clean the raw files
python scripts/load_to_sql.py   # Load into SQLite
python scripts/analysis.py      # Generate charts
python scripts/ml_model.py      # Train ML model
streamlit run app/streamlit_app.py   # Launch dashboard
```

### 5. Deploy to AWS
```bash
# Seed S3 with existing files
python scripts/upload_to_s3.py

# Deploy Lambda (requires AWS CLI)
bash lambda/deploy_lambda.sh
```

After deploy, any new ORR file uploaded to `s3://your-bucket/raw/` automatically triggers the full cleaning pipeline.

---

## 📈 Key Findings (all computed from the data in this repo)

- **Network Rail causes about 61% of delay minutes**; the three biggest causes are network management, non-track assets and external events.
- **Northern, Govia Thameslink and Great Western** have the most total delay minutes — but they also run the most trains, so the dashboard also shows delay per 1,000 stops for a fairer comparison.
- **Rail is strongly seasonal:** since April 2022, Oct–Dec averages about 76% on-time (3 min) against 81–83% in other quarters.
- **COVID cut delay minutes by more than half** (about 1.3M per 4-week period in 2019-20 versus 0.5M in 2020-21); they have since recovered to roughly 2019-20 levels.
- **In Oct–Dec 2025**, Avanti West Coast (51.8%), CrossCountry and TransPennine Express were weakest, while c2c, Greater Anglia and Merseyrail were strongest at about 90%.

![Delay trend](visuals/2_delay_trend.png)
![Operator punctuality](visuals/5_operator_punctuality_trend.png)

---

## 🤖 Risk Model

**Question:** given a full quarter of data, will an operator's on-time % (within 3 min) fall **below 80%** next quarter?

**Evaluation:** trained on data up to Oct–Dec 2022, tested only on **future** quarters (Apr–Jun 2023 to Jul–Sep 2025, 240 operator-quarters) with a one-quarter gap so no labels leak.

| Model | ROC AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Baseline: same as this quarter | 0.960 | 0.872 | 0.88 | 0.86 |
| Baseline: same quarter last year | 0.986 | 0.924 | 0.90 | 0.94 |
| Random forest | 0.981 | 0.922 | 0.88 | 0.97 |
| **Logistic regression (deployed)** | 0.991 | 0.942 | 0.93 | 0.95 |

**Honest reading:** rail punctuality is so seasonal that "same quarter last year" is already a strong baseline. The model matches or slightly beats it, so it is an early-warning signal rather than a breakthrough.

**Overconfidence check:** the first version gave extreme forecasts (under 2% or over 98%) for 48% of test cases. Those were mostly right in testing, but 240 correlated cases cannot prove certainty. I re-tuned the regularisation on past data only (extremes now 12%), cap displayed forecasts at 5–95%, and show past accuracy next to every forecast. The trade-off is mild under-confidence in the middle ranges (documented in the app).

![Model results](visuals/7_ml_model_results.png)

---

## 🧭 Design principles (v4)

The app is built for a commuter with no data background:
- **One question, one answer:** pick an operator (optional station, quarter) and get a traffic-light answer card with words, not just colour.
- **Every change is explained:** a plain-English "What changed" line, a "Why this number?" panel (source, period, method) and, for forecasts, a confidence level with past accuracy.
- **Nothing silently dropped or filled in:** missing data gets an explicit reason; cleaning exclusions are listed on the *Data and method* page.
- **Shareable:** selections live in the URL, so a result can be re-opened identically.
- **Calculations are separate from the UI** (`scripts/insights.py`) so they can be unit-tested.

## ✅ Tests

```bash
pip install -r requirements.txt pytest
python -m pytest tests -q
```

58 tests: hand-checked unit tests for every calculation (on-time %, quarter-on-quarter change, cause shares, forecast features), calibration guards for the model, and end-to-end tests that drive the running app and check the "What changed" text against numbers computed independently from the database. They run on every push via GitHub Actions.

---

## 📜 License

MIT — free to use, modify, distribute.

## 🙏 Data Attribution

Data from the **Office of Rail and Road (ORR)** under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
