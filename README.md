# Metro Manila Food Price Monitoring & Spike Forecasting (EDA)
**Course**: Intro to Data Science | **Methodology**: CRISP-DM  
**Team**: Group 2 – BSIT 3-5  
**Focal Target**: Well-Milled Rice (Local) in Metro Manila (NCR), Philippines

---

## 📌 Project Overview
This repository contains the **Exploratory Data Analysis (EDA)** and data preprocessing pipeline for forecasting staple food prices in Metro Manila, with a focus on predicting **Well-Milled Rice (Local)** 4 weeks ahead and investigating the association of price spikes with exogenous shocks (typhoons, fuel prices, and import volumes).

### CRISP-DM Research Questions
* **Descriptive**: *How have rice, vegetable, and fish prices moved over time (trends, seasonality, spikes)?*
* **Predictive**: *Can we forecast the Metro Manila rice price 4 weeks ahead accurately enough to flag spikes early?*
* **Prescriptive**: *Which shocks (typhoons, fuel, imports) are most associated with spikes, so the DA and traders know what to watch?*

---

## 📊 Dataset Summary
* **Source**: Department of Agriculture (DA) Bantay Presyo NCR Weekly Price Monitoring (`weekly_prices_ncr.csv`).
* **Granularity**: Weekly retail wet-market prices in Metro Manila.
* **Coverage**: August 21, 2023 to October 26, 2026 (163 reporting weeks, ~3.1 years continuous series).
* **Target Completeness**: 100% complete records (163/163 weeks) for Well-Milled Rice (Local) with average price ₱47.77/kg.

---

## 🔍 Key Findings from the EDA
1. **Spike Detection (>5% in 4 Weeks)**: Exactly 9 spike weeks were identified across two major historical clusters:
   * **November–December 2023**: Market rebound and restocking following the lifting of Executive Order 39 (price ceiling), peaking at $+6.92\%$ 4-week growth.
   * **April 2026**: Lean season transition surge reaching ₱50.45/kg ($+14.79\%$ 4-week growth).
2. **Volatility Asymmetry**:
   * **Rice** is economically "sticky" with low volatility ($\text{CV} = 4.70\%$).
   * **Highland & Lowland Vegetables** exhibit extreme volatility ($\text{CV} = 32.8\%$ and $28.4\%$) driven by perishable shelf life and weather vulnerability.
   * **Fish** shows intermediate volatility ($\text{CV} = 14.2\%–18.1\%$).
3. **Decoupled Shock Transmission**: On short 4-week horizons, rice price changes do not co-move with vegetables and fish ($r \approx -0.19$ to $+0.08$), confirming that rice price spikes are governed by national macro-policy and grain trade dynamics, whereas vegetable spikes are localized weather disruptions.

---

## 📁 Repository Structure
```
├── EDA_Food_Price_Monitoring.ipynb    # Executed Jupyter Notebook with all outputs & charts
├── EDA_Report.md                      # Comprehensive academic EDA report
├── Data Science Project Proposals...  # Approved CRISP-DM project proposal document
├── weekly_prices_ncr.csv              # DA Bantay Presyo NCR weekly market dataset
├── generate_eda.py                    # Script to clean data and generate analytical plots
├── build_notebook.py                  # Script to assemble the Jupyter Notebook structure
├── execute_notebook_inprocess.py      # In-process execution runner for notebook cells
├── images/                            # High-resolution output figures (Fig 1 – Fig 8)
├── .gitignore                         # Standard Python & Jupyter ignore rules
└── README.md                          # Repository documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/benhrnndz/metro-manila-food-price-eda.git
cd metro-manila-food-price-eda
```

### 2. Setup virtual environment & dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scipy statsmodels jupyter nbformat
```

### 3. Run or view the EDA
* Open [`EDA_Food_Price_Monitoring.ipynb`](EDA_Food_Price_Monitoring.ipynb) in VS Code or JupyterLab.
* Or read the detailed findings in [`EDA_Report.md`](EDA_Report.md).
