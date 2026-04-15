# ⚡ AI Energy Consumption Forecasting System

## 📌 Overview
This project builds an AI-powered system to forecast energy consumption using time-series machine learning techniques. It includes a full pipeline from data preprocessing to model deployment with an interactive dashboard.

---

## 🎯 Problem Statement
Energy demand fluctuates based on time, usage patterns, and external factors. Without accurate forecasting, organizations face inefficiencies, higher costs, and energy wastage.

This project aims to predict future energy consumption using historical data to support better decision-making.

---

## 🏢 Industry Relevance
- ⚡ Power Grid Optimization
- 🏭 Manufacturing Energy Planning
- 🖥️ Data Center Load Management
- 🌱 Renewable Energy Balancing
- 🏙️ Smart Cities Infrastructure

---

## 🧠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## 📊 Dataset
- Household Power Consumption Dataset (UCI)
- Contains minute-level energy usage data
- Converted to hourly time-series

---

## 🏗️ Project Architecture


Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Forecasting → Visualization/UI


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI-Energy-Consumption-Forecasting-System.git
cd AI-Energy-Consumption-Forecasting-System

pip install -r requirements.txt

▶️ Usage

Run ML Pipeline
python main.py

Run Dashboard
streamlit run app.py




📈 Results
Model	RMSE	R²
Linear Regression	~0.52	~0.52
Random Forest	~0.47	~0.60


📊 Screenshots


🔹 Dashboard UI

🔹 Actual vs Predicted

🔹 Forecast Output



🧠 Key Learnings
Importance of time-series feature engineering
Handling data leakage in ML models
Role of lag and rolling features
Model evaluation and interpretation
Building end-to-end ML systems with UI




🚀 Future Improvements
Add weather data integration
Use LSTM/Deep Learning models
Deploy on cloud (AWS / Streamlit Cloud)
Real-time energy monitoring system

💡 Conclusion
This project demonstrates how machine learning can be used to forecast energy consumption and simulate real-world decision-making systems.


## Dataset

The dataset is not included due to size limitations.

Download it from:
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Place it in:
data/raw/household_power_consumption.txt


👨‍💻 Author

Aniket


---

# 📅 1️⃣2️⃣ STEP-BY-STEP GITHUB PROOF PLAN

---

## 🟢 Day 1 — Setup

Commit:
```text
Project setup and folder structure
🟢 Day 2 — Dataset

Commit:

Added dataset and data loading module
🟢 Day 3 — Preprocessing

Commit:

Implemented preprocessing pipeline
🟢 Day 4 — Model

Commit:

Added feature engineering and model training
🟢 Day 5 — Evaluation

Commit:

Added evaluation metrics and forecasting
🟢 Day 6 — Visualization

Commit:

Added visualization and graphs
🟢 Day 7 — UI + Upload

Commit:

Added Streamlit UI and final project upload
