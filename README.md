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


## 📊 Screenshots

### 🔹 Dashboard UI
![Dashboard](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/Dashboard.png)

### 🔹 Actual vs Predicted (Dashboard View)
![Actual vs Predicted Dashboard](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/actual_vs_predicted_dashboard.png)

### 🔹 Actual vs Predicted (Analysis)
![Actual vs Predicted](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/actual_vs_predicted.png)

### 🔹 Next 24 Hours Forecast
![Next 24Hrs Forecast](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/Next_24Hrs_Forecast.png)

### 🔹 Feature Importance Analysis
![Feature Importance Analysis](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/Feature_importance_Analysis.png)

### 🔹 Dataset Preview
![Dataset Preview](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/dataset_preview.png)

### 🔹 Prediction Output
![Prediction](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/Prediction.png)

### 🎥 Dashboard Demo Video
[![Dashboard Demo](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/Dashboard.png)](https://github.com/Aniketsatpathy/AI-Energy-Consumption-Forecasting-System/raw/main/images/Dashboard_Demo.mp4)
*Click the image above to watch the dashboard demo video*



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
