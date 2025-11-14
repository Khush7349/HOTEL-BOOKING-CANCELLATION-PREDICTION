# 🏨 Hotel Booking Cancellation Prediction — Machine Learning Classification Project

## 📘 Project Overview
This project builds an end-to-end **machine learning pipeline** to predict **hotel booking cancellations** based on customer reservation details, pricing information, stay characteristics, and booking behavior.  
The model uses structured reservation data from a publicly available **Hotel Reservations Dataset** and applies advanced ML techniques to ensure **high accuracy, interpretability, and real-world usability**.

The goal is to assist hotels in improving **revenue management**, reducing **overbooking risks**, and understanding **key drivers behind cancellations**.

---

## 📂 Dataset
The dataset used in this project comes from the **Hotel Reservations Dataset on Kaggle**:

🔗 https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

| File | Description |
|------|-------------|
| `hotel_reservations.csv` | Main dataset containing booking, customer, stay, and pricing information |

---

## 🔍 Project Workflow

This project follows the full lifecycle of a real-world machine learning classification task:

1️⃣ **Data Loading & Cleaning** — Handles missing values, removes duplicates, verifies dates, and standardizes formats.  
2️⃣ **Feature Engineering** — Creates new predictors such as `total_stay`, `price_per_person`, `lead_time_bucket`, `arrival_month_name`, `is_family_trip`, and more.  
3️⃣ **Exploratory Data Analysis (EDA)** — Examines data patterns using insightful dashboards focusing on price behavior, seasonality, stay durations, and cancellation trends.  
4️⃣ **Feature Encoding & Model Training** — Uses `ColumnTransformer` pipelines for scaling and OneHotEncoding on mixed data types.  
5️⃣ **Model Comparison** — Evaluates several classifiers:  
   - Logistic Regression  
   - Random Forest  
   - Gradient Boosting  
   - KNN  
   - **LightGBM (Best Model)**  
6️⃣ **Model Calibration & Threshold Tuning** — Improves probability reliability and selects an optimal decision threshold for business needs.  
7️⃣ **Feature Importance & Insights** — Identifies the strongest drivers of cancellation risk.

---

## 🧠 Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|-----------|-----------|----------|
| Logistic Regression | 0.842 | 0.801 | 0.883 |
| Random Forest | 0.872 | 0.824 | 0.918 |
| Gradient Boosting | 0.865 | 0.816 | 0.907 |
| KNN | 0.793 | 0.740 | 0.828 |
| **LightGBM (Best)** | **0.894** | **0.835** | **0.953** |

- **Best Model:** LightGBM  
- **Key Predictors:** Lead Time, Average Room Price, Arrival Month, Total Stay, Special Requests  
- **Insights:**  
  - Longer **lead times** significantly increase cancellation likelihood.  
  - Higher **room prices** correlate with price sensitivity and cancellations.  
  - **Family-oriented bookings** typically show lower cancellation rates.  
  - Certain **months and segments** display higher cancellation risk patterns.

---

## 📊 Visualizations

The project includes a rich set of visual dashboards:

- Lead time distribution  
- Price and stay duration trends  
- Cancellation rates across months and segments  
- Correlation heatmap  
- PCA scatter visualization  
- Jitter-based scatter insights  
- ROC and PR curves  
- Feature importance rankings  

All visuals are clean, intuitive, and business-ready.

---

## ⚙️ Tools & Technologies Used

- **Language:** Python 3.11  
- **Libraries:**  
  - Pandas, NumPy — Data manipulation  
  - Matplotlib, Seaborn — Visualization  
  - Scikit-learn — Modeling, evaluation, preprocessing  
  - LightGBM — Final optimized model  
  - Joblib — Saving and loading trained models  

---

## 🔮 Insights & Future Enhancements

### ✅ Insights
- **Lead time** is the most powerful indicator of cancellation probability.  
- Price-driven cancellations suggest the need for dynamic pricing strategies.  
- Seasonal patterns can help hotels forecast and prepare for cancellation spikes.  
- Model calibration improves real-world utility by making probability outputs reliable.

### 🚀 Future Work
- Build a **Streamlit app** for interactive cancellation prediction.  
- Integrate **SHAP** for model explainability.  
- Add real-time booking data from APIs.  
- Explore **CatBoost** or **AutoML** for further accuracy improvements.  
- Implement additional **fairness and bias checks** across customer groups.

---

## 👩‍💻 Author
**Khushi Sharma**  
*Data Science & Analytics Enthusiast*  
Building practical machine learning solutions using Python, analytics, and interpretable AI.

---

## 📂 Repository Structure

```
hotel-booking-cancellation-prediction-ml/
│
├── data/
│   └── hotel_reservations.csv
│
├── notebooks/
│   └── final_notebook.ipynb
│
├── models/
│   └── lgbm_model.joblib
│
├── README.md
└── requirements.txt
```
