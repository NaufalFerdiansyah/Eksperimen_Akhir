# 📊 Capstone Project Data Mining  
## Prediksi Customer Churn Menggunakan Machine Learning

Proyek ini merupakan **Capstone Project Ujian Akhir Semester Mata Kuliah Data Mining** yang bertujuan untuk membangun solusi Machine Learning end-to-end dalam memprediksi **customer churn** pada industri telekomunikasi.  
Model terbaik kemudian diimplementasikan ke dalam **aplikasi web interaktif berbasis Streamlit** agar dapat digunakan oleh pengguna non-teknis.

---

## 📌 Problem Statement
Customer churn adalah kondisi ketika pelanggan berhenti menggunakan layanan suatu perusahaan. Tingginya churn rate dapat menyebabkan penurunan pendapatan dan meningkatnya biaya akuisisi pelanggan baru.  
Proyek ini berfokus pada:
- Memprediksi kemungkinan churn pelanggan
- Mengidentifikasi faktor utama penyebab churn
- Menyajikan hasil analisis dan prediksi melalui dashboard interaktif

---

## 📂 Dataset
- **Nama Dataset:** Telco Customer Churn  
- **Sumber:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
- **Jumlah Data:** 7.043 baris  
- **Jumlah Fitur:** 21 fitur  
- **Target:** `Churn` (Yes / No)

---

## 🛠️ Tools & Libraries
- **Bahasa Pemrograman:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Interpretation:** SHAP
- **Deployment:** Streamlit
- **Model Serialization:** Joblib / Pickle

---

## 🧠 Metodologi
Proyek ini mengikuti alur kerja **CRISP-DM**, yang meliputi:
1. Business Understanding
2. Data Understanding
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing & Feature Engineering
5. Modeling & Hyperparameter Tuning
6. Model Evaluation
7. Model Interpretation (SHAP)
8. Deployment menggunakan Streamlit

---

## 📊 Model yang Digunakan
- Logistic Regression (Baseline)
- Random Forest Classifier
- **XGBoost Classifier (Model Terbaik)**

**Metrik Evaluasi:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 🚀 Deployment (Streamlit App)
Aplikasi Streamlit menyediakan fitur:
- **Dashboard EDA**
- **Prediksi Churn Pelanggan**
- **Evaluasi Model**
- **Interpretasi Model (SHAP)**
- **Dokumentasi Proyek**

🔗 **Link Aplikasi Streamlit:**  
(https://churn-prediction-9ux2j6z4asmnlvqfucjaqd.streamlit.app/)

---

## 🗂️ Struktur Repository
capstone-project-data-mining/
│
├── data/
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_modeling.ipynb
│ └── 03_interpretation.ipynb
│
├── src/
├── models/
│ ├── best_model.pkl
│ └── preprocessing.pkl
│
├── app/
│ ├── app.py
│ ├── pages/
│ └── assets/
│
├── reports/
│ ├── final_report.pdf
│ └── presentation.pptx
│
├── requirements.txt
├── README.md
└── .gitignore
