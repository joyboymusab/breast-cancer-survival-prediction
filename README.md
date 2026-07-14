# 🩺 Breast Cancer Survivability Prediction using Machine Learning

A Machine Learning web application that predicts breast cancer survivability based on clinical patient data. The application was developed using **Python**, **Scikit-learn**, and **Streamlit** to provide an interactive interface for healthcare professionals and researchers.

> **Disclaimer:** This application is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## 📌 Project Overview

Breast cancer remains one of the leading causes of cancer-related deaths worldwide. Early prediction of patient survivability can help healthcare professionals better understand patient risk factors and support treatment planning.

This project develops a predictive Machine Learning model that estimates whether a patient is likely to survive based on clinical features collected from breast cancer records.

The application includes:

* Interactive Streamlit web interface
* Machine Learning prediction model
* Probability estimation
* Dynamic radar chart visualization of patient features
* Automated preprocessing pipeline

---

## 🚀 Features

* Predicts patient survivability (Alive / Dead)
* Displays prediction confidence (probabilities)
* Interactive feature selection using sliders and dropdown menus
* Radar chart visualization of patient characteristics
* Automatic preprocessing and feature encoding
* Simple and user-friendly interface

---

## 🧠 Machine Learning Model

**Algorithm**

* Logistic Regression

**Preprocessing**

* Missing value handling
* Feature encoding
* One-hot encoding
* StandardScaler normalization
* Numerical conversion of categorical variables

**Train/Test Split**

* 80% Training
* 20% Testing

---

## 📊 Input Features

The model predicts survivability using clinical information including:

* Age
* Tumor Size
* Tumor Stage (T Stage)
* Node Stage (N Stage)
* AJCC 6th Stage
* Tumor Differentiation
* Grade
* A Stage
* Estrogen Receptor Status
* Progesterone Receptor Status
* Regional Nodes Examined
* Regional Nodes Positive
* Survival Months
* Race
* Marital Status

---

## 📈 Output

The application provides:

* Predicted Status (Alive / Dead)
* Probability of Survival
* Probability of Death
* Radar chart summarizing patient features

---

## 🛠 Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Plotly
* Pickle

---

## 📂 Project Structure

```
Breast-Cancer-Survivability-Prediction/
│
├── app.py
├── train_model.py
├── model/
│   ├── model.pkl
│   └── scaler.pkl
├── Breast_Cancer.csv
├── requirements.txt
├── README.md
└── images/
    ├── home.png
    └── prediction.png
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Breast-Cancer-Survivability-Prediction.git
```

Move into the project directory

```bash
cd Breast-Cancer-Survivability-Prediction
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

Add screenshots of your application here.

Example:

* Home Page
* Patient Feature Selection
* Radar Chart
* Prediction Results

---

## 📚 Dataset

This project uses a publicly available breast cancer clinical dataset containing patient demographic information, tumor characteristics, receptor status, lymph node information, and survival outcomes.

---

## 💡 Future Improvements

* Deep Learning models
* XGBoost and Random Forest comparison
* SHAP Explainable AI visualization
* Cloud deployment
* Model comparison dashboard
* Survival analysis using Kaplan–Meier curves
* Feature importance visualization

---

## 👨‍💻 Author

**Musab Mahmoud Abukalam**

Biomedical Engineer

Interested in Machine Learning, Bioinformatics, and AI for Healthcare.

GitHub: github.com/joyboymusab
LinkedIn: linkedin.com/in/musab-mahmoud-abukalam
