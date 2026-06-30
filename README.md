# GlucoFit-Ai
# 🩺 GlucoFit-AI

An AI-powered Diabetes Risk Prediction and Health Recommendation System that combines Machine Learning, Lifestyle Analysis, Nutrition Intelligence, and Personalized Healthcare Guidance.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview

Diabetes is one of the fastest-growing chronic diseases worldwide. Early detection can significantly reduce long-term health complications.

**GlucoFit-AI** is an intelligent web-based healthcare application that predicts diabetes risk using Machine Learning and provides:

- Diabetes risk prediction
- Probability-based risk score (Diabetes Meter)
- Lifestyle assessment
- Personalized dietary recommendations
- Nutrition analysis
- AI-powered health insights
- Downloadable health reports

The system enables users to assess their diabetes risk by entering clinical and lifestyle information through an interactive web interface.

---

## 🚀 Features

### 🔍 Diabetes Risk Prediction
- Predicts diabetic/non-diabetic status
- Generates probability-based risk score
- Uses Random Forest Machine Learning model

### 🥗 Nutrition Analysis
- Analyzes user food intake
- Calculates nutritional information
- Provides dietary recommendations

### 🤖 AI-Powered Health Guidance
- Personalized health suggestions
- Risk reduction recommendations
- Dynamic explanation generation using Gemini AI

### 📊 Interactive Dashboard
- User-friendly Streamlit interface
- Real-time predictions
- Health insights visualization

### 📄 PDF Report Generation
- Downloadable health reports
- Prediction summary
- Lifestyle recommendations

### 🔐 Secure Authentication
- Google OAuth integration
- Secure user access
- Privacy-focused design

---
```

---

## 🧠 Machine Learning Pipeline

1. Dataset Collection
2. Data Cleaning
3. Feature Engineering
4. Z-Score Normalization
5. Stratified Train-Test Split
6. Class Imbalance Handling
7. Random Forest Training
8. Prediction & Probability Scoring
9. Recommendation Generation
10. Deployment using Streamlit

---

## 📂 Dataset

### PIMA Indians Diabetes Dataset

Source:
https://archive.ics.uci.edu/ml/datasets/diabetes

Features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target:

- 0 → Non-Diabetic
- 1 → Diabetic

---

## 🛠️ Tech Stack

### Backend
- Python
- Scikit-Learn
- Pandas
- NumPy

### Frontend
- Streamlit
- HTML
- CSS
- JavaScript

### Machine Learning
- Random Forest Classifier
- StandardScaler

### APIs & Integrations
- Google Gemini API
- Edamam Nutrition API
- YouTube Data API
- Google OAuth

### Utilities
- Joblib
- Pillow
- Matplotlib
- FPDF2

---

## 📁 Project Structure

```text
GlucoFit-AI/
│
├── app.py
├── pages/
│   ├── 2_Assessment.py
│
├── diabetes_prediction.py
├── nutrition_analyzer.py
├── youtube_helper.py
│
├── models/
│   ├── diabetes_model.pkl
│   ├── scaler.pkl
│
├── assets/
│   ├── images/
│
├── reports/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/GlucoFit-AI.git
cd GlucoFit-AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📈 Model Performance

| Metric    | Value |
|---------- |--------|
| Accuracy  | ~95% |
| Prediction Type | Binary + Probability |
| Algorithm  | Random Forest |
| Trees      | 200 Estimators |

---

## 📸 Screenshots

### Home Page
<img width="743" height="449" alt="home1" src="https://github.com/user-attachments/assets/1d2867da-e4dd-459b-b0c5-aba97cf304aa" />
<img width="884" height="458" alt="home2" src="https://github.com/user-attachments/assets/d1192693-dc02-40e6-8d6c-47cd5c0cb3db" />

### Personal info Page

<img width="655" height="332" alt="personalinfo" src="https://github.com/user-attachments/assets/13bd946e-96f0-4c38-812e-1c017858b363" />

### lifestyle Report


<img width="651" height="473" alt="lifedietinfo" src="https://github.com/user-attachments/assets/dbf3b945-eac4-4ac9-9223-a63287ec34c0" />



### Prediction Dashboard

<img width="728" height="482" alt="prediction" src="https://github.com/user-attachments/assets/b82f004a-5c74-456d-ad10-397660bfbeec" />

---

## 🔮 Future Scope

- Wearable Device Integration
- Android & iOS Application
- Real-Time Health Monitoring
- Electronic Health Record Integration
- Voice-Based Input
- Cloud Storage Support
- Deep Learning Models
- Multilingual Support

## 👨‍💻 Contributors

- Ritik
- Punit Tanwar
- Umakant

Department of Electronics & Computer Engineering
J.C. Bose University of Science & Technology, YMCA, Faridabad
B.Tech  Project (6th Semester)
