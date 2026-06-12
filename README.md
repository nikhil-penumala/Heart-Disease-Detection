# 🫀 NeuroBee – AI-Powered Heart Disease Prediction System

NeuroBee is an intelligent healthcare platform that combines Machine Learning, Data Analytics, and Conversational AI to predict the risk of heart disease and provide personalized cardiovascular health guidance.

The system analyzes patient clinical data, estimates the probability of heart disease, visualizes the risk level through an interactive dashboard, and offers AI-powered assistance for heart-health-related queries.

---

# 🚀 Features

## 🧠 Heart Disease Prediction

* Predicts cardiovascular disease risk using Machine Learning.
* Supports trained Scikit-Learn models (`.pkl` files).
* Generates probability-based risk scores.
* Provides risk categorization and health recommendations.

## 📊 Interactive Risk Analysis Dashboard

* Animated risk score gauge.
* Real-time prediction results.
* Risk classification:

  * 🟢 Low Risk
  * 🟠 Moderate Risk
  * 🔴 High Risk
* Personalized precaution suggestions.

## 🤖 CardioAI Assistant

* AI-powered cardiac health assistant.
* Explains prediction results.
* Answers heart-health-related questions.
* Provides information about:

  * Heart disease symptoms
  * Cholesterol management
  * Blood pressure control
  * Heart-healthy diets
  * Exercise recommendations
  * Stress management

## 🎨 Modern Healthcare Interface

* Animated ECG welcome screen.
* Professional healthcare-themed dashboard.
* Responsive design.
* Interactive visualizations.

## 🔄 Automatic Fallback Prediction

* Automatically switches to Approximation Mode if trained model files are unavailable.
* Ensures uninterrupted application functionality.
* Useful for demonstrations and development.

---

## 📂 Dataset Information

The model is trained using the **Heart Disease Dataset (heart.csv)**, which contains patient cardiovascular health indicators used for heart disease prediction.

### Dataset Features

| Feature | Description |
|----------|------------|
| Age | Age of the patient |
| Sex | Gender of the patient (Male/Female) |
| ChestPainType | Type of chest pain experienced |
| RestingBP | Resting blood pressure (mmHg) |
| Cholesterol | Serum cholesterol level (mg/dL) |
| FastingBS | Fasting blood sugar (1 = High, 0 = Normal) |
| RestingECG | Resting electrocardiogram results |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of the peak exercise ST segment |
| HeartDisease | Target variable indicating heart disease presence |

### Target Variable

| Value | Meaning |
|---------|----------|
| 0 | No Heart Disease |
| 1 | Heart Disease Present |

Dataset File:

heart.csv

## Target Variable

| Value | Meaning               |
| ----- | --------------------- |
| 0     | No Heart Disease      |
| 1     | Heart Disease Present |

Dataset File:

```text
heart.csv
```

---

# 🏗️ System Architecture

```text
Patient Clinical Data
          │
          ▼
Feature Processing
          │
          ▼
Machine Learning Model
          │
          ▼
Risk Probability Prediction
          │
          ├────────► Risk Analysis Dashboard
          │
          ▼
CardioAI Assistant
          │
          ▼
Personalized Health Guidance
```

---

# 🛠️ Technology Stack

## Backend

* Python
* Flask
* NumPy
* Scikit-Learn
* Pickle

## Frontend

* HTML5
* CSS3
* JavaScript

## AI Integration

* OpenRouter API
* Claude AI

## Machine Learning

* Classification Algorithms
* Feature Scaling
* Probability-Based Prediction

---

# 📁 Project Structure

```text
NeuroBee/
│
├── neurobee_web.py
├── heart.csv
├── heart_model.pkl
├── heart_scaler.pkl
├── README.md
│
└── assets/
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/yourusername/NeuroBee.git

cd NeuroBee
```

## Install Dependencies

```bash
pip install flask scikit-learn numpy requests
```

---

# 🧠 Prediction Modes

NeuroBee supports two prediction modes.

## ✅ Real Model Mode

Uses trained Machine Learning artifacts:

```text
heart_model.pkl
heart_scaler.pkl
```

Features:

* Higher prediction accuracy
* Learns patterns from the dataset
* Recommended for production use

Dashboard Status:

```text
● REAL MODEL
```

---

## ⚠️ Approximation Mode

If the trained model or scaler cannot be loaded, NeuroBee automatically switches to a built-in approximation engine.

Features:

* No application downtime
* Automatic fallback mechanism
* Useful for demonstrations
* Useful during development and testing

Dashboard Status:

```text
● APPROX MODE
```

---

# ▶️ Running the Application

Update the file paths:

```python
MODEL_PATH = "heart_model.pkl"
SCALER_PATH = "heart_scaler.pkl"
```

Run the application:

```bash
python neurobee_web.py
```

Open:

```text
http://localhost:5000
```

---

# 📊 Risk Categories

| Risk Score    | Category      |
| ------------- | ------------- |
| Below 35%     | Low Risk      |
| 35% – 64%     | Moderate Risk |
| 65% and Above | High Risk     |

---

# 🎯 Applications

* Preventive Healthcare
* Clinical Decision Support Systems
* Medical Education
* Healthcare Analytics
* Telemedicine Platforms
* AI-Assisted Patient Monitoring

---

# 🔮 Future Enhancements

* ECG Signal Analysis
* Medical Report Upload and Analysis
* Explainable AI (XAI)
* Cloud Deployment
* Wearable Device Integration
* Multi-Disease Prediction
* Voice-Based Health Assistant
* Electronic Health Record Integration

---

# 👥 Team

| Name            | Role                                             |
| --------------- | ------------------------------------------------ |
| Nikhil Penumala | Machine Learning, Backend Development, UI Design |
| Hamsini Kummara   | Frontend Development                             |
| Chaithanya Puttur  | AI Integration & Testing                         |

---

# ⚠️ Disclaimer

This project is developed for educational, research, and demonstration purposes only.

The predictions generated by NeuroBee are not intended to replace professional medical diagnosis, treatment, or clinical decision-making. Users should always consult qualified healthcare professionals for medical advice.


