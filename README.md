# 🩺 Chronic Kidney Disease (CKD) Prediction App

This is a **Flask web application** that predicts the risk of **Chronic Kidney Disease** using a trained machine learning model.  
The app takes basic patient data as input, scales it, and runs a prediction to determine whether CKD is detected.

---

## 📌 Features
- **Web Interface** for input (age, blood pressure, specific gravity, albumin, sugar)
- **Machine Learning Model** for prediction (saved with `joblib`)
- **JSON Response** for easy frontend integration
- **Probability & Risk Score** displayed

---

## 🛠️ Requirements

Before running the project, make sure you have the following installed:

- Python 3.8+
- Flask
- NumPy
- scikit-learn
- joblib

Install dependencies with:

```bash
pip install -r requirements.txt
