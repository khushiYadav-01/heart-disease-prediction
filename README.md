# ❤️ Heart Disease Prediction using K-Nearest Neighbors (KNN)

## 📌 Project Overview

This project is a **Machine Learning-based Heart Disease Prediction System** developed using **Python**, **Scikit-learn**, and **Streamlit**. The application predicts whether a person is likely to have heart disease based on clinical parameters.

The model has been trained using the **K-Nearest Neighbors (KNN)** algorithm and deployed as an interactive web application using **Streamlit**.

---

## 🚀 Features

- Predicts heart disease risk using Machine Learning
- Interactive Streamlit web interface
- User-friendly input form
- Fast prediction
- Pre-trained KNN model
- Feature scaling using StandardScaler

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Structure

```
heart-disease-prediction/
│── app.py
│── KNN_heart.pkl
│── Scaler.pkl
│── columns.pkl
│── README.md
```

---

## 📊 Input Features

The model uses the following features:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

---

## 🤖 Machine Learning Model

**Algorithm Used:**

- K-Nearest Neighbors (KNN)

The model was trained after:

- Data Cleaning
- Handling Missing Values
- Feature Encoding
- Feature Scaling
- Model Training
- Model Evaluation

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **86.41%** |
| F1 Score | **88.15%** |

---

## ▶️ Run the Project

### Clone the repository

```bash
git clone https://github.com/khushiYadav-01/heart-disease-prediction.git
```

### Move to the project folder

```bash
cd heart-disease-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

*(Add a screenshot of your Streamlit application here.)*

---

## 📚 Future Improvements

- Improve prediction accuracy
- Add more ML algorithms for comparison
- Deploy on Streamlit Cloud
- Add data visualization dashboard
- Improve UI/UX

---

## 👩‍💻 Author

**Khushi Yadav**

GitHub: https://github.com/khushiYadav-01

---

## ⭐ If you found this project useful, don't forget to Star this repository!
