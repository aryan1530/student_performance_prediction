# 🎓 Student Performance Prediction System

A Machine Learning-powered web application that analyzes student-related factors and predicts academic performance using both regression and classification techniques. The project demonstrates a complete end-to-end ML workflow, including data generation, preprocessing, model training, evaluation, visualization, and deployment through a user-friendly web interface.

---

## 📌 Project Overview

Educational institutions often need insights into factors affecting student performance. This project leverages Machine Learning algorithms to predict student outcomes based on various academic and behavioral attributes.

The system provides:

* **Score Prediction** using Linear Regression
* **Pass/Fail Classification** using Logistic Regression
* Interactive visualizations and performance metrics
* Web-based prediction interface built with Flask
* Ready-to-deploy structure for cloud platforms

---

## 🚀 Features

### 📊 Data Generation

* Synthetic student dataset generation
* Realistic academic and behavioral attributes
* Automated CSV dataset creation

### 🔍 Exploratory Data Analysis (EDA)

* Data visualization and insights
* Correlation analysis
* Distribution plots
* Performance trend analysis

### 🤖 Machine Learning Models

#### Linear Regression

Predicts the expected student score based on input features.

**Performance**

* R² Score: **0.84**

#### Logistic Regression

Classifies whether a student is likely to pass or fail.

**Performance**

* Accuracy: **98%**

### 💾 Model Persistence

* Trained models saved using Pickle
* Easy loading for real-time predictions

### 🌐 Web Application

* User-friendly interface
* Real-time predictions
* Lightweight Flask backend

### ☁️ Deployment Ready

* GitHub integration
* Render deployment support
* Procfile included

---

## 🏗️ Project Structure

```text
student-performance-prediction/
│
├── data/
│   ├── generate_data.py
│   └── student_data.csv
│
├── src/
│   └── model.py
│
├── models/
│   └── *.pkl
│
├── outputs/
│   └── plots & reports
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

---

## ⚙️ Technologies Used

### Programming Language

* Python 3.x

### Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Flask
* Joblib / Pickle

### Deployment

* GitHub
* Render

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git

cd student-performance-prediction
```

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Generate Dataset

Run the dataset generation script:

```bash
python data/generate_data.py
```

This will create:

```text
data/student_data.csv
```

---

## 🧠 Train the Models

Execute:

```bash
python src/model.py
```

The training pipeline will:

1. Load dataset
2. Preprocess data
3. Perform EDA
4. Train ML models
5. Evaluate performance
6. Save trained models
7. Generate visualizations

Generated files will be stored in:

```text
models/
outputs/
```

---

## 🌐 Run the Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser:

```text
http://localhost:5000
```

Enter student details and receive instant predictions.

---

## 📈 Model Performance

| Model               | Task                     | Performance    |
| ------------------- | ------------------------ | -------------- |
| Linear Regression   | Score Prediction         | R² = 0.84      |
| Logistic Regression | Pass/Fail Classification | Accuracy = 98% |

---

## 📷 Outputs

The project automatically generates:

* Correlation Heatmaps
* Feature Distributions
* Prediction Visualizations
* Model Evaluation Graphs

All outputs are saved inside:

```text
outputs/
```

---

## 🔮 Future Enhancements

* Random Forest Regressor
* XGBoost Integration
* Student Performance Dashboard
* Database Connectivity
* User Authentication
* Real-Time Analytics
* Model Monitoring

---

## 🎯 Learning Outcomes

This project demonstrates:

* Data Generation
* Data Cleaning & Preprocessing
* Exploratory Data Analysis
* Regression Modeling
* Classification Modeling
* Model Evaluation
* Model Deployment
* Flask Web Development
* End-to-End Machine Learning Pipeline

---

## 👨‍💻 Author

Developed as an end-to-end Machine Learning project to showcase practical implementation of predictive analytics in education.

If you found this project useful, consider giving it a ⭐ on GitHub.
