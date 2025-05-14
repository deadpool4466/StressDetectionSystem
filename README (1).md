
# Real-Time Personalized Physiologically Based Stress Detection for Hazardous Operations

## 🧠 Overview

This project focuses on real-time stress detection using physiological signals and machine learning for hazardous operational environments. Unlike generalized models, it uses **personalized features** and **time-series analysis** to improve accuracy, aiming to detect stress levels (Low, Medium, High) in real-time, especially in high-stakes virtual training scenarios like spaceflight emergencies.

## 🚀 Features

- Real-time stress level prediction (Low, Medium, High)
- Personalized machine learning model training
- Time-series windowing for physiological signal analysis
- VR & N-back task simulation for data collection
- Visualization with accuracy charts
- Role-based access for System Operators and Remote Users

## 🧩 Modules

### 🛠 System Operator
- Login and manage datasets
- Train/test models
- View prediction stats
- Monitor user stress status
- Download datasets
- View system-wide results

### 👨‍💻 Remote Access User
- Register/Login
- Predict stress status
- View personal profiles and stress reports

## 🧪 Technologies Used

| Layer        | Tech Stack                   |
|-------------|------------------------------|
| Frontend     | HTML, CSS, JavaScript        |
| Backend      | Python, Django ORM           |
| Database     | MySQL (via WAMP Server)      |
| Others       | Matplotlib, Scikit-learn     |

## 💻 System Requirements

### Hardware
- Intel i3 CPU or above
- 512 MB RAM (minimum)
- 40 GB HDD
- Monitor (15" VGA), Mouse, Keyboard

### Software
- Windows 7 Ultimate
- Python (3.x)
- Django Framework
- MySQL Server (WAMP)
- Web Browser

## 🧬 Input & Output

### Input
- Physiological parameters like Heart Rate, Blood Pressure, EDA, Respiration
- User profiles and task-specific stress indicators

### Output
- Stress classification result
- Accuracy metrics
- Bar charts and ratio visualizations

## 📊 Architecture

- Web-based client-server model
- Role-based interaction
- Time-series processing pipeline
- Modular MVC-based Django backend

## 📚 Key Algorithms

- Approximate Bayes Classifier (ABayes)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

## 📉 Performance Metrics

- Accuracy
- Cross-validation scores
- Window-size impact analysis
- Feature selection dynamics

## 📌 How to Run

1. Install Python and dependencies (`pip install -r requirements.txt`)
2. Set up MySQL and configure in `settings.py`
3. Run Django server:
   ```bash
   python manage.py runserver
   ```
4. Open browser and navigate to `http://localhost:8000`

## 🧪 Results

- Personalized models outperformed generalized ones
- ABayes achieved superior accuracy over traditional classifiers
- Blood pressure was consistently the most predictive feature

## 📘 References

- Research papers on stress detection using physiological data
- Machine learning algorithms for time-series health data
- VR-based emergency training studies

## 📎 License

This project is part of an academic research submission and is intended for educational purposes only.
